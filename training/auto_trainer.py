"""Autonomous Self-Training Engine for NumPy-GPT.

Provides continuous background auto-training, self-play / synthetic generation,
adaptive learning rate scheduling, and automatic best-checkpoint persistence.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer
from training.dataset import ChatDataset, TextDataset
from training.optimizer import AdamW
from training.scheduler import CosineWarmupScheduler


class AutoTrainer:
    """Manages autonomous self-training cycles in the background."""

    SELF_PLAY_SEEDS = [
        "Explain the role of Query, Key, and Value vectors in self-attention.",
        "What is the mathematical definition of RMSNorm?",
        "Why are Rotary Position Embeddings (RoPE) superior to learned absolute embeddings?",
        "Write a clean Python function to calculate softmax with numerical stability.",
        "Describe how KV-caching reduces generation complexity from O(T^2) to O(1) per step.",
        "What is the purpose of weight tying between token embeddings and language model head?",
        "Explain how SwiGLU activation improves transformer representation capacity.",
        "How does decoupled AdamW weight decay prevent norm parameter explosion?",
        "What is gradient clipping and why is it necessary for transformer training?",
        "Write a Python function to compute cross entropy loss from logits and targets.",
    ]

    def __init__(
        self,
        model: GPT,
        tokenizer: ByteBPETokenizer,
        dataset: Union[ChatDataset, TextDataset],
        base_lr: float = 0.0003,
        batch_size: int = 2,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        lock: Optional[threading.Lock] = None,
        on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.base_lr = base_lr
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.lock = lock or threading.Lock()
        self.on_step_callback = on_step_callback

        self.optimizer = AdamW(model.params, lr=base_lr, weight_decay=0.05, grad_clip=1.0)
        self.scheduler = CosineWarmupScheduler(
            base_lr=base_lr,
            warmup_steps=20,
            max_steps=2000,
            min_lr=1e-5,
        )

        # Autonomous State
        self.is_running = False
        self.mode = "autonomous_loop"  # "autonomous_loop" or "self_play"
        self.cycle_count = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.val_loss = 0.0
        self.best_val_loss = float("inf")
        self.auto_checkpoints = 0
        self.status_message = "Idle"
        self.history: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.synthetic_pairs_generated = 0

    def start(self, mode: str = "autonomous_loop", target_lr: Optional[float] = None) -> None:
        """Start the background autonomous training loop."""
        if self.is_running:
            return

        self.mode = mode
        if target_lr:
            self.base_lr = target_lr
            self.optimizer.lr = target_lr

        self._stop_event.clear()
        self.is_running = True
        self.status_message = f"Auto-training started ({mode})"
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the auto-trainer to stop cleanly."""
        self._stop_event.set()
        self.is_running = False
        self.status_message = "Auto-training stopped"
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _run_loop(self) -> None:
        """Continuous background execution loop."""
        step_interval = 0.04

        while not self._stop_event.is_set():
            self.cycle_count += 1
            cycle_steps = 10

            for _ in range(cycle_steps):
                if self._stop_event.is_set():
                    break

                self.total_steps += 1
                curr_lr = self.scheduler.get_lr(self.total_steps)

                loss_val = self._train_step(curr_lr)
                self.current_loss = round(float(loss_val), 4)

                # Throttle CPU consumption and allow other threads to acquire model lock
                time.sleep(step_interval)

            # Cycle evaluation & auto-checkpointing
            if not self._stop_event.is_set():
                val_l = self._evaluate()
                self.val_loss = round(float(val_l), 4)

                # Check for improvement
                saved_new_best = False
                if val_l < self.best_val_loss:
                    self.best_val_loss = val_l
                    self._save_auto_checkpoint()
                    self.auto_checkpoints += 1
                    saved_new_best = True

                status_desc = f"Cycle #{self.cycle_count}: Loss {self.current_loss:.4f} | Val {self.val_loss:.4f}"
                if saved_new_best:
                    status_desc += " (Auto-checkpoint saved)"
                self.status_message = status_desc

                record = {
                    "cycle": self.cycle_count,
                    "step": self.total_steps,
                    "loss": self.current_loss,
                    "val_loss": self.val_loss,
                    "lr": curr_lr,
                    "mode": self.mode,
                    "timestamp": time.time(),
                }
                self.history.append(record)
                if len(self.history) > 30:
                    self.history.pop(0)

                if self.on_step_callback:
                    try:
                        self.on_step_callback(record)
                    except Exception:
                        pass

                # If in self-play mode, generate synthetic training sample periodically
                if self.mode == "self_play" and self.cycle_count % 2 == 0:
                    self._generate_synthetic_sample()

                # Rest briefly between cycles
                time.sleep(0.1)

        self.is_running = False

    def _train_step(self, lr: float) -> float:
        """Run a single gradient descent step under thread safety."""
        with self.lock:
            if isinstance(self.dataset, ChatDataset):
                x, y, mask = self.dataset.get_batch(batch_size=self.batch_size, split="train")
                loss, grads = self.model.loss_and_gradients(x, y, target_mask=mask)
            else:
                x, y = self.dataset.get_batch(batch_size=self.batch_size, split="train")
                loss, grads = self.model.loss_and_gradients(x, y)

            self.optimizer.step(grads, lr=lr)
            return float(loss)

    def _evaluate(self) -> float:
        """Run validation evaluation."""
        losses = []
        with self.lock:
            for _ in range(3):
                if isinstance(self.dataset, ChatDataset):
                    val_x, val_y, val_mask = self.dataset.get_batch(batch_size=self.batch_size, split="val")
                    loss, _ = self.model.loss_and_gradients(val_x, val_y, target_mask=val_mask)
                else:
                    val_x, val_y = self.dataset.get_batch(batch_size=self.batch_size, split="val")
                    loss, _ = self.model.loss_and_gradients(val_x, val_y)
                losses.append(float(loss))
        return float(np.mean(losses)) if losses else 0.0

    def _generate_synthetic_sample(self) -> None:
        """Self-play generation: synthesize QA sample and add to training set."""
        seed_prompt = random.choice(self.SELF_PLAY_SEEDS)
        try:
            from inference.generate import generate_text

            with self.lock:
                reply = generate_text(
                    self.model,
                    self.tokenizer,
                    prompt=f"<user>{seed_prompt}<assistant>",
                    max_new_tokens=40,
                    temperature=0.7,
                    stop_tokens=["<eos>", "<user>"],
                    use_kv_cache=True,
                )

            # Strip prompt
            reply = reply.replace(f"<user>{seed_prompt}<assistant>", "").replace("<eos>", "").strip()
            if len(reply) > 10:
                self.synthetic_pairs_generated += 1
                if isinstance(self.dataset, ChatDataset):
                    sample = {
                        "messages": [
                            {"role": "user", "content": seed_prompt},
                            {"role": "assistant", "content": reply},
                        ]
                    }
                    encoded = self.dataset._encode_sample(sample)
                    if encoded:
                        self.dataset.train_samples.append(encoded)
        except Exception:
            pass

    def _save_auto_checkpoint(self) -> None:
        """Persist best auto-trained checkpoint weights."""
        ckpt_file = self.checkpoint_dir / "auto_trained_model.npz"
        cfg_file = self.checkpoint_dir / "auto_trained_model.json"
        with self.lock:
            self.model.save(
                ckpt_file,
                cfg_file,
                metadata={
                    "auto_cycle": self.cycle_count,
                    "auto_steps": self.total_steps,
                    "val_loss": self.best_val_loss,
                    "timestamp": time.time(),
                },
            )

    def get_telemetry(self) -> Dict[str, Any]:
        """Return full telemetry dictionary for API and UI."""
        return {
            "is_running": self.is_running,
            "mode": self.mode,
            "cycle_count": self.cycle_count,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "val_loss": self.val_loss,
            "best_val_loss": round(float(self.best_val_loss), 4) if self.best_val_loss < 900 else None,
            "auto_checkpoints": self.auto_checkpoints,
            "status_message": self.status_message,
            "synthetic_pairs_generated": self.synthetic_pairs_generated,
            "history": self.history[-15:],
        }
