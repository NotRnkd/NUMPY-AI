"""Trainer engine orchestrating training loops, gradient accumulation, and checkpoints."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from hardware.backend import get_active_device, get_backend, synchronize
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer
from training.dataset import ChatDataset, TextDataset
from training.optimizer import AdamW
from training.scheduler import CosineWarmupScheduler


class Trainer:
    def __init__(
        self,
        model: GPT,
        tokenizer: ByteBPETokenizer,
        optimizer: AdamW,
        scheduler: CosineWarmupScheduler,
        checkpoint_path: Union[str, Path] = "checkpoint.npz",
        config_path: Union[str, Path] = "checkpoint.json",
        best_checkpoint_path: Optional[Union[str, Path]] = None,
        grad_accum_steps: int = 1,
        eval_interval: int = 50,
        eval_steps: int = 10,
        save_interval: int = 100,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.best_checkpoint_path = (
            Path(best_checkpoint_path)
            if best_checkpoint_path
            else self.checkpoint_path.with_name("best_" + self.checkpoint_path.name)
        )
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.best_val_loss = float("inf")

    def evaluate(self, dataset: Union[TextDataset, ChatDataset]) -> float:
        """Compute validation loss averaged over eval_steps."""
        losses = []
        for _ in range(self.eval_steps):
            if isinstance(dataset, ChatDataset):
                x, y, mask = dataset.get_batch(batch_size=4, split="val")
                loss, _ = self.model.loss_and_gradients(x, y, target_mask=mask)
            else:
                x, y = dataset.get_batch(batch_size=4, split="val")
                loss, _ = self.model.loss_and_gradients(x, y)
            losses.append(loss)
        return float(np.mean(losses))

    def train(
        self,
        dataset: Union[TextDataset, ChatDataset],
        total_steps: int,
        batch_size: int = 4,
        start_step: int = 0,
    ) -> None:
        xp = get_backend()
        device_name = get_active_device()
        print(f"Starting training on {device_name.upper()} backend.")
        print(f"Total Steps: {total_steps} | Micro-batch: {batch_size} | Grad Accum: {self.grad_accum_steps} | Effective Batch: {batch_size * self.grad_accum_steps}")
        print(f"Model Parameters: {self.model.parameter_count():,} | Context: {self.model.config.context_length}")
        print("-" * 80)

        step = start_step
        tokens_per_accum = batch_size * self.model.config.context_length * self.grad_accum_steps
        start_time = time.time()
        last_log_time = start_time
        accum_loss = 0.0

        try:
            while step < total_steps:
                step_start_time = time.time()

                # Initialize accumulated gradients dictionary
                accum_grads = {k: xp.zeros_like(v) for k, v in self.model.params.items()}
                accum_loss = 0.0

                # Micro-batch gradient accumulation
                for _ in range(self.grad_accum_steps):
                    if isinstance(dataset, ChatDataset):
                        x, y, mask = dataset.get_batch(batch_size=batch_size, split="train")
                        micro_loss, micro_grads = self.model.loss_and_gradients(x, y, target_mask=mask)
                    else:
                        x, y = dataset.get_batch(batch_size=batch_size, split="train")
                        micro_loss, micro_grads = self.model.loss_and_gradients(x, y)

                    accum_loss += micro_loss / self.grad_accum_steps
                    for name in accum_grads:
                        if micro_grads.get(name) is not None:
                            accum_grads[name] += micro_grads[name] / self.grad_accum_steps

                synchronize()

                # Optimizer step with scheduler learning rate
                lr = self.scheduler.get_lr(step)
                grad_norm = self.optimizer.step(accum_grads, lr=lr)
                step += 1

                # Periodic Logging
                if step % self.log_interval == 0 or step == total_steps:
                    now = time.time()
                    elapsed_step = now - last_log_time
                    tok_per_sec = (tokens_per_accum * self.log_interval) / max(1e-5, elapsed_step)
                    last_log_time = now

                    # ETA estimation
                    steps_remaining = total_steps - step
                    avg_step_time = (now - start_time) / max(1, step - start_step)
                    eta_sec = int(steps_remaining * avg_step_time)
                    eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"

                    print(
                        f"Step {step:6d}/{total_steps:6d} | "
                        f"Loss: {accum_loss:6.4f} | "
                        f"LR: {lr:8.2e} | "
                        f"GradNorm: {grad_norm:5.2f} | "
                        f"Speed: {tok_per_sec:6.0f} tok/s | "
                        f"ETA: {eta_str}"
                    )

                # Periodic Validation
                if step % self.eval_interval == 0 or step == total_steps:
                    val_loss = self.evaluate(dataset)
                    val_ppl = np.exp(min(val_loss, 20.0))
                    print(f"  --> Evaluation at step {step}: Val Loss = {val_loss:.4f}, Val PPL = {val_ppl:.2f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        best_cfg_p = self.best_checkpoint_path.with_suffix(".json")
                        self.model.save(
                            self.best_checkpoint_path,
                            best_cfg_p,
                            optimizer_state=self.optimizer.state_dict(),
                            metadata={"step": step, "val_loss": val_loss},
                        )
                        print(f"  --> Saved new best checkpoint: {self.best_checkpoint_path.name}")

                # Periodic Checkpointing
                if step % self.save_interval == 0 or step == total_steps:
                    self.model.save(
                        self.checkpoint_path,
                        self.config_path,
                        optimizer_state=self.optimizer.state_dict(),
                        metadata={"step": step, "loss": accum_loss},
                    )

        except KeyboardInterrupt:
            print("\n[Trainer] KeyboardInterrupt received. Saving current progress...")
            self.model.save(
                self.checkpoint_path,
                self.config_path,
                optimizer_state=self.optimizer.state_dict(),
                metadata={"step": step, "loss": accum_loss, "interrupted": True},
            )
            print(f"[Trainer] Checkpoint successfully saved to {self.checkpoint_path}")
            return

        total_elapsed = time.time() - start_time
        print(f"Training completed in {total_elapsed / 60:.1f} minutes. Best Val Loss: {self.best_val_loss:.4f}")
