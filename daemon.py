"""Lightweight internal HTTP daemon providing fast REST/SSE endpoints for NumPy-GPT Studio."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
from evaluation.evaluate import benchmark_generation, evaluate_perplexity
from export.onnx_export import export_gpt_to_onnx, run_onnx_inference
from hardware.backend import get_active_device, select_backend
from hardware.detection import detect_hardware
from inference.generate import generate_stream, generate_text
from model.config import GPTConfig, get_preset_config
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer
from training.dataset import ChatDataset, TextDataset
from training.optimizer import AdamW
from training.scheduler import CosineWarmupScheduler
from training.trainer import Trainer

PORT = int(os.environ.get("PYTHON_BACKEND_PORT", 5005))

# Global runtime state
state_lock = threading.Lock()
current_model: Optional[GPT] = None
current_tokenizer: Optional[ByteBPETokenizer] = None
current_config: Optional[GPTConfig] = None
active_training_thread: Optional[threading.Thread] = None
training_stop_requested = False
training_metrics = {
    "is_training": False,
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "val_loss": 0.0,
    "lr": 0.0,
    "tok_per_sec": 0.0,
    "eta": "",
    "history": [],
}


def ensure_default_model():
    """Ensure a baseline model is loaded into memory."""
    global current_model, current_tokenizer, current_config
    with state_lock:
        if current_model is not None:
            return

        vocab_path = Path("smoke_checkpoint.vocab.json")
        if not vocab_path.exists():
            vocab_path = Path("checkpoint.vocab.json")

        if vocab_path.exists():
            current_tokenizer = ByteBPETokenizer.load(vocab_path)
        else:
            sample_text = Path("data.txt").read_text(encoding="utf-8", errors="replace")
            current_tokenizer = ByteBPETokenizer.train(sample_text, vocab_size=512, verbose=False)
            current_tokenizer.save("smoke_checkpoint.vocab.json")

        current_config = GPTConfig(
            vocab_size=len(current_tokenizer),
            context_length=128,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            norm_type="rmsnorm",
            pos_emb_type="rope",
            activation="swiglu",
            weight_tying=True,
        )

        current_model = GPT(current_config, seed=42)
        current_model.to_device()


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        ensure_default_model()

        if parsed.path == "/api/status":
            with state_lock:
                dev = get_active_device()
                params_count = current_model.parameter_count() if current_model else 0
                self._send_json({
                    "status": "ready",
                    "device": dev,
                    "model_params": params_count,
                    "vocab_size": len(current_tokenizer) if current_tokenizer else 0,
                    "context_length": current_config.context_length if current_config else 0,
                    "embedding_dim": current_config.embedding_dim if current_config else 0,
                    "num_heads": current_config.num_heads if current_config else 0,
                    "num_layers": current_config.num_layers if current_config else 0,
                    "norm_type": current_config.norm_type if current_config else "rmsnorm",
                    "activation": current_config.activation if current_config else "swiglu",
                })
        elif parsed.path == "/api/hardware":
            try:
                rep = detect_hardware()
                self._send_json({
                    "os": rep.platform_name,
                    "cpu_name": rep.cpu_model,
                    "cpu_cores": rep.cpu_cores,
                    "ram_gb": rep.ram_gb,
                    "has_cuda": rep.cuda_available,
                    "cuda_device_name": rep.cuda_device_name,
                    "cuda_vram_gb": rep.cuda_vram_gb,
                    "has_npu": rep.npu_available,
                    "npu_details": f"Providers: {', '.join(rep.npu_providers) if rep.npu_providers else 'None'}",
                    "has_avx2": True,
                    "has_avx512": False,
                    "recommended_backend": rep.recommended_backend,
                    "active_device": get_active_device(),
                })
            except Exception as ex:
                self._send_json({"error": str(ex)}, status=500)
        elif parsed.path == "/api/train/status":
            with state_lock:
                self._send_json(dict(training_metrics))
        else:
            self._send_json({"error": "Not Found"}, status=404)

    def do_POST(self) -> None:
        global active_training_thread, training_stop_requested
        parsed = urlparse(self.path)
        ensure_default_model()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/generate":
            prompt = payload.get("prompt", "Artificial intelligence")
            max_new_tokens = min(int(payload.get("max_new_tokens", 80)), 300)
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 40))
            top_p = float(payload.get("top_p", 0.9))
            repetition_penalty = float(payload.get("repetition_penalty", 1.15))
            use_cache = bool(payload.get("use_cache", True))

            t0 = time.perf_counter()
            with state_lock:
                text = generate_text(
                    model=current_model,
                    tokenizer=current_tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_cache=use_cache,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            tok_count = len(current_tokenizer.encode(text, allowed_special=True))

            self._send_json({
                "prompt": prompt,
                "text": text,
                "tokens_generated": tok_count,
                "latency_ms": round(latency_ms, 2),
                "speed_tok_s": round(tok_count / max(0.001, (latency_ms / 1000)), 1),
            })

        elif parsed.path == "/api/chat":
            messages = payload.get("messages", [])
            system_prompt = payload.get("system_prompt", "You are NumPy-GPT, an intelligent local AI model running on pure NumPy.")
            temperature = float(payload.get("temperature", 0.7))
            top_k = int(payload.get("top_k", 40))
            top_p = float(payload.get("top_p", 0.9))
            repetition_penalty = float(payload.get("repetition_penalty", 1.15))
            max_new_tokens = min(int(payload.get("max_new_tokens", 120)), 300)

            # Build conversation prompt
            parts = [f"<system>{system_prompt}"]
            for m in messages:
                r = m.get("role", "user").lower()
                c = m.get("content", "").strip()
                if r == "user":
                    parts.append(f"<user>{c}")
                elif r == "assistant":
                    parts.append(f"<assistant>{c}<eos>")
            parts.append("<assistant>")
            full_prompt = "".join(parts)

            t0 = time.perf_counter()
            with state_lock:
                reply = generate_text(
                    model=current_model,
                    tokenizer=current_tokenizer,
                    prompt=full_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                )
            latency_ms = (time.perf_counter() - t0) * 1000

            self._send_json({
                "reply": reply.strip(),
                "latency_ms": round(latency_ms, 2),
            })

        elif parsed.path == "/api/benchmark":
            batch_size = int(payload.get("batch_size", 2))
            with state_lock:
                bench = benchmark_generation(
                    model=current_model,
                    tokenizer=current_tokenizer,
                    prompt="The history of computation demonstrates that",
                    generate_tokens=30,
                )
            self._send_json(bench)

        elif parsed.path == "/api/export":
            output_file = Path("model.onnx")
            with state_lock:
                export_gpt_to_onnx(current_model, output_file)
                # Verify with onnxruntime
                test_tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
                logits = run_onnx_inference(output_file, test_tokens)
            self._send_json({
                "status": "success",
                "filename": output_file.name,
                "size_kb": round(output_file.stat().st_size / 1024, 1),
                "output_shape": list(logits.shape),
            })

        elif parsed.path == "/api/train/start":
            with state_lock:
                if training_metrics["is_training"]:
                    self._send_json({"error": "Training already in progress"}, status=400)
                    return

                training_stop_requested = False
                steps = min(int(payload.get("steps", 50)), 5000)
                lr = float(payload.get("lr", 5e-4))
                batch_size = int(payload.get("batch_size", 2))
                grad_accum = int(payload.get("grad_accum", 1))

                def run_train():
                    global training_stop_requested
                    try:
                        training_metrics["is_training"] = True
                        training_metrics["total_steps"] = steps
                        training_metrics["step"] = 0
                        training_metrics["history"] = []

                        data_p = Path("data.txt")
                        text_content = data_p.read_text(encoding="utf-8", errors="replace")
                        tokens = current_tokenizer.encode(text_content, allowed_special=True)
                        dataset = TextDataset(tokens, context_length=current_config.context_length)

                        opt = AdamW(current_model.params, lr=lr, weight_decay=0.1)
                        sched = CosineWarmupScheduler(base_lr=lr, warmup_steps=max(5, steps // 10), max_steps=steps)

                        for s in range(1, steps + 1):
                            if training_stop_requested:
                                break

                            x, y = dataset.get_batch(batch_size=batch_size, split="train")
                            loss, grads = current_model.loss_and_gradients(x, y)
                            curr_lr = sched.get_lr(s)
                            opt.step(grads, lr=curr_lr)

                            if s % 5 == 0 or s == steps:
                                val_x, val_y = dataset.get_batch(batch_size=batch_size, split="val")
                                val_loss, _ = current_model.loss_and_gradients(val_x, val_y)
                                training_metrics["step"] = s
                                training_metrics["loss"] = round(float(loss), 4)
                                training_metrics["val_loss"] = round(float(val_loss), 4)
                                training_metrics["lr"] = curr_lr
                                training_metrics["history"].append({
                                    "step": s,
                                    "loss": round(float(loss), 4),
                                    "val_loss": round(float(val_loss), 4),
                                })

                            time.sleep(0.02)

                    finally:
                        training_metrics["is_training"] = False

                active_training_thread = threading.Thread(target=run_train, daemon=True)
                active_training_thread.start()

            self._send_json({"status": "training_started", "steps": steps})

        elif parsed.path == "/api/train/stop":
            training_stop_requested = True
            self._send_json({"status": "stopping"})
        else:
            self._send_json({"error": "Not Found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard logs
        pass


def start_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), RequestHandler)
    print(f"[NumPy-GPT Daemon] Running internal API server on port {PORT}...")
    server.serve_forever()


if __name__ == "__main__":
    ensure_default_model()
    start_server()
