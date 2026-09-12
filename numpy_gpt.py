#!/usr/bin/env python3
"""NumPy GPT: Production-grade, offline-first Neural Language Model in pure NumPy / CuPy.

Unified CLI for training, inference, chat, evaluation, hardware diagnostics, and ONNX export.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from evaluation.evaluate import (
    benchmark_generation,
    evaluate_perplexity,
    run_instruction_benchmark,
)
from export.onnx_export import export_gpt_to_onnx, run_onnx_inference
from hardware.backend import get_active_device, select_backend
from hardware.detection import detect_hardware, print_hardware_summary
from inference.chat import interactive_chat_repl
from inference.generate import generate_stream, generate_text
from model.config import CONFIG_PRESETS, GPTConfig, get_preset_config
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer, SpecialTokens
from training.auto_trainer import AutoTrainer
from training.dataset import ChatDataset, TextDataset
from training.optimizer import AdamW
from training.scheduler import CosineWarmupScheduler
from training.trainer import Trainer


def _resolve_paths(checkpoint_arg: str, config_arg: Optional[str], vocab_arg: Optional[str]):
    ckpt_path = Path(checkpoint_arg)
    if not ckpt_path.suffix:
        ckpt_path = ckpt_path.with_suffix(".npz")

    # Config path
    if config_arg:
        cfg_path = Path(config_arg)
    else:
        cfg_path = ckpt_path.with_suffix(".json")

    # Vocab path
    if vocab_arg:
        voc_path = Path(vocab_arg)
    else:
        voc_path = ckpt_path.with_name(ckpt_path.stem + ".vocab.json")
        if not voc_path.exists():
            voc_path = Path("smoke_checkpoint.vocab.json")

    return ckpt_path, cfg_path, voc_path


def cmd_hardware(args: argparse.Namespace) -> None:
    print_hardware_summary()


def cmd_train(args: argparse.Namespace) -> None:
    # 1. Hardware backend setup
    device = select_backend(args.device)
    print(f"[Hardware] Selected compute backend: {device.upper()}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    ckpt_path, cfg_path, voc_path = _resolve_paths(args.checkpoint, args.config, args.tokenizer_vocab)

    # 2. Tokenizer initialization or training
    if voc_path.exists() and not args.retrain_tokenizer:
        print(f"[Tokenizer] Loading vocabulary from {voc_path} ...")
        tokenizer = ByteBPETokenizer.load(voc_path)
    else:
        print(f"[Tokenizer] Training Byte-BPE tokenizer on {data_path} (target vocab_size={args.vocab_size})...")
        raw_text = data_path.read_text(encoding="utf-8", errors="replace")
        tokenizer = ByteBPETokenizer.train(
            raw_text,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            verbose=True,
        )
        voc_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(voc_path)
        print(f"[Tokenizer] Saved trained tokenizer to {voc_path}")

    # 3. Model Configuration
    if args.resume and cfg_path.exists():
        print(f"[Model] Resuming configuration from {cfg_path} ...")
        raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        config = GPTConfig.from_dict(raw_cfg)
    elif args.preset:
        print(f"[Model] Using '{args.preset}' architecture preset ...")
        config = get_preset_config(args.preset, vocab_size=len(tokenizer))
    else:
        config = GPTConfig(
            vocab_size=len(tokenizer),
            context_length=args.context_length,
            embedding_dim=args.dim,
            num_heads=args.heads,
            num_layers=args.layers,
            norm_type=args.norm_type,
            pos_emb_type=args.pos_emb_type,
            activation=args.activation,
            weight_tying=(not args.no_weight_tying),
        )

    # 4. Model instantiation & weight loading
    start_step = 0
    saved_opt_state = None

    if args.resume and ckpt_path.exists():
        print(f"[Model] Loading checkpoint weights from {ckpt_path} ...")
        model, saved_opt_state = GPT.load(ckpt_path, cfg_path)
        if saved_opt_state and "step" in saved_opt_state:
            start_step = int(saved_opt_state["step"])
            print(f"[Model] Resuming from step {start_step}")
    else:
        print(f"[Model] Initializing new model with {config.estimate_params():,} estimated parameters ...")
        model = GPT(config, seed=args.seed)

    model.to_device()

    # 5. Dataset loading
    is_chat = data_path.suffix in (".json", ".jsonl")
    if is_chat:
        print(f"[Dataset] Loading chat dataset from {data_path} with loss masking ...")
        dataset = ChatDataset.load_json(data_path, tokenizer, context_length=config.context_length)
    else:
        print(f"[Dataset] Loading plain text corpus from {data_path} ...")
        text_content = data_path.read_text(encoding="utf-8", errors="replace")
        tokens = tokenizer.encode(text_content, allowed_special=True)
        print(f"[Dataset] Tokenized corpus: {len(tokens):,} tokens")
        dataset = TextDataset(tokens, context_length=config.context_length)

    # 6. Optimizer & Scheduler
    optimizer = AdamW(
        model.params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
    )
    if saved_opt_state:
        optimizer.load_state_dict(saved_opt_state)

    scheduler = CosineWarmupScheduler(
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        min_lr=args.min_lr,
    )

    # 7. Training Engine
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        grad_accum_steps=args.grad_accum,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
    )

    trainer.train(
        dataset=dataset,
        total_steps=args.steps,
        batch_size=args.batch_size,
        start_step=start_step,
    )


def cmd_autotrain(args: argparse.Namespace) -> None:
    """Run autonomous background self-training until interrupted."""
    device = select_backend(args.device)
    print(f"[Hardware] Selected compute backend: {device.upper()}")

    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path("corpus/sample_chatgpt_dataset.json")
        if not data_path.exists():
            data_path = Path("data.txt")

    ckpt_path, cfg_path, voc_path = _resolve_paths(args.checkpoint, args.config, args.tokenizer_vocab)

    # 1. Tokenizer
    if voc_path.exists():
        tokenizer = ByteBPETokenizer.load(voc_path)
    else:
        sample_text = data_path.read_text(encoding="utf-8", errors="replace")[:10000]
        tokenizer = ByteBPETokenizer.train(sample_text, vocab_size=args.vocab_size or 512, verbose=False)
        tokenizer.save(voc_path)

    # 2. Config & Model
    if cfg_path.exists():
        config = GPTConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    else:
        config = GPTConfig(
            vocab_size=len(tokenizer),
            context_length=args.context_length,
            embedding_dim=args.dim,
            num_heads=args.heads,
            num_layers=args.layers,
            norm_type=args.norm_type,
            pos_emb_type=args.pos_emb_type,
            activation=args.activation,
        )

    if ckpt_path.exists():
        print(f"[AutoTrain] Resuming from existing checkpoint: {ckpt_path}")
        model = GPT.load(ckpt_path, config=config)
    else:
        print(f"[AutoTrain] Initializing new model with {config.estimate_parameter_count():,} parameters...")
        model = GPT(config, seed=args.seed)

    # 3. Dataset
    if data_path.suffix in (".json", ".jsonl"):
        print(f"[AutoTrain] Using ChatDataset with loss masking from {data_path} ...")
        dataset = ChatDataset.load_json(data_path, tokenizer, context_length=config.context_length)
    else:
        print(f"[AutoTrain] Using plain text corpus from {data_path} ...")
        text_content = data_path.read_text(encoding="utf-8", errors="replace")
        tokens = tokenizer.encode(text_content, allowed_special=True)
        dataset = TextDataset(tokens, context_length=config.context_length)

    auto_trainer = AutoTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        base_lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\n=======================================================")
    print(f"   NumPy-GPT Autonomous Auto-Training Engine Active")
    print(f"   Mode: {args.mode.upper()} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"   Checkpoints saved automatically to: {args.checkpoint_dir}/")
    print(f"   Press Ctrl+C at any time to pause or exit.")
    print(f"=======================================================\n")

    auto_trainer.start(mode=args.mode, target_lr=args.lr)

    try:
        last_step = 0
        while True:
            time.sleep(1.5)
            tel = auto_trainer.get_telemetry()
            if tel["total_steps"] != last_step:
                last_step = tel["total_steps"]
                val_str = f"{tel['val_loss']:.4f}" if tel['val_loss'] else "evaluating..."
                best_str = f"{tel['best_val_loss']:.4f}" if tel['best_val_loss'] else "N/A"
                print(
                    f"Auto-Cycle #{tel['cycle_count']:3d} | Step {tel['total_steps']:5d} | "
                    f"Loss: {tel['current_loss']:.4f} | Val: {val_str} (Best: {best_str}) | "
                    f"Saved: {tel['auto_checkpoints']} ckpts"
                )
    except KeyboardInterrupt:
        print("\n[AutoTrain] Gracefully stopping autonomous trainer...")
        auto_trainer.stop()
        print("[AutoTrain] Stopped. Best model saved in checkpoints/auto_trained_model.npz.")


def cmd_generate(args: argparse.Namespace) -> None:
    select_backend(args.device)
    ckpt_path, cfg_path, voc_path = _resolve_paths(args.checkpoint, args.config, args.tokenizer_vocab)

    print(f"Loading model from {ckpt_path} ...")
    model, _ = GPT.load(ckpt_path, cfg_path)
    model.to_device()

    tokenizer = ByteBPETokenizer.load(voc_path)

    print(f"Prompt: {args.prompt}")
    print("Generating:")
    print("-" * 50)

    if args.stream:
        stream = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=(not args.no_cache),
            seed=args.seed,
        )
        for piece in stream:
            print(piece, end="", flush=True)
        print()
    else:
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=(not args.no_cache),
            seed=args.seed,
        )
        print(text)


def cmd_chat(args: argparse.Namespace) -> None:
    select_backend(args.device)
    ckpt_path, cfg_path, voc_path = _resolve_paths(args.checkpoint, args.config, args.tokenizer_vocab)

    print(f"Loading chat model from {ckpt_path} ...")
    model, _ = GPT.load(ckpt_path, cfg_path)
    model.to_device()

    tokenizer = ByteBPETokenizer.load(voc_path)

    interactive_chat_repl(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    select_backend(args.device)
    ckpt_path, cfg_path, voc_path = _resolve_paths(args.checkpoint, args.config, args.tokenizer_vocab)

    model, _ = GPT.load(ckpt_path, cfg_path)
    model.to_device()
    tokenizer = ByteBPETokenizer.load(voc_path)

    print("=" * 60)
    print("                MODEL EVALUATION SUITE")
    print("=" * 60)
    print(f"Model Parameters: {model.parameter_count():,}")
    print(f"Context Length:   {model.config.context_length}")
    print(f"Active Backend:   {get_active_device().upper()}")

    # 1. Perplexity evaluation
    if args.data:
        data_file = Path(args.data)
        if data_file.exists():
            print(f"\nEvaluating Perplexity on {data_file.name} ...")
            test_text = data_file.read_text(encoding="utf-8", errors="replace")
            res = evaluate_perplexity(model, tokenizer, test_text, max_chunks=args.max_eval_chunks)
            print(f"  Cross-Entropy Loss: {res['loss']:.4f}")
            print(f"  Perplexity (PPL):   {res['perplexity']:.2f}")

    # 2. Generation Speed & KV-Cache Benchmark
    print("\nBenchmarking Generation Speed & KV Cache ...")
    bench = benchmark_generation(model, tokenizer, prompt="The history of science shows that", generate_tokens=40)
    print(f"  Speed with KV Cache:    {bench['kv_cached_speed_tok_s']} tokens/sec")
    print(f"  Speed without KV Cache: {bench['uncached_speed_tok_s']} tokens/sec")
    print(f"  KV Cache Acceleration:  {bench['kv_cache_speedup']}x faster")

    # 3. Instruction battery
    if args.instruction_bench:
        run_instruction_benchmark(model, tokenizer)

    print("=" * 60)


def cmd_benchmark(args: argparse.Namespace) -> None:
    device = select_backend(args.device)
    print("=" * 60)
    print(f"          SYSTEM HARDWARE & MODEL BENCHMARK ({device.upper()})")
    print("=" * 60)

    # Instantiate representative benchmark model
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        embedding_dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        norm_type="rmsnorm",
        activation="swiglu",
    )
    model = GPT(cfg, seed=42)
    model.to_device()
    print(f"Model: {cfg.num_layers}L-{cfg.embedding_dim}D-{cfg.num_heads}H ({model.parameter_count():,} parameters)")
    print(f"Context: {cfg.context_length} | Micro-batch: {args.batch_size}")

    # Create dummy batch
    xp = model.params["wte"].__class__
    x = np.random.randint(0, cfg.vocab_size, size=(args.batch_size, cfg.context_length), dtype=np.int64)
    y = np.random.randint(0, cfg.vocab_size, size=(args.batch_size, cfg.context_length), dtype=np.int64)

    # Warmup
    for _ in range(2):
        _ = model.forward(x)
        _, _ = model.loss_and_gradients(x, y)

    # 1. Forward Pass Bench
    times_fwd = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model.forward(x)
        times_fwd.append(time.perf_counter() - t0)
    fwd_ms = np.mean(times_fwd) * 1000

    # 2. Forward + Backward Pass Bench
    times_bwd = []
    for _ in range(10):
        t0 = time.perf_counter()
        _, _ = model.loss_and_gradients(x, y)
        times_bwd.append(time.perf_counter() - t0)
    bwd_ms = np.mean(times_bwd) * 1000

    tok_per_sec = (args.batch_size * cfg.context_length) / max(1e-5, (bwd_ms / 1000))

    print(f"  Forward Pass Latency:           {fwd_ms:.2f} ms")
    print(f"  Forward + Backward Step:        {bwd_ms:.2f} ms")
    print(f"  Training Throughput:            {tok_per_sec:.0f} tokens/second")
    print("=" * 60)


def cmd_export(args: argparse.Namespace) -> None:
    ckpt_path, cfg_path, _ = _resolve_paths(args.checkpoint, args.config, None)
    output_path = Path(args.output) if args.output else ckpt_path.with_suffix(".onnx")

    print(f"Loading model from {ckpt_path} ...")
    model, _ = GPT.load(ckpt_path, cfg_path)

    print(f"Exporting to ONNX at {output_path} ...")
    export_gpt_to_onnx(model, output_path)

    if args.test:
        print("Testing exported ONNX model with ONNX Runtime...")
        test_tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
        out = run_onnx_inference(output_path, test_tokens)
        print(f"ONNX Runtime Output shape: {out.shape} (Valid!)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="numpy_gpt",
        description="NumPy-GPT: High-Performance, Offline-First Language Model in NumPy/CuPy",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # --- Hardware Subcommand ---
    subparsers.add_parser("hardware", help="Inspect local CPU, CUDA, and NPU/DirectML hardware")

    # --- Train Subcommand ---
    p_train = subparsers.add_parser("train", help="Train model on corpus or instruction dataset")
    p_train.add_argument("--data", type=str, required=True, help="Path to text or chat JSON dataset")
    p_train.add_argument("--checkpoint", type=str, default="checkpoint.npz", help="Path to checkpoint")
    p_train.add_argument("--config", type=str, default=None, help="Path to model config JSON")
    p_train.add_argument("--tokenizer-vocab", type=str, default=None, help="Path to tokenizer vocab JSON")
    p_train.add_argument("--retrain-tokenizer", action="store_true", help="Force retrain tokenizer")
    p_train.add_argument("--vocab-size", type=int, default=4096, help="Tokenizer vocab size")
    p_train.add_argument("--min-frequency", type=int, default=2, help="Tokenizer min frequency")
    p_train.add_argument("--preset", type=str, default=None, choices=list(CONFIG_PRESETS.keys()), help="Model preset")
    p_train.add_argument("--context-length", "--block-size", dest="context_length", type=int, default=256)
    p_train.add_argument("--dim", "--n-embd", dest="dim", type=int, default=128)
    p_train.add_argument("--heads", "--n-head", dest="heads", type=int, default=4)
    p_train.add_argument("--layers", "--n-layer", dest="layers", type=int, default=4)
    p_train.add_argument("--norm-type", type=str, default="rmsnorm", choices=["rmsnorm", "layernorm"])
    p_train.add_argument("--pos-emb-type", type=str, default="rope", choices=["rope", "learned"])
    p_train.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "gelu", "relu"])
    p_train.add_argument("--no-weight-tying", action="store_true", help="Do not tie lm_head to wte")
    p_train.add_argument("--steps", type=int, default=1000, help="Total training steps")
    p_train.add_argument("--batch-size", type=int, default=4, help="Micro-batch size")
    p_train.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    p_train.add_argument("--lr", type=float, default=5e-4, help="Base learning rate")
    p_train.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate")
    p_train.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    p_train.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
    p_train.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    p_train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--log-interval", type=int, default=10)
    p_train.add_argument("--eval-interval", type=int, default=50)
    p_train.add_argument("--eval-steps", type=int, default=10)
    p_train.add_argument("--save-interval", type=int, default=100)

    # --- Generate Subcommand ---
    p_gen = subparsers.add_parser("generate", help="Autoregressively generate text from prompt")
    p_gen.add_argument("--prompt", type=str, required=True, help="Input prompt")
    p_gen.add_argument("--checkpoint", type=str, default="checkpoint.npz", help="Path to checkpoint")
    p_gen.add_argument("--config", type=str, default=None, help="Path to model config JSON")
    p_gen.add_argument("--tokenizer-vocab", type=str, default=None, help="Path to tokenizer vocab JSON")
    p_gen.add_argument("--max-new-tokens", type=int, default=150)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument("--top-p", type=float, default=0.9)
    p_gen.add_argument("--repetition-penalty", type=float, default=1.15)
    p_gen.add_argument("--stream", action="store_true", default=True, help="Stream tokens to console")
    p_gen.add_argument("--no-cache", action="store_true", help="Disable KV-cache acceleration")
    p_gen.add_argument("--device", type=str, default="auto")
    p_gen.add_argument("--seed", type=int, default=None)

    # --- Chat Subcommand ---
    p_chat = subparsers.add_parser("chat", help="Launch interactive multi-turn terminal chat")
    p_chat.add_argument("--checkpoint", type=str, default="checkpoint.npz", help="Path to checkpoint")
    p_chat.add_argument("--config", type=str, default=None, help="Path to model config JSON")
    p_chat.add_argument("--tokenizer-vocab", type=str, default=None, help="Path to tokenizer vocab JSON")
    p_chat.add_argument("--system-prompt", type=str, default=None, help="Custom system prompt")
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--top-k", type=int, default=40)
    p_chat.add_argument("--top-p", type=float, default=0.9)
    p_chat.add_argument("--repetition-penalty", type=float, default=1.15)
    p_chat.add_argument("--device", type=str, default="auto")

    # --- Evaluate Subcommand ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model perplexity and instruction following")
    p_eval.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    p_eval.add_argument("--config", type=str, default=None)
    p_eval.add_argument("--tokenizer-vocab", type=str, default=None)
    p_eval.add_argument("--data", type=str, default=None, help="Validation text dataset")
    p_eval.add_argument("--max-eval-chunks", type=int, default=25)
    p_eval.add_argument("--instruction-bench", action="store_true", default=True)
    p_eval.add_argument("--device", type=str, default="auto")

    # --- Benchmark Subcommand ---
    p_bench = subparsers.add_parser("benchmark", help="Benchmark hardware compute speed and throughput")
    p_bench.add_argument("--vocab-size", type=int, default=4096)
    p_bench.add_argument("--context-length", type=int, default=256)
    p_bench.add_argument("--dim", type=int, default=128)
    p_bench.add_argument("--heads", type=int, default=4)
    p_bench.add_argument("--layers", type=int, default=4)
    p_bench.add_argument("--batch-size", type=int, default=4)
    p_bench.add_argument("--device", type=str, default="auto")

    # --- Export Subcommand ---
    p_exp = subparsers.add_parser("export", help="Export model to ONNX for NPU/DirectML execution")
    p_exp.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    p_exp.add_argument("--config", type=str, default=None)
    p_exp.add_argument("--output", type=str, default=None, help="Output .onnx path")
    p_exp.add_argument("--test", action="store_true", default=True, help="Test exported model with ONNX Runtime")

    # --- Auto-Train Subcommand ---
    p_auto = subparsers.add_parser("auto-train", help="Autonomous self-training loop with automatic evaluation and checkpoints")
    p_auto.add_argument("--data", type=str, default="corpus/sample_chatgpt_dataset.json", help="Path to JSON/JSONL or text dataset")
    p_auto.add_argument("--mode", type=str, default="autonomous_loop", choices=["autonomous_loop", "self_play"], help="Self-training mode")
    p_auto.add_argument("--checkpoint", type=str, default="checkpoint.npz")
    p_auto.add_argument("--config", type=str, default=None)
    p_auto.add_argument("--tokenizer-vocab", type=str, default=None)
    p_auto.add_argument("--vocab-size", type=int, default=512)
    p_auto.add_argument("--context-length", type=int, default=128)
    p_auto.add_argument("--dim", type=int, default=128)
    p_auto.add_argument("--heads", type=int, default=4)
    p_auto.add_argument("--layers", type=int, default=4)
    p_auto.add_argument("--norm-type", type=str, default="rmsnorm")
    p_auto.add_argument("--pos-emb-type", type=str, default="rope")
    p_auto.add_argument("--activation", type=str, default="swiglu")
    p_auto.add_argument("--batch-size", type=int, default=2)
    p_auto.add_argument("--lr", type=float, default=3e-4)
    p_auto.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_auto.add_argument("--device", type=str, default="auto")
    p_auto.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.subcommand == "hardware":
        cmd_hardware(args)
    elif args.subcommand == "train":
        cmd_train(args)
    elif args.subcommand == "auto-train":
        cmd_autotrain(args)
    elif args.subcommand == "generate":
        cmd_generate(args)
    elif args.subcommand == "chat":
        cmd_chat(args)
    elif args.subcommand == "evaluate":
        cmd_evaluate(args)
    elif args.subcommand == "benchmark":
        cmd_benchmark(args)
    elif args.subcommand == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
