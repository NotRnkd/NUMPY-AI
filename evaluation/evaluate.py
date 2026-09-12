"""Evaluation suite: perplexity, generation speed, repetition metrics, and instruction benchmarks."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from hardware.backend import synchronize
from inference.generate import generate_text
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer


def calculate_repetition_rate(text: str, n: int = 3) -> float:
    """Calculate the ratio of duplicate n-grams in generated text."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / max(1, len(ngrams)))


def evaluate_perplexity(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    text: str,
    max_chunks: int = 25,
) -> Dict[str, float]:
    """Compute cross-entropy loss and perplexity on test text."""
    tokens = tokenizer.encode(text, allowed_special=True)
    ctx_len = model.config.context_length
    if len(tokens) <= ctx_len + 1:
        tokens = tokens * ((ctx_len + 2) // len(tokens) + 1)

    losses = []
    num_chunks = min(max_chunks, (len(tokens) - 1) // ctx_len)

    for i in range(num_chunks):
        start = i * ctx_len
        x = np.array([tokens[start : start + ctx_len]], dtype=np.int64)
        y = np.array([tokens[start + 1 : start + ctx_len + 1]], dtype=np.int64)
        loss, _ = model.loss_and_gradients(x, y)
        losses.append(loss)

    mean_loss = float(np.mean(losses))
    perplexity = float(np.exp(min(mean_loss, 20.0)))
    return {"loss": round(mean_loss, 4), "perplexity": round(perplexity, 2)}


def benchmark_generation(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    prompt: str = "Artificial intelligence is",
    generate_tokens: int = 50,
) -> Dict[str, Any]:
    """Benchmark autoregressive generation speed with and without KV cache."""
    # Warmup
    _ = generate_text(model, tokenizer, prompt, max_new_tokens=5, use_cache=True)
    synchronize()

    # 1. With KV Cache
    t0 = time.perf_counter()
    out_cached = generate_text(model, tokenizer, prompt, max_new_tokens=generate_tokens, use_cache=True)
    synchronize()
    t_cached = time.perf_counter() - t0
    speed_cached = generate_tokens / max(1e-5, t_cached)

    # 2. Without KV Cache
    t0 = time.perf_counter()
    out_no_cache = generate_text(model, tokenizer, prompt, max_new_tokens=generate_tokens, use_cache=False)
    synchronize()
    t_no_cache = time.perf_counter() - t0
    speed_no_cache = generate_tokens / max(1e-5, t_no_cache)

    speedup = speed_cached / max(1e-5, speed_no_cache)

    return {
        "tokens_generated": generate_tokens,
        "kv_cached_speed_tok_s": round(speed_cached, 1),
        "uncached_speed_tok_s": round(speed_no_cache, 1),
        "kv_cache_speedup": round(speedup, 2),
        "sample_output": out_cached,
    }


def run_instruction_benchmark(
    model: GPT,
    tokenizer: ByteBPETokenizer,
) -> List[Dict[str, Any]]:
    """Run standardized battery of instruction prompts."""
    test_prompts = [
        ("Identity", "<system>You are NumPy-GPT.<user>Who are you?<assistant>"),
        ("Math/Logic", "<system>You are a helpful assistant.<user>What is 15 + 27?<assistant>"),
        ("Knowledge", "<system>You are a helpful assistant.<user>What is Python?<assistant>"),
        ("Code", "<system>You are a coding assistant.<user>Write a Python function to square a number.<assistant>"),
    ]

    results = []
    print("\n--- Running Instruction & Conversational Benchmark ---")
    for category, prompt in test_prompts:
        t0 = time.perf_counter()
        resp = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=60,
            temperature=0.6,
            use_cache=True,
        )
        elapsed = time.perf_counter() - t0
        rep_3gram = calculate_repetition_rate(resp, n=3)

        results.append({
            "category": category,
            "prompt": prompt,
            "response": resp.strip(),
            "elapsed_sec": round(elapsed, 3),
            "repetition_3gram": round(rep_3gram, 3),
        })
        print(f"[{category}] -> {resp.strip()[:80]}... (rep: {rep_3gram:.2f})")

    return results
