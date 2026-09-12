"""Autoregressive text generation with streaming, sampling, and KV caching."""

from __future__ import annotations

from typing import Any, Callable, Generator, List, Optional, Sequence, Union

import numpy as np
from hardware.backend import get_backend, to_cpu
from model.transformer import GPT
from tokenizer.bpe import ByteBPETokenizer


def sample_next_token(
    logits: np.ndarray,
    generated_tokens: Sequence[int],
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample next token from raw 1D logits using temperature, top-k, top-p, and repetition penalty."""
    if rng is None:
        rng = np.random.default_rng()

    logits = logits.copy().astype(np.float64)

    # 1. Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        recent = set(generated_tokens[-64:])
        for t in recent:
            if t < len(logits):
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty

    # 2. Temperature scaling
    temp = max(temperature, 1e-5)
    logits /= temp

    # 3. Top-K filtering
    if 0 < top_k < len(logits):
        k_val = np.partition(logits, -top_k)[-top_k]
        logits[logits < k_val] = -np.inf

    # 4. Softmax
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)

    # 5. Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cum_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cum_probs, top_p)
        valid = sorted_indices[: max(1, cutoff + 1)]
        filtered = np.zeros_like(probs)
        filtered[valid] = probs[valid]
        probs = filtered / np.sum(filtered)

    return int(rng.choice(len(probs), p=probs))


def generate_stream(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stop_words: Optional[List[str]] = None,
    stop_tokens: Optional[List[int]] = None,
    use_cache: bool = True,
    seed: Optional[int] = None,
) -> Generator[str, None, None]:
    """Yield newly generated text pieces token-by-token for interactive terminal streaming."""
    xp = get_backend()
    rng = np.random.default_rng(seed)

    prompt_ids = tokenizer.encode(prompt, allowed_special=True)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    # Clamp prompt to fit within context length
    max_prompt_len = max(1, model.config.context_length - 2)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    generated_ids = list(prompt_ids)
    stop_token_set = set(stop_tokens or [])
    stop_token_set.add(tokenizer.eos_id)
    stop_token_set.add(tokenizer.pad_id)
    if tokenizer.user_id:
        stop_token_set.add(tokenizer.user_id)

    stop_words = stop_words or ["<user>", "<eos>", "\nUser:", "\nHuman:"]

    # Initialize KV cache list for each layer
    kv_caches = [None] * model.config.num_layers if use_cache else None

    # Prefill step
    ctx = xp.asarray([prompt_ids], dtype=xp.int64)
    logits, _, kv_caches = model.forward(ctx, start_pos=0, kv_caches=kv_caches)
    next_logits = to_cpu(logits[0, -1])

    accumulated_text = ""

    for _ in range(max_new_tokens):
        next_id = sample_next_token(
            next_logits,
            generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            rng=rng,
        )
        generated_ids.append(next_id)

        if next_id in stop_token_set:
            break

        # Decode newly generated token
        piece = tokenizer.decode([next_id], skip_special_tokens=True)
        accumulated_text += piece

        # Check for stop words in accumulated response
        stop_triggered = False
        for sw in stop_words:
            if sw in accumulated_text:
                stop_triggered = True
                break
        if stop_triggered:
            break

        yield piece

        if len(generated_ids) >= model.config.context_length:
            break

        # Decode step forward using KV cache
        if use_cache and kv_caches is not None:
            step_input = xp.asarray([[next_id]], dtype=xp.int64)
            start_pos = len(generated_ids) - 1
            logits, _, kv_caches = model.forward(step_input, start_pos=start_pos, kv_caches=kv_caches)
            next_logits = to_cpu(logits[0, -1])
        else:
            ctx_slice = generated_ids[-model.config.context_length :]
            logits, _, _ = model.forward(xp.asarray([ctx_slice], dtype=xp.int64), start_pos=0)
            next_logits = to_cpu(logits[0, -1])


def generate_text(
    model: GPT,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stop_words: Optional[List[str]] = None,
    stop_tokens: Optional[List[int]] = None,
    use_cache: bool = True,
    seed: Optional[int] = None,
) -> str:
    """Generate complete text output string."""
    pieces = list(
        generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words,
            stop_tokens=stop_tokens,
            use_cache=use_cache,
            seed=seed,
        )
    )
    return "".join(pieces)
