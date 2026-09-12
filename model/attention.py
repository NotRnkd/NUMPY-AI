"""Multi-Head Causal Self-Attention with RoPE, causal masking, and KV caching."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from hardware.backend import get_backend
from model.layers import RotaryEmbedding


class CausalSelfAttention:
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        use_rope: bool = True,
        has_bias: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length
        self.use_rope = use_rope
        self.has_bias = has_bias
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=context_length * 2) if use_rope else None

    def forward(
        self,
        x: Any,
        w_qkv: Any,
        b_qkv: Optional[Any],
        w_proj: Any,
        b_proj: Optional[Any],
        start_pos: int = 0,
        kv_cache: Optional[Tuple[Any, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[Any, dict, Optional[Tuple[Any, Any]]]:
        """Forward pass for causal attention.
        
        x: (batch, time, embedding_dim)
        Returns: (output, cache_dict_for_backward, new_kv_cache)
        """
        xp = get_backend()
        b, t, c = x.shape
        h = self.num_heads
        d = self.head_dim
        scale = 1.0 / math.sqrt(d)

        # QKV projection
        qkv = x @ w_qkv
        if b_qkv is not None:
            qkv = qkv + b_qkv

        # Split into Q, K, V: (batch, time, 3, heads, head_dim) -> (batch, heads, time, head_dim)
        qkv = qkv.reshape(b, t, 3, h, d).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply Rotary Position Embeddings if enabled
        if self.use_rope and self.rope is not None:
            q = self.rope.apply(q, start_pos=start_pos)
            k = self.rope.apply(k, start_pos=start_pos)

        # Update or use KV cache if requested
        new_kv_cache = None
        if use_cache or kv_cache is not None:
            if kv_cache is not None and isinstance(kv_cache, tuple) and kv_cache[0] is not None:
                cached_k, cached_v = kv_cache
                k = xp.concatenate([cached_k, k], axis=2)
                v = xp.concatenate([cached_v, v], axis=2)
            new_kv_cache = (k, v)

        total_k_len = k.shape[2]

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # (b, h, t, total_k_len)

        # Apply causal mask only if sequence length > 1
        if t > 1:
            # Create boolean future mask where col > row + (total_k_len - t)
            # col index: 0..total_k_len-1, row index: 0..t-1
            row_idx = xp.arange(t)[:, None]
            col_idx = xp.arange(total_k_len)[None, :]
            future_mask = col_idx > (row_idx + (total_k_len - t))
            scores = xp.where(future_mask[None, None, :, :], -1e9, scores)

        # Numerically stable softmax
        max_scores = xp.max(scores, axis=-1, keepdims=True)
        exp_scores = xp.exp(scores - max_scores)
        attention = exp_scores / (xp.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        # Value aggregation
        attended = attention @ v  # (b, h, t, d)
        merged = attended.transpose(0, 2, 1, 3).reshape(b, t, c)

        # Final projection
        projected = merged @ w_proj
        if b_proj is not None:
            projected = projected + b_proj

        cache = {
            "x": x,
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "attention": attention,
            "merged": merged,
            "start_pos": start_pos,
        }
        return projected, cache, new_kv_cache

    def backward(
        self,
        dprojected: Any,
        cache: dict,
        w_qkv: Any,
        w_proj: Any,
    ) -> Tuple[Any, Any, Optional[Any], Any, Optional[Any]]:
        """Exact backward pass for causal attention."""
        xp = get_backend()
        c = self.embedding_dim
        h = self.num_heads
        d = self.head_dim
        scale = 1.0 / math.sqrt(d)
        b, t, _ = dprojected.shape

        x = cache["x"]
        q = cache["q"]
        k = cache["k"]
        v = cache["v"]
        attention = cache["attention"]
        merged = cache["merged"]
        start_pos = cache.get("start_pos", 0)

        # Backward through projection
        dw_proj = merged.reshape(-1, c).T @ dprojected.reshape(-1, c)
        db_proj = dprojected.sum(axis=(0, 1)) if self.has_bias else None
        dmerged = dprojected @ w_proj.T

        # Reshape to (b, h, t, d)
        dattended = dmerged.reshape(b, t, h, d).transpose(0, 2, 1, 3)

        # Backward through attended = attention @ v
        # dattended: (b, h, t, d), v: (b, h, t, d)
        dattention = dattended @ v.transpose(0, 1, 3, 2)  # (b, h, t, t)
        dv = attention.transpose(0, 1, 3, 2) @ dattended  # (b, h, t, d)

        # Backward through softmax
        # dscores = attention * (dattention - sum(dattention * attention, axis=-1, keepdims=True))
        sum_dattn_attn = xp.sum(dattention * attention, axis=-1, keepdims=True)
        dscores = attention * (dattention - sum_dattn_attn)

        # Zero out gradients at masked causal positions
        if t > 1:
            future_mask = xp.triu(xp.ones((t, t), dtype=bool), k=1)
            dscores = xp.where(future_mask[None, None, :, :], 0.0, dscores)

        # Backward through scores = (q @ k.T) * scale
        dq = (dscores @ k) * scale  # (b, h, t, d)
        dk = (dscores.transpose(0, 1, 3, 2) @ q) * scale  # (b, h, t, d)

        # Backward through RoPE if applied (inverse rotation)
        if self.use_rope and self.rope is not None:
            dq = self.rope.apply(dq, start_pos=start_pos, inverse=True)
            dk = self.rope.apply(dk, start_pos=start_pos, inverse=True)

        # Pack dq, dk, dv into dqkv
        # shape: (3, b, h, t, d) -> (b, t, 3, h, d) -> (b, t, 3 * c)
        dqkv = xp.stack([dq, dk, dv], axis=0)  # (3, b, h, t, d)
        dqkv = dqkv.transpose(1, 3, 0, 2, 4).reshape(b, t, 3 * c)

        # Backward through QKV linear projection
        dw_qkv = x.reshape(-1, c).T @ dqkv.reshape(-1, 3 * c)
        db_qkv = dqkv.sum(axis=(0, 1)) if self.has_bias else None
        dx = dqkv @ w_qkv.T

        return dx, dw_qkv, db_qkv, dw_proj, db_proj
