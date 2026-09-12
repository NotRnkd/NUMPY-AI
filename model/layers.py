"""Neural network layers: RMSNorm, LayerNorm, RoPE, SwiGLU, GELU, and activations."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np_cpu
from hardware.backend import get_backend, to_cpu


# ---------------------------------------------------------------------------
# Normalization Layers
# ---------------------------------------------------------------------------

def rms_norm_forward(
    x: Any,
    gamma: Any,
    eps: float = 1e-5,
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """RMSNorm forward pass.
    
    y = (x / sqrt(mean(x^2) + eps)) * gamma
    """
    xp = get_backend()
    # Compute root mean square
    variance = xp.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / xp.sqrt(variance + eps)
    x_hat = x * inv_rms
    out = x_hat * gamma
    cache = (x_hat, inv_rms, gamma)
    return out, cache


def rms_norm_backward(
    dout: Any,
    cache: Tuple[Any, Any, Any],
) -> Tuple[Any, Any]:
    """RMSNorm backward pass.
    
    Returns (dx, dgamma).
    """
    xp = get_backend()
    x_hat, inv_rms, gamma = cache
    dim = dout.shape[-1]
    reduce_axes = tuple(range(dout.ndim - 1))

    # dgamma = sum(dout * x_hat)
    dgamma = xp.sum(dout * x_hat, axis=reduce_axes)

    # dx = inv_rms * (dout * gamma - x_hat * mean(dout * gamma * x_hat))
    dout_gamma = dout * gamma
    sum_dout_xhat = xp.mean(dout_gamma * x_hat, axis=-1, keepdims=True)
    dx = inv_rms * (dout_gamma - x_hat * sum_dout_xhat)
    return dx, dgamma


def layer_norm_forward(
    x: Any,
    gamma: Any,
    beta: Any,
    eps: float = 1e-5,
) -> Tuple[Any, Tuple[Any, Any, Any, Tuple[int, ...]]]:
    """LayerNorm forward pass."""
    xp = get_backend()
    mean = xp.mean(x, axis=-1, keepdims=True)
    variance = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / xp.sqrt(variance + eps)
    x_hat = (x - mean) * inv_std
    out = gamma * x_hat + beta
    cache = (x_hat, inv_std, gamma, x.shape)
    return out, cache


def layer_norm_backward(
    dout: Any,
    cache: Tuple[Any, Any, Any, Tuple[int, ...]],
) -> Tuple[Any, Any, Any]:
    """LayerNorm backward pass."""
    xp = get_backend()
    x_hat, inv_std, gamma, original_shape = cache
    n = original_shape[-1]
    reduce_axes = tuple(range(dout.ndim - 1))
    dgamma = xp.sum(dout * x_hat, axis=reduce_axes)
    dbeta = xp.sum(dout, axis=reduce_axes)
    dx = (
        gamma
        * inv_std
        / n
        * (
            n * dout
            - xp.sum(dout, axis=-1, keepdims=True)
            - x_hat * xp.sum(dout * x_hat, axis=-1, keepdims=True)
        )
    )
    return dx, dgamma, dbeta


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding:
    """Rotary Position Embedding (RoPE) with precomputed cos/sin tables."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._cos_cached: Optional[Any] = None
        self._sin_cached: Optional[Any] = None
        self._build_tables()

    def _build_tables(self) -> None:
        xp = get_backend()
        half_dim = self.dim // 2
        inv_freq = 1.0 / (self.base ** (np_cpu.arange(0, half_dim, dtype=np_cpu.float32) / half_dim))
        t = np_cpu.arange(self.max_seq_len, dtype=np_cpu.float32)
        freqs = np_cpu.outer(t, inv_freq)  # (max_seq_len, half_dim)

        cos = np_cpu.cos(freqs)
        sin = np_cpu.sin(freqs)
        self._cos_cached = xp.asarray(cos, dtype=xp.float32)
        self._sin_cached = xp.asarray(sin, dtype=xp.float32)

    def apply(self, x: Any, start_pos: int = 0, inverse: bool = False) -> Any:
        """Apply RoPE to tensor x of shape (batch, heads, seq_len, head_dim).
        
        If inverse is True, applies the inverse rotation (used in backward pass).
        """
        xp = get_backend()
        b, h, t, d = x.shape
        half_dim = d // 2

        if self._cos_cached is None or start_pos + t > self._cos_cached.shape[0]:
            self.max_seq_len = max(self.max_seq_len * 2, start_pos + t + 256)
            self._build_tables()

        cos = self._cos_cached[start_pos : start_pos + t][None, None, :, :]  # (1, 1, t, half_dim)
        sin = self._sin_cached[start_pos : start_pos + t][None, None, :, :]

        if inverse:
            sin = -sin

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

        # Rotation: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        return xp.concatenate([rot_x1, rot_x2], axis=-1)


# ---------------------------------------------------------------------------
# Activations & FFN / MLP
# ---------------------------------------------------------------------------

def sigmoid(x: Any) -> Any:
    xp = get_backend()
    # Numerically safe sigmoid
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -30.0, 30.0)))


def silu(x: Any) -> Tuple[Any, Any]:
    """SiLU (Swish-1) forward: x * sigmoid(x)."""
    s = sigmoid(x)
    return x * s, s


def silu_backward(dout: Any, x: Any, s: Any) -> Any:
    """SiLU backward: d/dx (x * s) = s + x * s * (1 - s) = s * (1 + x * (1 - s))."""
    return dout * (s * (1.0 + x * (1.0 - s)))


def gelu_forward(x: Any) -> Tuple[Any, Any]:
    """Approximate GELU forward: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    xp = get_backend()
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_pi * (x + 0.044715 * (x ** 3))
    tanh_inner = xp.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_inner)
    cache = (x, tanh_inner, inner)
    return out, cache


def gelu_backward(dout: Any, cache: Tuple[Any, Any, Any]) -> Any:
    """Approximate GELU backward."""
    xp = get_backend()
    x, tanh_inner, inner = cache
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    dtanh = 1.0 - tanh_inner ** 2
    dinner_dx = sqrt_2_pi * (1.0 + 3.0 * 0.044715 * (x ** 2))
    dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * dtanh * dinner_dx
    return dout * dx


def relu_forward(x: Any) -> Tuple[Any, Any]:
    xp = get_backend()
    out = xp.maximum(x, 0.0)
    return out, x


def relu_backward(dout: Any, cache: Any) -> Any:
    return dout * (cache > 0.0)


# ---------------------------------------------------------------------------
# Cross-Entropy Loss with Numerically Stable Log-Sum-Exp
# ---------------------------------------------------------------------------

def cross_entropy_loss_and_grad(
    logits: Any,
    targets: Any,
    mask: Optional[Any] = None,
) -> Tuple[float, Any]:
    """Numerically stable cross-entropy loss and gradient.
    
    logits: (B * T, V) or (B, T, V)
    targets: (B * T,) or (B, T)
    mask: (B * T,) or (B, T) - optional loss weights (0.0 for masked tokens)
    
    Returns: (scalar loss, dlogits matching logits.shape)
    """
    xp = get_backend()
    original_shape = logits.shape
    v = original_shape[-1]
    flat_logits = logits.reshape(-1, v)
    flat_targets = targets.reshape(-1)

    # Stable log-sum-exp: logsumexp(z) = max(z) + log(sum(exp(z - max(z))))
    max_logits = xp.max(flat_logits, axis=-1, keepdims=True)
    shifted = flat_logits - max_logits
    exp_shifted = xp.exp(shifted)
    sum_exp = xp.sum(exp_shifted, axis=-1, keepdims=True)
    log_sum_exp = max_logits + xp.log(sum_exp + 1e-12)

    # Target log-probs: log_p(target) = logits[target] - logsumexp
    target_logits = flat_logits[xp.arange(flat_targets.size), flat_targets]
    target_loss = -(target_logits - log_sum_exp.squeeze(-1))

    if mask is not None:
        weights = mask.reshape(-1).astype(xp.float32)
    else:
        weights = xp.ones(flat_targets.size, dtype=xp.float32)

    normalizer = xp.maximum(xp.sum(weights), 1.0)
    loss = float(to_cpu(xp.sum(target_loss * weights) / normalizer))

    # Gradient: dlogits = (softmax(logits) - target_one_hot) * (weight / normalizer)
    probs = exp_shifted / sum_exp
    probs[xp.arange(flat_targets.size), flat_targets] -= 1.0
    scale = (weights / normalizer)[:, None]
    dlogits = (probs * scale).reshape(original_shape)

    return loss, dlogits
