"""AdamW Optimizer with weight decay exclusion for 1D parameters and gradient clipping."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np_cpu
from hardware.backend import get_backend, to_cpu, to_device


class AdamW:
    def __init__(
        self,
        params: Dict[str, Any],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.step_count = 0

        xp = get_backend()
        self.exp_avg: Dict[str, Any] = {}
        self.exp_avg_sq: Dict[str, Any] = {}

        for name, p in self.params.items():
            self.exp_avg[name] = xp.zeros_like(p)
            self.exp_avg_sq[name] = xp.zeros_like(p)

    def _should_decay(self, name: str, param: Any) -> bool:
        # Never decay 1D parameters (biases, layer norm gains, position embeddings)
        if param.ndim < 2:
            return False
        if "ln" in name or "norm" in name or "wpe" in name:
            return False
        return True

    def clip_gradients(self, grads: Dict[str, Any]) -> float:
        """Clip gradients by global L2 norm."""
        if self.grad_clip <= 0.0:
            return 0.0

        xp = get_backend()
        total_sq = 0.0
        for g in grads.values():
            if g is not None:
                total_sq += float(to_cpu(xp.sum(g * g)))

        global_norm = math.sqrt(total_sq)
        if global_norm > self.grad_clip:
            scale = self.grad_clip / (global_norm + 1e-6)
            for name in grads:
                if grads[name] is not None:
                    grads[name] *= scale

        return global_norm

    def step(self, grads: Dict[str, Any], lr: Optional[float] = None) -> float:
        """Perform a single optimization step. Returns the global gradient norm."""
        current_lr = self.lr if lr is None else lr
        self.step_count += 1
        xp = get_backend()

        # 1. Gradient Clipping
        norm = self.clip_gradients(grads)

        bias_correction1 = 1.0 - (self.beta1 ** self.step_count)
        bias_correction2 = 1.0 - (self.beta2 ** self.step_count)

        # 2. Update parameters
        for name, p in self.params.items():
            g = grads.get(name)
            if g is None:
                continue

            m = self.exp_avg[name]
            v = self.exp_avg_sq[name]

            # Decoupled weight decay
            if self.weight_decay > 0.0 and self._should_decay(name, p):
                p -= current_lr * self.weight_decay * p

            # Update biased 1st and 2nd moment estimate
            m *= self.beta1
            m += (1.0 - self.beta1) * g

            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)

            # Compute bias-corrected moments
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Parameter update
            p -= current_lr * m_hat / (xp.sqrt(v_hat) + self.eps)

        return norm

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "lr": self.lr,
            **{f"m_{k}": to_cpu(v) for k, v in self.exp_avg.items()},
            **{f"v_{k}": to_cpu(v) for k, v in self.exp_avg_sq.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "step" in state:
            self.step_count = int(state["step"])
        if "lr" in state:
            self.lr = float(state["lr"])

        for name in self.params:
            m_key = f"m_{name}"
            v_key = f"v_{name}"
            if m_key in state:
                self.exp_avg[name] = to_device(state[m_key])
            if v_key in state:
                self.exp_avg_sq[name] = to_device(state[v_key])
