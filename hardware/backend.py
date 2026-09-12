"""Backend abstraction managing NumPy (CPU) and CuPy (CUDA GPU)."""

from __future__ import annotations

import sys
from typing import Any, Tuple

import numpy as np_cpu

try:
    import cupy as np_gpu
except Exception:
    np_gpu = None

# Active backend array module (defaults to numpy CPU)
xp = np_cpu
_active_device: str = "cpu"


def select_backend(device: str = "auto") -> str:
    """Select NumPy CPU or CuPy CUDA arrays.
    
    device: 'auto', 'cpu', or 'cuda'
    """
    global xp, _active_device
    use_gpu = False

    if device == "cuda":
        use_gpu = True
    elif device == "auto":
        if np_gpu is not None:
            try:
                use_gpu = np_gpu.cuda.runtime.getDeviceCount() > 0
            except Exception:
                use_gpu = False

    if use_gpu:
        if np_gpu is None:
            raise RuntimeError(
                "CuPy is not installed. To enable CUDA acceleration, run:\n"
                "pip install cupy-cuda12x (or cupy-cuda13x depending on your CUDA driver)"
            )
        try:
            np_gpu.cuda.Device(0).use()
            # Test simple allocation
            _ = np_gpu.zeros((2, 2), dtype=np_gpu.float32)
            xp = np_gpu
            _active_device = "cuda"
            return "cuda"
        except Exception as exc:
            print(f"[Warning] Failed to initialize CUDA GPU: {exc}. Falling back to CPU.", file=sys.stderr)
            xp = np_cpu
            _active_device = "cpu"
            return "cpu"
    else:
        xp = np_cpu
        _active_device = "cpu"
        return "cpu"


def get_backend():
    return xp


def get_active_device() -> str:
    return _active_device


def is_cuda() -> bool:
    return _active_device == "cuda"


def to_cpu(value: Any) -> np_cpu.ndarray:
    """Convert any array (NumPy or CuPy) to a standard NumPy CPU ndarray."""
    if np_gpu is not None and isinstance(value, np_gpu.ndarray):
        return np_gpu.asnumpy(value)
    return np_cpu.asarray(value)


def to_device(value: Any) -> Any:
    """Move an array to the currently active device (CPU or GPU)."""
    if _active_device == "cuda" and np_gpu is not None:
        if isinstance(value, np_gpu.ndarray):
            return value
        return np_gpu.asarray(value)
    else:
        if np_gpu is not None and isinstance(value, np_gpu.ndarray):
            return np_gpu.asnumpy(value)
        return np_cpu.asarray(value)


def indexed_add(target: Any, indices: Any, values: Any) -> None:
    """In-place scatter-add rows at specified indices on CPU or GPU."""
    if _active_device == "cuda" and np_gpu is not None and isinstance(target, np_gpu.ndarray):
        from cupyx import scatter_add
        scatter_add(target, indices, values)
    else:
        np_cpu.add.at(target, indices, values)


def synchronize() -> None:
    """Synchronize device execution (useful for accurate profiling)."""
    if _active_device == "cuda" and np_gpu is not None:
        np_gpu.cuda.Device(0).synchronize()
