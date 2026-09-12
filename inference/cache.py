"""Key-Value Cache structures for fast linear-time autoregressive decoding."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


class KVCacheManager:
    """Manages layer-wise (Key, Value) tensors for transformer inference."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.caches: List[Optional[Tuple[Any, Any]]] = [None] * num_layers

    def reset(self) -> None:
        self.caches = [None] * self.num_layers

    def get(self, layer_idx: int) -> Optional[Tuple[Any, Any]]:
        return self.caches[layer_idx]

    def set(self, layer_idx: int, key: Any, value: Any) -> None:
        self.caches[layer_idx] = (key, value)

    def current_seq_len(self) -> int:
        for c in self.caches:
            if c is not None and c[0] is not None:
                return c[0].shape[2]
        return 0
