"""Data loaders for standard text corpora and masked instruction/chat datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from tokenizer.bpe import ByteBPETokenizer, SpecialTokens


class TextDataset:
    """Standard pre-training dataset chunker with train/validation split."""

    def __init__(
        self,
        token_ids: Union[List[int], np.ndarray],
        context_length: int,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.context_length = context_length
        tokens = np.asarray(token_ids, dtype=np.int64)

        if len(tokens) <= context_length + 1:
            raise ValueError(
                f"Token count ({len(tokens)}) must be greater than context_length + 1 ({context_length + 1})"
            )

        split_idx = int(len(tokens) * (1.0 - val_ratio))
        self.train_tokens = tokens[:split_idx]
        self.val_tokens = tokens[split_idx:]
        self.rng = np.random.default_rng(seed)

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = self.train_tokens if split == "train" else self.val_tokens
        max_start = len(data) - self.context_length - 1
        if max_start <= 0:
            data = self.train_tokens
            max_start = len(data) - self.context_length - 1

        starts = self.rng.integers(0, max_start, size=batch_size)
        x = np.stack([data[s : s + self.context_length] for s in starts])
        y = np.stack([data[s + 1 : s + self.context_length + 1] for s in starts])
        return x, y


class ChatDataset:
    """Chat & Instruction dataset with role masking.
    
    Only tokens produced by the assistant have mask=1.0 (gradients & loss active).
    System and User prompts have mask=0.0 to focus learning on high-quality responses.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: ByteBPETokenizer,
        context_length: int,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rng = np.random.default_rng(seed)

        # Pre-process conversations into tokenized sequences with loss masks
        self.processed: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for sample in samples:
            encoded = self._encode_sample(sample)
            if encoded is not None:
                self.processed.append(encoded)

        if not self.processed:
            raise ValueError("No valid samples could be extracted from chat dataset.")

        split_idx = max(1, int(len(self.processed) * (1.0 - val_ratio)))
        self.train_samples = self.processed[:split_idx]
        self.val_samples = self.processed[split_idx:] if split_idx < len(self.processed) else self.processed

    def _encode_sample(self, sample: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        tok = self.tokenizer
        messages = sample.get("messages")
        if not messages:
            # Handle prompt/response format
            instruction = sample.get("instruction") or sample.get("prompt") or ""
            response = sample.get("response") or sample.get("output") or ""
            if not instruction or not response:
                return None
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]

        all_tokens: List[int] = []
        all_masks: List[float] = []

        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()

            if role == "system":
                role_prefix = f"<system>{content}"
                t_ids = tok.encode(role_prefix, allowed_special=True)
                all_tokens.extend(t_ids)
                all_masks.extend([0.0] * len(t_ids))
            elif role == "user":
                role_prefix = f"<user>{content}"
                t_ids = tok.encode(role_prefix, allowed_special=True)
                all_tokens.extend(t_ids)
                all_masks.extend([0.0] * len(t_ids))
            elif role == "assistant":
                role_prefix = f"<assistant>{content}<eos>"
                t_ids = tok.encode(role_prefix, allowed_special=True)
                all_tokens.extend(t_ids)
                # First token is <assistant>, which is part of prompt setup
                # Content and <eos> are to be learned by the assistant
                all_masks.append(0.0)
                all_masks.extend([1.0] * (len(t_ids) - 1))

        if len(all_tokens) < 3:
            return None

        # Truncate or pad to context_length + 1
        tokens_arr = np.array(all_tokens[: self.context_length + 1], dtype=np.int64)
        masks_arr = np.array(all_masks[: self.context_length + 1], dtype=np.float32)

        if len(tokens_arr) < self.context_length + 1:
            pad_len = (self.context_length + 1) - len(tokens_arr)
            tokens_arr = np.pad(tokens_arr, (0, pad_len), constant_values=tok.pad_id)
            masks_arr = np.pad(masks_arr, (0, pad_len), constant_values=0.0)

        x = tokens_arr[:-1]
        y = tokens_arr[1:]
        mask = masks_arr[1:]  # target mask aligns with y
        return x, y, mask

    def get_batch(
        self,
        batch_size: int,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pool = self.train_samples if split == "train" else self.val_samples
        indices = self.rng.integers(0, len(pool), size=batch_size)
        x = np.stack([pool[i][0] for i in indices])
        y = np.stack([pool[i][1] for i in indices])
        mask = np.stack([pool[i][2] for i in indices])
        return x, y, mask

    @classmethod
    def load_json(
        cls,
        path: Union[str, Path],
        tokenizer: ByteBPETokenizer,
        context_length: int,
        val_ratio: float = 0.05,
    ) -> "ChatDataset":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        samples = raw if isinstance(raw, list) else [raw]
        return cls(samples, tokenizer, context_length, val_ratio)
