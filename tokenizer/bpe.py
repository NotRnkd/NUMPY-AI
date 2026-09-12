"""Enhanced Byte-Level BPE Tokenizer with fast caching, special tokens, and legacy support."""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

Pair = Tuple[int, int]


class SpecialTokens:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    SYSTEM = "<system>"
    USER = "<user>"
    ASSISTANT = "<assistant>"

    DEFAULT_DICT = {
        PAD: 256,
        BOS: 257,
        EOS: 258,
        UNK: 259,
        SYSTEM: 260,
        USER: 261,
        ASSISTANT: 262,
    }


# High-efficiency pre-tokenization regex preserving contractions, numbers, words, and whitespace
_PRETOKEN_REGEX = re.compile(
    r"""'(?:[sStT]|re|ve|m|ll|d)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


class ByteBPETokenizer:
    BASE_VOCAB_SIZE = 256

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int, int]]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
    ) -> None:
        self.special_tokens = dict(special_tokens) if special_tokens else dict(SpecialTokens.DEFAULT_DICT)
        self.special_to_id = self.special_tokens
        self.id_to_special = {idx: token for token, idx in self.special_tokens.items()}

        self.merges: List[Tuple[int, int, int]] = merges or []
        self.merge_ranks: Dict[Pair, int] = {
            (left, right): rank for rank, (left, right, _new_id) in enumerate(self.merges)
        }

        # Initialize base byte mapping
        self.token_bytes: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for left, right, new_id in self.merges:
            if left in self.token_bytes and right in self.token_bytes:
                self.token_bytes[new_id] = self.token_bytes[left] + self.token_bytes[right]

        # Fast cache for encoded pretoken pieces
        self._piece_cache: Dict[str, List[int]] = {}

        # Precompile special token regex pattern for fast extraction
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(k) for k in sorted_specials) + ")"
            self._special_regex = re.compile(pattern)
        else:
            self._special_regex = None

    @property
    def pad_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.PAD, 256)

    @property
    def bos_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.BOS, 257)

    @property
    def eos_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.EOS, 258)

    @property
    def unk_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.UNK, 259)

    @property
    def system_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.SYSTEM, 260)

    @property
    def user_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.USER, 261)

    @property
    def assistant_id(self) -> int:
        return self.special_to_id.get(SpecialTokens.ASSISTANT, 262)

    @property
    def vocab_size(self) -> int:
        if not self.merges:
            return self.BASE_VOCAB_SIZE + len(self.special_tokens)
        max_id = max(
            max(self.special_tokens.values(), default=255),
            max(m[2] for m in self.merges),
        )
        return max_id + 1

    def __len__(self) -> int:
        return self.vocab_size

    @staticmethod
    def _pieces(text: str) -> List[str]:
        return _PRETOKEN_REGEX.findall(text)

    @staticmethod
    def _merge_sequence(sequence: List[int], pair: Pair, new_id: int) -> List[int]:
        result: List[int] = []
        i = 0
        n = len(sequence)
        while i < n:
            if i + 1 < n and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(sequence[i])
                i += 1
        return result

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None,
        verbose: bool = True,
    ) -> "ByteBPETokenizer":
        if special_tokens is None:
            special_tokens = dict(SpecialTokens.DEFAULT_DICT)

        base_count = cls.BASE_VOCAB_SIZE + len(special_tokens)
        if vocab_size <= base_count:
            if verbose:
                print(f"[Tokenizer] Requested vocab_size {vocab_size} <= base count {base_count}, using base.")
            return cls([], special_tokens=special_tokens)

        # Count frequencies of pre-tokenized pieces
        piece_counts = Counter(cls._pieces(text))
        sequences: List[List[int]] = []
        frequencies: List[int] = []
        for piece, frequency in piece_counts.items():
            sequences.append(list(piece.encode("utf-8")))
            frequencies.append(frequency)

        pair_counts: Dict[Pair, int] = defaultdict(int)
        pair_sequences: Dict[Pair, Set[int]] = defaultdict(set)
        for seq_idx, seq in enumerate(sequences):
            freq = frequencies[seq_idx]
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += freq
                pair_sequences[pair].add(seq_idx)

        heap: List[Tuple[int, int, int]] = [
            (-count, left, right) for (left, right), count in pair_counts.items()
        ]
        heapq.heapify(heap)
        merges: List[Tuple[int, int, int]] = []
        next_id = max(special_tokens.values()) + 1

        target_merges = vocab_size - base_count

        while len(merges) < target_merges and heap:
            while heap:
                neg_count, left, right = heapq.heappop(heap)
                pair = (left, right)
                actual_count = pair_counts.get(pair, 0)
                if actual_count == -neg_count and actual_count >= min_frequency:
                    break
            else:
                break

            affected_indices = list(pair_sequences.get(pair, set()))
            for idx in affected_indices:
                old_seq = sequences[idx]
                freq = frequencies[idx]

                # Deduct old adjacent pairs
                for old_pair in zip(old_seq, old_seq[1:]):
                    pair_counts[old_pair] -= freq
                    pair_sequences[old_pair].discard(idx)
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_sequences.pop(old_pair, None)

                # Merge
                new_seq = cls._merge_sequence(old_seq, pair, next_id)
                sequences[idx] = new_seq

                # Add new adjacent pairs
                for new_pair in zip(new_seq, new_seq[1:]):
                    pair_counts[new_pair] += freq
                    pair_sequences[new_pair].add(idx)
                    heapq.heappush(heap, (-pair_counts[new_pair], new_pair[0], new_pair[1]))

            merges.append((left, right, next_id))
            next_id += 1

        if verbose:
            print(f"[Tokenizer] Trained {len(merges)} merges. Total vocab size: {base_count + len(merges)}")

        return cls(merges, special_tokens=special_tokens)

    def _encode_piece(self, piece: str) -> List[int]:
        if piece in self._piece_cache:
            return self._piece_cache[piece]

        sequence = list(piece.encode("utf-8"))
        while len(sequence) > 1:
            best_pair = None
            best_rank = float("inf")
            for pair in zip(sequence, sequence[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break
            new_id = self.merges[best_rank][2]
            sequence = self._merge_sequence(sequence, best_pair, new_id)

        if len(self._piece_cache) < 100000:
            self._piece_cache[piece] = sequence
        return sequence

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Union[bool, Set[str]] = True,
    ) -> List[int]:
        """Encode text to token IDs.
        
        If allowed_special is True, recognized special tokens (e.g. <user>, <assistant>)
        are tokenized into their dedicated IDs instead of being split into bytes.
        """
        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.bos_id)

        if not text:
            if add_eos:
                token_ids.append(self.eos_id)
            return token_ids

        # If special tokens are allowed, parse them out first
        if allowed_special and self._special_regex is not None:
            parts = self._special_regex.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.special_to_id:
                    token_ids.append(self.special_to_id[part])
                else:
                    for piece in self._pieces(part):
                        token_ids.extend(self._encode_piece(piece))
        else:
            for piece in self._pieces(text):
                token_ids.extend(self._encode_piece(piece))

        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        output = bytearray()
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.token_bytes:
                output.extend(self.token_bytes[token_id])
            elif token_id in self.id_to_special:
                if not skip_special_tokens:
                    output.extend(self.id_to_special[token_id].encode("utf-8"))
            else:
                # Unknown byte or token
                output.extend(b"?")
        return bytes(output).decode("utf-8", errors="replace")

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        payload = {
            "format": "byte-bpe-v2",
            "special_tokens": self.special_tokens,
            "merges": [[left, right, new_id] for left, right, new_id in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ByteBPETokenizer":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Handle legacy character-level vocabulary list: ["<unk>", "\n", " ", ...]
        if isinstance(raw, list):
            # Migrate character list to special tokens
            specials = {char: i for i, char in enumerate(raw)}
            return cls([], special_tokens=specials)

        # Handle byte-bpe-v1 or byte-bpe-v2 format
        if isinstance(raw, dict):
            merges = [tuple(m) for m in raw.get("merges", [])]
            special_tokens = raw.get("special_tokens")
            return cls(merges=merges, special_tokens=special_tokens)

        raise ValueError(f"Unrecognized vocabulary format in {path}")
