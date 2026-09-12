"""Backward-compatible wrapper importing from tokenizer.bpe."""

from tokenizer.bpe import ByteBPETokenizer, SpecialTokens

__all__ = ["ByteBPETokenizer", "SpecialTokens"]
