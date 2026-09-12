"""Standalone script to train and save a Byte-level BPE tokenizer on a corpus."""

import argparse
from pathlib import Path
from tokenizer.bpe import ByteBPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a Byte-Level BPE Tokenizer")
    parser.add_argument("--data", type=str, required=True, help="Path to text training file")
    parser.add_argument("--output", type=str, default="tokenizer.vocab.json", help="Path to save tokenizer json")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocabulary size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum merge frequency")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {data_path}")

    print(f"Reading {data_path} ...")
    text = data_path.read_text(encoding="utf-8")
    print(f"Corpus size: {len(text):,} characters")

    tokenizer = ByteBPETokenizer.train(
        text,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        verbose=True,
    )
    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}")


if __name__ == "__main__":
    main()
