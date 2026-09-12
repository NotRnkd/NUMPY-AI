#!/usr/bin/env python3
"""Prepare and convert ChatGPT datasets from voidful/awesome-chatgpt-dataset for NumPy-GPT.

Supports:
- Stanford Alpaca / Alpaca-Cleaned (instruction, input, output)
- Databricks Dolly 15k (instruction, context, response)
- ShareGPT / Vicuna (conversations: human / gpt turns)
- HC3 Human vs ChatGPT Comparison (question, chatgpt_answers)
- Standard OpenAI ChatML format (messages: role / content)
- Local JSON and JSONL files
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen


POPULAR_ONLINE_DATASETS = {
    "alpaca-sample": {
        "url": "https://raw.githubusercontent.com/gururise/Alpaca-Data-Cleaned/main/alpaca_data_cleaned.json",
        "description": "Stanford Alpaca Cleaned (52k instruction following)",
        "format": "alpaca",
    }
}


def normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert any ChatGPT dataset record into standardized messages format."""
    # 1. Standard messages
    if "messages" in raw and isinstance(raw["messages"], list):
        return {"messages": raw["messages"]}

    # 2. ShareGPT conversations format
    if "conversations" in raw and isinstance(raw["conversations"], list):
        msgs = []
        for turn in raw["conversations"]:
            role = str(turn.get("from", "") or turn.get("role", "")).lower()
            val = str(turn.get("value", "") or turn.get("content", "") or turn.get("text", "")).strip()
            if not val:
                continue
            if role in ("human", "user"):
                msgs.append({"role": "user", "content": val})
            elif role in ("gpt", "chatgpt", "assistant", "bot", "model"):
                msgs.append({"role": "assistant", "content": val})
            elif role in ("system",):
                msgs.append({"role": "system", "content": val})
        if msgs:
            return {"messages": msgs}

    # 3. Alpaca / Dolly / Instruction format
    instruction = str(raw.get("instruction") or raw.get("prompt") or raw.get("question") or "").strip()
    context = str(raw.get("input") or raw.get("context") or "").strip()
    response = str(raw.get("response") or raw.get("output") or raw.get("answer") or "").strip()

    if not response and "chatgpt_answers" in raw:
        answers = raw["chatgpt_answers"]
        response = answers[0] if isinstance(answers, list) and answers else str(answers)

    if instruction and response:
        user_content = f"{instruction}\n\nContext:\n{context}" if context else instruction
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
        }

    return None


def load_raw_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL file into a list of dicts."""
    content = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if content.startswith("["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare datasets from awesome-chatgpt-dataset for NumPy-GPT"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to local JSON or JSONL dataset file",
    )
    parser.add_argument(
        "--dataset",
        choices=["alpaca-sample"],
        help="Download and convert a curated public sample dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of conversation samples to include (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="corpus/chatgpt_training.json",
        help="Target output file path (.json for ChatDataset)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    raw_items: List[Dict[str, Any]] = []

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(f"Reading dataset from {path} ...")
        raw_items = load_raw_data(path)
    elif args.dataset:
        cfg = POPULAR_ONLINE_DATASETS[args.dataset]
        print(f"Downloading {cfg['description']} from {cfg['url']} ...")
        req = Request(cfg["url"], headers={"User-Agent": "NumPy-GPT-Dataset-Tool/1.0"})
        with urlopen(req, timeout=30) as resp:
            raw_items = json.loads(resp.read().decode("utf-8"))
    else:
        # Default: check if sample exists, or instruct user
        sample_path = Path("corpus/sample_chatgpt_dataset.json")
        if sample_path.exists():
            print(f"No source specified, using local {sample_path} ...")
            raw_items = load_raw_data(sample_path)
        else:
            parser.print_help()
            sys.exit(0)

    print(f"Loaded {len(raw_items):,} raw records.")

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        norm = normalize_record(item)
        if norm:
            normalized.append(norm)

    print(f"Successfully parsed {len(normalized):,} valid conversation samples.")

    # Shuffle and trim
    rng = random.Random(args.seed)
    rng.shuffle(normalized)
    if args.max_samples and len(normalized) > args.max_samples:
        normalized = normalized[: args.max_samples]
        print(f"Subsampled to {len(normalized):,} records.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved prepared dataset to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("\nYou can now train NumPy-GPT on this dataset using:")
    print(f"  python3 numpy_gpt.py train --data {out_path} --steps 200 --batch-size 4 --lr 3e-4")


if __name__ == "__main__":
    main()
