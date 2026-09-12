"""Download and convert selected public Hugging Face datasets to local chat text.

The NumPy GPT trainer consumes plain text. This script keeps the original
datasets separate from the generated training file and writes a small manifest
with the source and license information.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "corpus" / "custom" / "hf_chat_training_v2.txt"
SYSTEM = (
    "SYSTEM: You are a helpful offline English assistant. Give clear, useful "
    "answers, show steps when needed, and do not invent facts or tool results."
)


def text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def make_record(user: str, answer: str, context: str = "") -> str:
    user = text(user)
    answer = text(answer)
    context = text(context)
    if context:
        user = f"{user}\n\nContext:\n{context}"
    if not user or not answer:
        return ""
    return f"USER: {user}\nASSISTANT: {answer}\n\n"


def load_dolly() -> list[str]:
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    records = []
    for row in dataset:
        record = make_record(row.get("instruction"), row.get("response"), row.get("context"))
        if record:
            records.append(record)
    return records


def load_theoremqa() -> list[str]:
    dataset = load_dataset("TIGER-Lab/TheoremQA", split="test")
    records = []
    for row in dataset:
        question = text(row.get("Question"))
        answer = text(row.get("Answer"))
        answer_type = text(row.get("Answer_type"))
        if answer_type:
            answer = f"{answer} (answer type: {answer_type})"
        record = make_record(question, answer)
        if record:
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local chat data for the NumPy GPT")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    records = load_dolly()
    dolly_count = len(records)
    theorem_records = load_theoremqa()
    records.extend(theorem_records)
    random.Random(args.seed).shuffle(records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{SYSTEM}\n\n" + "".join(records), encoding="utf-8")

    manifest = output.with_suffix(".sources.json")
    manifest.write_text(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "sources": [
                    {
                        "dataset": "databricks/databricks-dolly-15k",
                        "records": dolly_count,
                        "license": "CC BY-SA 3.0; preserve attribution and share-alike terms",
                    },
                    {
                        "dataset": "TIGER-Lab/TheoremQA",
                        "records": len(theorem_records),
                        "license": "MIT",
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(records):,} chat records to {output}")
    print(f"wrote source manifest to {manifest}")
    print(f"training text size: {output.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
