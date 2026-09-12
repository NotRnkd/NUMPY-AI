"""Build the local mixed corpus used for the larger from-scratch model."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "english_mixed.txt"
SOURCES = [
    ROOT / "english.txt",
    ROOT / "corpus" / "custom" / "hf_chat_training_v2.txt",
    ROOT / "corpus" / "custom" / "assistant_training.txt",
]


def main() -> None:
    existing = [path for path in SOURCES if path.exists()]
    if not existing:
        raise FileNotFoundError("No source corpus files were found")
    with OUTPUT.open("w", encoding="utf-8") as destination:
        for path in existing:
            destination.write(f"\n\n===== SOURCE: {path.name} =====\n\n")
            destination.write(path.read_text(encoding="utf-8", errors="replace"))
            destination.write("\n")
    print(f"wrote {OUTPUT}")
    print(f"sources: {len(existing)}")
    print(f"size: {OUTPUT.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
