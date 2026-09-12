"""Download and prepare a small legal starter corpus for the NumPy GPT.

Sources:
* Project Gutenberg public-domain English books.
* Selected English Wikipedia extracts, which are openly licensed but require
  attribution under Wikipedia's license if redistributed.

This script does not download modern copyrighted novels or scrape paywalled
articles. Check the copyright status and license terms in your jurisdiction
before adding any other material.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "corpus" / "raw"
CUSTOM_DIR = ROOT / "corpus" / "custom"
OUTPUT_PATH = ROOT / "english.txt"

GUTENBERG_BOOKS = [
    ("pride_and_prejudice", "Pride and Prejudice — Jane Austen", 1342),
    ("alice_in_wonderland", "Alice's Adventures in Wonderland — Lewis Carroll", 11),
    ("frankenstein", "Frankenstein — Mary Shelley", 84),
    ("moby_dick", "Moby-Dick — Herman Melville", 2701),
    ("sherlock_holmes", "The Adventures of Sherlock Holmes — Arthur Conan Doyle", 1661),
    ("dorian_gray", "The Picture of Dorian Gray — Oscar Wilde", 174),
    ("the_time_machine", "The Time Machine — H. G. Wells", 35),
    ("dracula", "Dracula — Bram Stoker", 345),
    ("a_tale_of_two_cities", "A Tale of Two Cities — Charles Dickens", 98),
    ("little_women", "Little Women — Louisa May Alcott", 37106),
    ("common_sense", "Common Sense — Thomas Paine", 147),
    ("federalist_papers", "The Federalist Papers", 1404),
]

WIKIPEDIA_ARTICLES = [
    "Artificial intelligence",
    "Computer programming",
    "Python (programming language)",
    "Software engineering",
    "Algorithm",
    "Operating system",
    "Machine learning",
    "Natural language processing",
    "Transformer (deep learning architecture)",
    "Neural network",
]


def fetch(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "numpy-gpt-corpus-builder/1.0 (local educational project)",
            "Accept": "text/plain, application/json",
        },
    )
    with urlopen(request, timeout=60) as response:
        return response.read()


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def clean_gutenberg(text: str) -> str:
    start_markers = ("*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK")
    end_markers = ("*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK")
    start = 0
    for marker in start_markers:
        position = text.find(marker)
        if position >= 0:
            start = text.find("\n", position) + 1
            break
    end = len(text)
    for marker in end_markers:
        position = text.find(marker)
        if position >= 0:
            end = position
            break
    return clean_text(text[start:end])


def download_gutenberg() -> list[dict]:
    records = []
    for slug, title, ebook_id in GUTENBERG_BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        destination = RAW_DIR / f"{slug}.txt"
        print(f"Downloading {title} ...")
        raw = fetch(url).decode("utf-8", errors="replace")
        cleaned = clean_gutenberg(raw)
        destination.write_text(f"TITLE: {title}\n\n{cleaned}", encoding="utf-8")
        records.append({"title": title, "source": url, "file": str(destination.name), "license_note": "Project Gutenberg; verify local public-domain status."})
        time.sleep(0.25)
    return records


def download_wikipedia() -> list[dict]:
    records = []
    for title in WIKIPEDIA_ARTICLES:
        query = urlencode({
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "formatversion": "2",
        })
        url = f"https://en.wikipedia.org/w/api.php?{query}"
        print(f"Downloading Wikipedia extract: {title} ...")
        payload = json.loads(fetch(url).decode("utf-8"))
        pages = payload.get("query", {}).get("pages", [])
        if not pages or "extract" not in pages[0]:
            print(f"  skipped: no extract returned")
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        destination = RAW_DIR / f"wikipedia_{slug}.txt"
        text = clean_text(pages[0]["extract"])
        destination.write_text(f"TITLE: Wikipedia — {title}\n\n{text}", encoding="utf-8")
        records.append({"title": f"Wikipedia — {title}", "source": url, "file": str(destination.name), "license_note": "Wikipedia content; preserve attribution and comply with CC BY-SA."})
        time.sleep(0.25)
    return records


def load_custom_text() -> list[dict]:
    """Copy and normalize user-provided .txt files into the combined corpus."""
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for source_path in sorted(CUSTOM_DIR.glob("*.txt")):
        print(f"Adding your text: {source_path.name} ...")
        cleaned = clean_text(source_path.read_text(encoding="utf-8", errors="replace"))
        destination = RAW_DIR / f"custom_{source_path.name}"
        destination.write_text(cleaned, encoding="utf-8")
        records.append({"title": f"User text — {source_path.name}", "source": str(source_path), "file": destination.name, "license_note": "User-provided; confirm you have training rights."})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local public-domain/openly licensed English corpus")
    parser.add_argument("--skip-wikipedia", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = download_gutenberg()
    if not args.skip_wikipedia:
        records.extend(download_wikipedia())
    records.extend(load_custom_text())

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as combined:
        for record in records:
            source_path = RAW_DIR / record["file"]
            combined.write(f"\n\n===== {record['title']} =====\n\n")
            combined.write(source_path.read_text(encoding="utf-8"))

    (ROOT / "corpus" / "sources.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(records)} sources to {output_path}")
    print(f"Corpus size: {output_path.stat().st_size / (1024 * 1024):.2f} MiB")


if __name__ == "__main__":
    main()
