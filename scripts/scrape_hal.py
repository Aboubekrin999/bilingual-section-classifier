"""
Scrape French section-labelled paragraphs from HAL open-access papers.

Usage:
    python -m scripts.scrape_hal --target 800
    python -m scripts.scrape_hal --target 2000 --resume

Writes JSONL to ``data/hal/paragraphs.jsonl`` in the shape
``src.datasets.load_hal`` consumes. Resumable: docids already present in
the output are skipped, so an interrupted run continues where it stopped.

PDFs are fetched, read, and discarded — nothing is cached to disk. A HAL
PDF averages ~1 MB and a useful run covers hundreds of them, which is not
worth the space.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from src.hal import (
    DEFAULT_MIN_YEAR,
    HALError,
    extract_text,
    http_fetch,
    is_usable,
    looks_like_text,
    search,
    segment,
    to_records,
)

DEFAULT_OUTPUT = Path("data/hal/paragraphs.jsonl")

#: HAL is a public service run by a research institution. One request a
#: second is polite and still fills a corpus in well under an hour.
REQUEST_DELAY_SECONDS = 1.0

#: API page size. HAL accepts more; 100 keeps each failure cheap.
PAGE_SIZE = 100


def already_scraped(output: Path) -> set[str]:
    """docids present in a previous run's output."""
    if not output.exists():
        return set()
    seen: set[str] = set()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["docid"])
            except (ValueError, KeyError):
                continue
    return seen


def scrape_one(doc: dict) -> list[dict]:
    """Fetch and segment a single paper. Returns [] for anything unusable."""
    try:
        pdf = http_fetch(doc["fileMain_s"])
    except HALError:
        return []

    try:
        text = extract_text(pdf)
    except HALError:
        return []

    if not looks_like_text(text):
        return []

    segments = segment(text)
    if not is_usable(segments):
        return []

    return list(to_records(segments, docid=str(doc["docid"])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=800, help="papers to keep")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--resume", action="store_true", help="skip known docids")
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = already_scraped(args.output) if args.resume else set()
    if seen:
        print(f"Resuming — {len(seen):,} papers already scraped.")

    kept = 0
    examined = 0
    rejected: Counter[str] = Counter()
    mode = "a" if args.resume else "w"

    with args.output.open(mode, encoding="utf-8") as out:
        for page in range(args.max_pages):
            if kept >= args.target:
                break
            try:
                docs = search(
                    rows=PAGE_SIZE, start=page * PAGE_SIZE, min_year=args.min_year
                )
            except HALError as exc:
                print(f"search failed on page {page}: {exc}", file=sys.stderr)
                break
            if not docs:
                print("HAL returned no more documents.")
                break

            for doc in docs:
                if kept >= args.target:
                    break
                docid = str(doc["docid"])
                if docid in seen:
                    continue
                seen.add(docid)
                examined += 1

                records = scrape_one(doc)
                time.sleep(REQUEST_DELAY_SECONDS)

                if not records:
                    rejected["no usable sections"] += 1
                    continue

                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                kept += 1

                if kept % 25 == 0:
                    print(f"  kept {kept}/{args.target} (examined {examined})")

    print(f"\nKept {kept} papers from {examined} examined.")
    for reason, count in rejected.most_common():
        print(f"  rejected — {reason}: {count}")
    print(f"Output: {args.output}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
