"""
Check that every pinned dataset URL still resolves.

The shas in ``src.download`` are the only thing standing between this
corpus and silent data drift — but a pin that points at a commit which
does not exist is worse than no pin at all: it fails at download time,
several frames deep, with a bare 404.

Run this after changing a sha, and in CI if the network is available:

    python -m scripts.verify_sources

Exits non-zero and names every URL that did not return 200.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

from src.download import CSABSTRUCT_BASE, PUBMED_RCT_BASE

SOURCES: dict[str, list[str]] = {
    PUBMED_RCT_BASE: ["train.txt", "dev.txt", "test.txt"],
    CSABSTRUCT_BASE: ["train.jsonl", "dev.jsonl", "test.jsonl"],
}


def check(url: str) -> tuple[bool, str]:
    """HEAD-equivalent check. Returns ``(ok, detail)``."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            size = response.headers.get("Content-Length", "unknown")
            return True, f"{response.status} ({size} bytes)"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {exc.reason}"


def main() -> int:
    failures: list[str] = []

    for base, filenames in SOURCES.items():
        print(f"\n{base}")
        for filename in filenames:
            url = f"{base}/{filename}"
            ok, detail = check(url)
            print(f"  {'ok  ' if ok else 'FAIL'} {filename:<12} {detail}")
            if not ok:
                failures.append(url)

    if failures:
        print(f"\n{len(failures)} URL(s) did not resolve:")
        for url in failures:
            print(f"  {url}")
        print(
            "\nA pinned commit sha or a repository layout has changed. "
            "Update src/download.py deliberately, then re-run the dataset build."
        )
        return 1

    print("\nAll pinned sources resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
