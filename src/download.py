"""
Dataset download helpers — pure stdlib HTTP, dep-free.

The fetch step is split out as an injectable callable so tests can
hand in a fake without spinning up a server. Production wiring uses
``urllib.request`` directly: each download is a single ~MB-scale GET
against a public GitHub raw URL, so a heavyweight HTTP client is
overkill.

Datasets are pinned to a specific commit sha rather than ``master`` so
data drift can't silently rewrite the training corpus underneath us
between runs.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path

# Pin to a known-good commit sha so the corpus is reproducible. Update
# deliberately, with a re-run of the build + an EVAL re-run, if at all.
_PUBMED_RCT_SHA = "39255b13ea4ecb9d35dee62de76c66b78f8acd86"
_CSABSTRUCT_SHA = "a23a040db66ddd57b2cdde9a7081f9aa61df5f50"

PUBMED_RCT_BASE = (
    f"https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/"
    f"{_PUBMED_RCT_SHA}/PubMed_20k_RCT"
)
CSABSTRUCT_BASE = (
    f"https://raw.githubusercontent.com/allenai/sequential_sentence_classification/"
    f"{_CSABSTRUCT_SHA}/sequential_sentence_classification/data/CSAbstruct"
)

Fetcher = Callable[[str], bytes]


class DownloadError(RuntimeError):
    """Raised when a fetch returns nothing or the source URL is missing."""


def http_fetch(url: str) -> bytes:
    """Default fetcher — single GET via ``urllib``."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - well-known URLs
        return response.read()


def download_pubmed_rct(
    output_dir: Path,
    *,
    fetch: Fetcher = http_fetch,
) -> dict[str, Path]:
    """
    Download the PubMed-RCT 20k splits to ``output_dir``.

    Returns a mapping of split name (``train`` / ``dev`` / ``test``)
    to the on-disk path each was written to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
    return _download_splits(PUBMED_RCT_BASE, splits, output_dir, fetch)


def download_csabstruct(
    output_dir: Path,
    *,
    fetch: Fetcher = http_fetch,
) -> dict[str, Path]:
    """Download the CSAbstruct splits to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    return _download_splits(CSABSTRUCT_BASE, splits, output_dir, fetch)


def _download_splits(
    base_url: str,
    splits: dict[str, str],
    output_dir: Path,
    fetch: Fetcher,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for split, filename in splits.items():
        url = f"{base_url}/{filename}"
        body = fetch(url)
        if not body:
            raise DownloadError(f"empty response for {url}")
        path = output_dir / filename
        path.write_bytes(body)
        out[split] = path
    return out
