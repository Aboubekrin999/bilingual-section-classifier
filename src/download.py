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

import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Pin to a known-good commit sha so the corpus is reproducible. Update
# deliberately, with a re-run of the build + an EVAL re-run, if at all.
#
# Both shas are verified to exist and to serve all six files; see
# ``tests/test_download.py::TestPinnedSources``, which asserts the URLs are
# well-formed, and ``scripts/verify_sources.py``, which checks them live.
_PUBMED_RCT_SHA = "17ed2cb0590decfca0266add0c76f254f67232b4"
_CSABSTRUCT_SHA = "cf5ad6c663550dd8203f148cd703768d9ee86ff4"

PUBMED_RCT_BASE = (
    f"https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/"
    f"{_PUBMED_RCT_SHA}/PubMed_20k_RCT"
)
CSABSTRUCT_BASE = (
    f"https://raw.githubusercontent.com/allenai/sequential_sentence_classification/"
    f"{_CSABSTRUCT_SHA}/data/CSAbstruct"
)

Fetcher = Callable[[str], bytes]


class DownloadError(RuntimeError):
    """Raised when a fetch returns nothing or the source URL is missing."""


def http_fetch(url: str) -> bytes:
    """Default fetcher — single GET via ``urllib``.

    A 404 here almost always means a pinned sha or a repository path has
    moved, so the error says that rather than surfacing a bare HTTPError
    from six frames down.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - well-known URLs
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise DownloadError(
                f"{url} returned 404. The pinned commit sha or the repository "
                f"layout has changed; re-verify with scripts/verify_sources.py "
                f"and update the shas in this module deliberately."
            ) from exc
        raise DownloadError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DownloadError(f"could not reach {url}: {exc.reason}") from exc


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
