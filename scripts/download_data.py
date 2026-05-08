"""
Fetch PubMed-RCT 20k and CSAbstruct into ``data/``.

Usage:
    python -m scripts.download_data

Idempotent: re-running overwrites the existing files. Pinned commit
shas in ``src.download`` make every run reproducible.
"""

from __future__ import annotations

from pathlib import Path

from src.download import download_csabstruct, download_pubmed_rct


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    print("Downloading PubMed-RCT 20k…")
    pubmed_paths = download_pubmed_rct(data_dir / "pubmed_rct")
    for split, path in pubmed_paths.items():
        size = path.stat().st_size
        print(f"  {split}: {path.relative_to(repo_root)} ({size:,} bytes)")

    print("Downloading CSAbstruct…")
    csab_paths = download_csabstruct(data_dir / "csabstruct")
    for split, path in csab_paths.items():
        size = path.stat().st_size
        print(f"  {split}: {path.relative_to(repo_root)} ({size:,} bytes)")


if __name__ == "__main__":
    main()
