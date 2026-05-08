"""
Build the unified train/val/test parquet splits from downloaded sources.

Usage:
    python -m scripts.build_dataset

Looks for each source in its conventional ``data/`` location. Sources
that aren't present on disk are skipped — useful while only PubMed-RCT
and CSAbstruct are available; HAL adds itself once the scrape lands.
"""

from __future__ import annotations

from pathlib import Path

from src.build import SourceFiles, build_dataset


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    sources = SourceFiles(
        pubmed_rct=_optional(data_dir / "pubmed_rct" / "train.txt"),
        csabstruct=_optional(data_dir / "csabstruct" / "train.jsonl"),
        hal=_optional(data_dir / "hal" / "records.jsonl"),
    )
    if all(p is None for p in (sources.pubmed_rct, sources.csabstruct, sources.hal)):
        raise SystemExit(
            "no source data found under data/. "
            "run `python -m scripts.download_data` first."
        )

    output_dir = data_dir / "built"
    print(f"Building dataset → {output_dir.relative_to(repo_root)}/")
    result = build_dataset(sources, output_dir)
    print(f"  loaded:  {result.n_loaded:,} records")
    print(f"  train:   {result.n_train:,}")
    print(f"  val:     {result.n_val:,}")
    print(f"  test:    {result.n_test:,}")


def _optional(path: Path) -> Path | None:
    return path if path.exists() else None


if __name__ == "__main__":
    main()
