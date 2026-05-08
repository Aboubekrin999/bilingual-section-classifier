"""
Compose source loaders + stratified splitter + parquet writer into one
``build_dataset`` call. The CLI in ``scripts/build_dataset.py`` is a
thin shell over this; importing this module directly is also fine for
notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from src.datasets import Record, load_csabstruct, load_hal, load_pubmed_rct
from src.parquet import write_records
from src.splits import SplitResult, stratified_split


@dataclass(frozen=True)
class SourceFiles:
    """Optional per-source file paths. Missing sources are skipped."""

    pubmed_rct: Path | None = None
    csabstruct: Path | None = None
    hal: Path | None = None


@dataclass(frozen=True)
class BuildResult:
    """Counts written for each split and total source records loaded."""

    n_loaded: int
    n_train: int
    n_val: int
    n_test: int


def load_all(sources: SourceFiles) -> list[Record]:
    """Load every available source into a single list of unified Records."""
    records: list[Record] = []
    if sources.pubmed_rct is not None:
        records.extend(load_pubmed_rct(_iter_lines(sources.pubmed_rct)))
    if sources.csabstruct is not None:
        records.extend(load_csabstruct(_iter_lines(sources.csabstruct)))
    if sources.hal is not None:
        records.extend(load_hal(_iter_lines(sources.hal)))
    return records


def build_dataset(
    sources: SourceFiles,
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 13,
) -> BuildResult:
    """
    End-to-end: load → stratified-split → write three parquet files.

    Output layout:

        output_dir/
            train.parquet
            val.parquet
            test.parquet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all(sources)
    splits: SplitResult = stratified_split(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    n_train = write_records(splits.train, output_dir / "train.parquet")
    n_val = write_records(splits.val, output_dir / "val.parquet")
    n_test = write_records(splits.test, output_dir / "test.parquet")

    return BuildResult(
        n_loaded=len(records),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fh:
        yield from fh
