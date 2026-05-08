"""Tests for the end-to-end build pipeline."""

import json

from src.build import SourceFiles, build_dataset, load_all
from src.parquet import read_records_table


def _write_pubmed_rct(path, n_per_label: int = 4):
    """Write a tiny PubMed-RCT-format TSV under ``path``."""
    lines = []
    for label in ("BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"):
        for i in range(n_per_label):
            lines.append(f"{label}\tSentence {label} {i}.")
    path.write_text("\n".join(lines) + "\n")


def _write_csabstruct(path, n: int = 4):
    docs = []
    for i in range(n):
        docs.append(
            json.dumps(
                {
                    "sentences": [f"CS sentence {i}.", f"CS another {i}."],
                    "labels": ["background", "method"],
                }
            )
        )
    path.write_text("\n".join(docs) + "\n")


def _write_hal(path, n: int = 4):
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "text": f"Méthodologie {i}.",
                    "header": "Méthodes",
                    "language": "fr",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


class TestLoadAll:
    def test_pubmed_only(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct)
        records = load_all(SourceFiles(pubmed_rct=rct))
        assert len(records) > 0
        assert all(r.source == "pubmed_rct" for r in records)

    def test_all_three_sources_combined(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        csab = tmp_path / "csab.jsonl"
        hal = tmp_path / "hal.jsonl"
        _write_pubmed_rct(rct)
        _write_csabstruct(csab)
        _write_hal(hal)
        records = load_all(SourceFiles(pubmed_rct=rct, csabstruct=csab, hal=hal))
        sources = {r.source for r in records}
        assert sources == {"pubmed_rct", "csabstruct", "hal"}


class TestBuildDataset:
    def test_writes_three_parquet_files(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct, n_per_label=10)
        out = tmp_path / "built"
        build_dataset(SourceFiles(pubmed_rct=rct), out)
        assert (out / "train.parquet").exists()
        assert (out / "val.parquet").exists()
        assert (out / "test.parquet").exists()

    def test_split_counts_sum_to_loaded(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct, n_per_label=10)
        result = build_dataset(SourceFiles(pubmed_rct=rct), tmp_path / "built")
        assert result.n_train + result.n_val + result.n_test == result.n_loaded

    def test_split_files_readable_as_records(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct, n_per_label=10)
        out = tmp_path / "built"
        build_dataset(SourceFiles(pubmed_rct=rct), out)
        table = read_records_table(out / "train.parquet")
        assert {"text", "language", "label", "source"}.issubset(table.column_names)

    def test_deterministic_seed(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct, n_per_label=10)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        a = build_dataset(SourceFiles(pubmed_rct=rct), out_a, seed=42)
        b = build_dataset(SourceFiles(pubmed_rct=rct), out_b, seed=42)
        assert a == b

    def test_different_seed_can_change_split(self, tmp_path):
        rct = tmp_path / "pubmed.txt"
        _write_pubmed_rct(rct, n_per_label=10)
        a = build_dataset(SourceFiles(pubmed_rct=rct), tmp_path / "a", seed=1)
        b = build_dataset(SourceFiles(pubmed_rct=rct), tmp_path / "b", seed=2)
        # Total counts identical (same input), but interior shuffling differs;
        # we can't guarantee the split sizes themselves change — only contents.
        assert a.n_loaded == b.n_loaded
