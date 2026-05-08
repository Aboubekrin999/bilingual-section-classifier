"""Tests for the Parquet writer."""


from src.datasets import Record
from src.labels import Section
from src.parquet import read_records_table, write_records


def _make_records(n: int) -> list[Record]:
    return [
        Record(
            text=f"sentence {i}",
            language="en" if i % 2 == 0 else "fr",
            label=Section.METHODS,
            source="hal",
        )
        for i in range(n)
    ]


class TestRoundTrip:
    def test_count_matches_input(self, tmp_path):
        records = _make_records(20)
        n = write_records(records, tmp_path / "out.parquet")
        assert n == 20

    def test_columns_present(self, tmp_path):
        write_records(_make_records(3), tmp_path / "out.parquet")
        table = read_records_table(tmp_path / "out.parquet")
        assert set(table.column_names) == {"text", "language", "label", "source"}

    def test_text_preserved(self, tmp_path):
        write_records(_make_records(5), tmp_path / "out.parquet")
        table = read_records_table(tmp_path / "out.parquet")
        texts = table.column("text").to_pylist()
        assert texts == [f"sentence {i}" for i in range(5)]

    def test_label_serialised_as_string_value(self, tmp_path):
        write_records(_make_records(2), tmp_path / "out.parquet")
        table = read_records_table(tmp_path / "out.parquet")
        labels = table.column("label").to_pylist()
        # The Section enum value is the canonical string we want stored.
        assert labels == ["methods", "methods"]


class TestEmpty:
    def test_empty_records_writes_empty_table(self, tmp_path):
        n = write_records([], tmp_path / "out.parquet")
        assert n == 0
        table = read_records_table(tmp_path / "out.parquet")
        assert table.num_rows == 0
        assert set(table.column_names) == {"text", "language", "label", "source"}


class TestUnicode:
    def test_french_text_preserved(self, tmp_path):
        records = [
            Record(
                text="Les méthodes utilisées sont décrites ci-dessous.",
                language="fr",
                label=Section.METHODS,
                source="hal",
            )
        ]
        write_records(records, tmp_path / "fr.parquet")
        table = read_records_table(tmp_path / "fr.parquet")
        assert table.column("text").to_pylist() == [
            "Les méthodes utilisées sont décrites ci-dessous."
        ]
