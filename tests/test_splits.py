"""Tests for the stratified splitter."""

import pytest

from src.datasets import Record
from src.labels import Section
from src.splits import stratified_split


def _make_records(label: Section, language: str, n: int) -> list[Record]:
    """Build n records with distinct text so set-equality checks work."""
    return [
        Record(text=f"{label.value}-{language}-{i}", language=language, label=label, source="hal")
        for i in range(n)
    ]


class TestSizing:
    def test_basic_80_10_10_split(self):
        records = _make_records(Section.METHODS, "en", 100)
        result = stratified_split(records, val_ratio=0.1, test_ratio=0.1, seed=0)
        assert len(result.train) + len(result.val) + len(result.test) == 100
        assert len(result.test) == 10
        assert len(result.val) == 10
        assert len(result.train) == 80

    def test_disjoint_splits(self):
        records = _make_records(Section.RESULTS, "fr", 50)
        result = stratified_split(records, seed=42)
        train_set = {r.text for r in result.train}
        val_set = {r.text for r in result.val}
        test_set = {r.text for r in result.test}
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_no_records_lost(self):
        records = _make_records(Section.METHODS, "en", 17)
        result = stratified_split(records, seed=1)
        all_split = {r.text for r in result.train + result.val + result.test}
        assert all_split == {r.text for r in records}


class TestStratification:
    def test_label_distribution_preserved_per_split(self):
        records = (
            _make_records(Section.METHODS, "en", 100)
            + _make_records(Section.RESULTS, "en", 100)
            + _make_records(Section.METHODS, "fr", 100)
            + _make_records(Section.RESULTS, "fr", 100)
        )
        result = stratified_split(records, val_ratio=0.1, test_ratio=0.1, seed=7)
        # Each split should contain all four (label, language) combinations.
        for split in (result.train, result.val, result.test):
            cells = {(r.label, r.language) for r in split}
            assert cells == {
                (Section.METHODS, "en"),
                (Section.RESULTS, "en"),
                (Section.METHODS, "fr"),
                (Section.RESULTS, "fr"),
            }, f"missing cells in split of size {len(split)}"

    def test_language_balance_preserved(self):
        records = (
            _make_records(Section.METHODS, "en", 200)
            + _make_records(Section.METHODS, "fr", 200)
        )
        result = stratified_split(records, val_ratio=0.1, test_ratio=0.1, seed=3)
        for split in (result.train, result.val, result.test):
            en = sum(1 for r in split if r.language == "en")
            fr = sum(1 for r in split if r.language == "fr")
            # 50/50 in input → roughly 50/50 in each split (small rounding allowed)
            assert abs(en - fr) <= 2, f"split skewed: en={en} fr={fr}"


class TestDeterminism:
    def test_same_seed_same_output(self):
        records = _make_records(Section.METHODS, "en", 50)
        a = stratified_split(records, seed=99)
        b = stratified_split(records, seed=99)
        assert [r.text for r in a.train] == [r.text for r in b.train]
        assert [r.text for r in a.val] == [r.text for r in b.val]
        assert [r.text for r in a.test] == [r.text for r in b.test]

    def test_different_seed_different_output(self):
        records = _make_records(Section.METHODS, "en", 50)
        a = stratified_split(records, seed=1)
        b = stratified_split(records, seed=2)
        # Different seeds permute groups differently — at least one split's
        # ordering should differ.
        assert (
            [r.text for r in a.train] != [r.text for r in b.train]
            or [r.text for r in a.val] != [r.text for r in b.val]
        )


class TestSmallGroups:
    def test_singleton_group_goes_to_train(self):
        records = _make_records(Section.METHODS, "en", 1)
        result = stratified_split(records, seed=0)
        assert result.train == records
        assert result.val == []
        assert result.test == []

    def test_two_record_group_goes_to_train(self):
        records = _make_records(Section.METHODS, "en", 2)
        result = stratified_split(records, seed=0)
        assert len(result.train) == 2
        assert result.val == []
        assert result.test == []

    def test_three_record_group_splits_one_each(self):
        records = _make_records(Section.METHODS, "en", 3)
        result = stratified_split(records, val_ratio=0.34, test_ratio=0.34, seed=0)
        assert len(result.train) == 1
        assert len(result.val) == 1
        assert len(result.test) == 1


class TestValidation:
    def test_negative_ratio_rejected(self):
        with pytest.raises(ValueError):
            stratified_split([], val_ratio=-0.1, test_ratio=0.1)

    def test_combined_ratio_too_large_rejected(self):
        with pytest.raises(ValueError):
            stratified_split([], val_ratio=0.6, test_ratio=0.5)

    def test_empty_input_returns_empty_splits(self):
        result = stratified_split([], seed=0)
        assert result.train == []
        assert result.val == []
        assert result.test == []
