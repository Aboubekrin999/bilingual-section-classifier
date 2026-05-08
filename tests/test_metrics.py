"""Tests for pure classification metrics."""

import pytest

from src.metrics import (
    classification_report,
    confusion_matrix,
    per_language_report,
    precision_recall_f1,
)


class TestPrecisionRecallF1:
    def test_perfect_classification(self):
        p, r, f, s = precision_recall_f1(["a", "a", "b"], ["a", "a", "b"], "a")
        assert (p, r, f, s) == (1.0, 1.0, 1.0, 2)

    def test_no_correct_predictions(self):
        p, r, f, s = precision_recall_f1(["a", "a"], ["b", "b"], "a")
        assert p == 0.0 and r == 0.0 and f == 0.0
        assert s == 2

    def test_high_precision_low_recall(self):
        # 1 true positive, 0 false positive, 1 false negative
        p, r, f, _ = precision_recall_f1(["a", "a"], ["a", "b"], "a")
        assert p == 1.0
        assert r == 0.5
        assert f == pytest.approx(2 * 1 * 0.5 / 1.5)

    def test_high_recall_low_precision(self):
        # 2 true positive, 1 false positive, 0 false negative
        p, r, f, _ = precision_recall_f1(["a", "a", "b"], ["a", "a", "a"], "a")
        assert p == pytest.approx(2 / 3)
        assert r == 1.0

    def test_label_with_no_examples_returns_zero(self):
        p, r, f, s = precision_recall_f1(["a"], ["a"], "missing-label")
        assert (p, r, f, s) == (0.0, 0.0, 0.0, 0)

    def test_misaligned_lengths_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            precision_recall_f1(["a"], ["a", "b"], "a")


class TestClassificationReport:
    def test_perfect_report(self):
        report = classification_report(["a", "b", "a"], ["a", "b", "a"], ["a", "b"])
        assert report.macro_f1 == 1.0
        assert report.weighted_f1 == 1.0
        assert report.accuracy == 1.0

    def test_completely_wrong_report(self):
        report = classification_report(["a", "b"], ["b", "a"], ["a", "b"])
        assert report.macro_f1 == 0.0
        assert report.accuracy == 0.0

    def test_partial_correctness(self):
        # 3/4 correct, both classes have at least one error
        report = classification_report(
            ["a", "a", "b", "b"],
            ["a", "b", "b", "b"],
            ["a", "b"],
        )
        assert report.accuracy == 0.75

    def test_includes_one_record_per_label(self):
        report = classification_report(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"])
        assert {c.label for c in report.per_class} == {"a", "b", "c"}

    def test_supports_count_correctly(self):
        report = classification_report(
            ["a", "a", "b", "c"], ["a", "a", "b", "c"], ["a", "b", "c"]
        )
        supports = {c.label: c.support for c in report.per_class}
        assert supports == {"a": 2, "b": 1, "c": 1}

    def test_empty_labels_rejected(self):
        with pytest.raises(ValueError, match="at least one label"):
            classification_report(["a"], ["a"], [])


class TestPerLanguageReport:
    def test_separates_languages(self):
        # All English correct, all French wrong
        reports = per_language_report(
            y_true=["a", "b", "a", "b"],
            y_pred=["a", "b", "b", "a"],
            languages=["en", "en", "fr", "fr"],
            labels=["a", "b"],
        )
        assert reports["en"].accuracy == 1.0
        assert reports["fr"].accuracy == 0.0

    def test_per_language_macro_f1_is_independent(self):
        reports = per_language_report(
            y_true=["a", "b"] * 2,
            y_pred=["a", "b"] * 2,
            languages=["en", "en", "fr", "fr"],
            labels=["a", "b"],
        )
        assert reports["en"].macro_f1 == 1.0
        assert reports["fr"].macro_f1 == 1.0

    def test_misaligned_languages_rejected(self):
        with pytest.raises(ValueError, match="must all align"):
            per_language_report(["a"], ["a"], ["en", "fr"], ["a"])


class TestConfusionMatrix:
    def test_perfect_matrix(self):
        matrix = confusion_matrix(["a", "b"], ["a", "b"], ["a", "b"])
        assert matrix == [[1, 0], [0, 1]]

    def test_off_diagonal_records_confusion(self):
        matrix = confusion_matrix(["a", "b"], ["b", "a"], ["a", "b"])
        # All "a" predicted as "b", and vice versa
        assert matrix == [[0, 1], [1, 0]]

    def test_ignores_labels_outside_index(self):
        matrix = confusion_matrix(
            ["a", "unknown"], ["a", "a"], ["a", "b"]
        )
        # 'unknown' isn't in labels, so its row/col is dropped
        assert matrix == [[1, 0], [0, 0]]

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError, match="must align"):
            confusion_matrix(["a"], ["a", "b"], ["a", "b"])
