"""Tests for source-dataset loaders."""

import pytest

from src.datasets import Record, load_csabstruct, load_hal, load_pubmed_rct
from src.labels import Section


class TestPubMedRCT:
    def test_basic_record(self):
        lines = ["BACKGROUND\tThe aim of this study was to investigate."]
        records = list(load_pubmed_rct(lines))
        assert records == [
            Record(
                text="The aim of this study was to investigate.",
                language="en",
                label=Section.INTRODUCTION,
                source="pubmed_rct",
            )
        ]

    def test_doc_separator_is_skipped(self):
        lines = [
            "###24832012",
            "BACKGROUND\tFirst sentence.",
            "###24862194",
            "METHODS\tSecond sentence.",
        ]
        records = list(load_pubmed_rct(lines))
        assert len(records) == 2
        assert records[0].label is Section.INTRODUCTION
        assert records[1].label is Section.METHODS

    def test_blank_lines_skipped(self):
        lines = ["", "BACKGROUND\tHello.", "", "RESULTS\tWorld."]
        assert len(list(load_pubmed_rct(lines))) == 2

    def test_objective_folds_into_introduction(self):
        # PubMed-RCT separates BACKGROUND and OBJECTIVE; both are
        # 'introduction' in the canonical schema.
        lines = ["OBJECTIVE\tTo measure outcomes."]
        records = list(load_pubmed_rct(lines))
        assert records[0].label is Section.INTRODUCTION

    def test_unknown_label_dropped(self):
        lines = ["MYSTERY\tShould be dropped."]
        assert list(load_pubmed_rct(lines)) == []

    def test_lines_without_tab_dropped(self):
        lines = ["no tab in this line"]
        assert list(load_pubmed_rct(lines)) == []

    def test_empty_text_dropped(self):
        lines = ["BACKGROUND\t   "]
        assert list(load_pubmed_rct(lines)) == []

    def test_trailing_whitespace_stripped(self):
        lines = ["BACKGROUND\tHello world.   \n"]
        records = list(load_pubmed_rct(lines))
        assert records[0].text == "Hello world."

    def test_language_always_en(self):
        lines = ["BACKGROUND\tWhatever."]
        assert list(load_pubmed_rct(lines))[0].language == "en"


class TestCSAbstruct:
    def test_basic_jsonl_doc(self):
        lines = [
            '{"sentences": ["First sentence.", "Second sentence."], '
            '"labels": ["background", "method"]}'
        ]
        records = list(load_csabstruct(lines))
        assert len(records) == 2
        assert records[0].label is Section.INTRODUCTION
        assert records[1].label is Section.METHODS

    def test_empty_lines_skipped(self):
        lines = [
            "",
            '{"sentences": ["x"], "labels": ["result"]}',
            "",
        ]
        records = list(load_csabstruct(lines))
        assert len(records) == 1
        assert records[0].label is Section.RESULTS

    def test_label_case_insensitive(self):
        lines = ['{"sentences": ["x"], "labels": ["RESULT"]}']
        records = list(load_csabstruct(lines))
        assert records[0].label is Section.RESULTS

    def test_unknown_label_dropped(self):
        lines = ['{"sentences": ["a", "b"], "labels": ["mystery", "result"]}']
        records = list(load_csabstruct(lines))
        assert len(records) == 1
        assert records[0].text == "b"

    def test_empty_sentence_dropped(self):
        lines = ['{"sentences": ["", "real"], "labels": ["result", "result"]}']
        records = list(load_csabstruct(lines))
        assert len(records) == 1

    def test_mismatched_lengths_raises(self):
        lines = ['{"sentences": ["a", "b"], "labels": ["result"]}']
        with pytest.raises(ValueError, match="mismatched lengths"):
            list(load_csabstruct(lines))

    def test_other_label_preserved(self):
        # CSAbstruct has an 'other' bucket that maps to Section.OTHER.
        lines = ['{"sentences": ["misc"], "labels": ["other"]}']
        records = list(load_csabstruct(lines))
        assert records[0].label is Section.OTHER


class TestHAL:
    def test_french_record(self):
        lines = [
            '{"text": "Les expériences montrent...", '
            '"header": "Méthodes", '
            '"language": "fr"}'
        ]
        records = list(load_hal(lines))
        assert records == [
            Record(
                text="Les expériences montrent...",
                language="fr",
                label=Section.METHODS,
                source="hal",
            )
        ]

    def test_english_record(self):
        lines = ['{"text": "The results show.", "header": "Results", "language": "en"}']
        records = list(load_hal(lines))
        assert records[0].language == "en"
        assert records[0].label is Section.RESULTS

    def test_unknown_header_falls_to_other(self):
        # Unlike PubMed-RCT/CSAbstruct, unknown HAL headers keep the record
        # with Section.OTHER — header heuristics have a real noise floor.
        lines = [
            '{"text": "Acknowledgements text.", '
            '"header": "Acknowledgements", '
            '"language": "en"}'
        ]
        records = list(load_hal(lines))
        assert len(records) == 1
        assert records[0].label is Section.OTHER

    def test_unsupported_language_raises(self):
        lines = ['{"text": "x", "header": "Methods", "language": "de"}']
        with pytest.raises(ValueError, match="unsupported language"):
            list(load_hal(lines))

    def test_empty_text_skipped(self):
        lines = ['{"text": "   ", "header": "Methods", "language": "en"}']
        assert list(load_hal(lines)) == []

    def test_blank_lines_skipped(self):
        lines = [
            "",
            '{"text": "x", "header": "Methods", "language": "en"}',
            "",
        ]
        assert len(list(load_hal(lines))) == 1


class TestRecord:
    def test_record_is_frozen(self):
        r = Record(text="x", language="en", label=Section.METHODS, source="hal")
        with pytest.raises(AttributeError):
            r.text = "mutated"  # type: ignore[misc]
