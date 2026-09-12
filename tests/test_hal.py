"""
Tests for the HAL scrape pipeline.

No network: ``search`` takes an injectable fetcher, and every other stage
is a pure function over strings. Segmentation is the interesting logic —
real HAL PDFs arrive with a generated cover sheet, repeated running heads,
and numbered headings — so most of these exercise that.
"""

from __future__ import annotations

import json

import pytest

from src.hal import (
    DEFAULT_MIN_YEAR,
    looks_like_header,
    sanitise,
    HALError,
    Segment,
    build_search_url,
    detect_language,
    drop_running_heads,
    header_section,
    is_usable,
    looks_like_text,
    search,
    segment,
    split_sentences,
    strip_cover_page,
    to_records,
)
from src.labels import Section

COVER = """HAL Id: hal-03201206
Submitted on 18 Apr 2021
HAL is a multi-disciplinary open access archive for the deposit
L'archive ouverte pluridisciplinaire HAL, est destinee au depot
To cite this version:
Some Author. A Paper Title. Journal, 2021."""

PAPER = """Introduction
Cette etude porte sur la classification des sections scientifiques.
Les travaux anterieurs se concentrent sur l'anglais uniquement.

2. Methodes
Nous avons entraine un modele multilingue sur un corpus bilingue.
Le corpus comprend des articles francais et anglais annotes.

3.1 Resultats
Le modele atteint une performance superieure a la reference.
Les resultats montrent une amelioration nette pour le francais.

Conclusion
Nous concluons que l'approche multilingue est efficace ici."""


class TestBuildSearchUrl:
    def test_filters_to_french_open_access_articles(self) -> None:
        url = build_search_url()
        for expected in ("language_s%3Afr", "openAccess_bool%3Atrue", "docType_s%3AART"):
            assert expected in url

    def test_applies_the_year_floor(self) -> None:
        assert f"{DEFAULT_MIN_YEAR}+TO+%2A" in build_search_url()

    def test_year_floor_is_overridable(self) -> None:
        assert "2018+TO" in build_search_url(min_year=2018)

    def test_sorts_deterministically_so_paging_is_stable(self) -> None:
        assert "sort=docid+asc" in build_search_url()

    @pytest.mark.parametrize("rows", [0, -1, 1001])
    def test_rejects_out_of_range_rows(self, rows: int) -> None:
        with pytest.raises(ValueError, match="rows must be"):
            build_search_url(rows=rows)

    def test_rejects_negative_start(self) -> None:
        with pytest.raises(ValueError, match="start must be"):
            build_search_url(start=-1)


class TestSearch:
    def _payload(self, docs: list[dict]) -> bytes:
        return json.dumps({"response": {"docs": docs}}).encode()

    def test_returns_documents_with_a_pdf(self) -> None:
        docs = [{"docid": "1", "fileMain_s": "https://hal.science/x/document"}]
        assert search(fetch=lambda url: self._payload(docs)) == docs

    def test_drops_documents_without_a_pdf(self) -> None:
        docs = [
            {"docid": "1", "fileMain_s": "https://hal.science/x/document"},
            {"docid": "2"},
            {"docid": "3", "fileMain_s": None},
        ]
        result = search(fetch=lambda url: self._payload(docs))
        assert [d["docid"] for d in result] == ["1"]

    def test_non_json_is_an_error(self) -> None:
        with pytest.raises(HALError, match="non-JSON"):
            search(fetch=lambda url: b"<html>")

    def test_unexpected_shape_is_an_error(self) -> None:
        with pytest.raises(HALError, match="unexpected HAL response"):
            search(fetch=lambda url: b'{"oops": true}')


class TestLooksLikeText:
    def test_accepts_real_prose(self) -> None:
        assert looks_like_text("Le corpus francais. " * 200)

    def test_rejects_a_near_empty_extraction(self) -> None:
        """Image-only scans extract to almost nothing."""
        assert not looks_like_text("abc")

    def test_rejects_punctuation_soup_from_bad_ocr(self) -> None:
        assert not looks_like_text("- . , ; / \\ | " * 400)


class TestSanitise:
    """Malformed embedded fonts make pypdf emit unencodable surrogates."""

    def test_removes_lone_surrogates(self) -> None:
        raw = "Méthodes " + chr(0xDCE9) + " et résultats"
        with pytest.raises(UnicodeEncodeError):
            raw.encode("utf-8")

        cleaned = sanitise(raw)
        cleaned.encode("utf-8")  # no longer raises
        assert "Méthodes" in cleaned and "résultats" in cleaned

    def test_preserves_accented_french(self) -> None:
        assert sanitise("Résumé, méthodes, résultats") == "Résumé, méthodes, résultats"

    def test_empty_string_is_unchanged(self) -> None:
        assert sanitise("") == ""


class TestStripCoverPage:
    def test_removes_hal_boilerplate(self) -> None:
        cleaned = strip_cover_page(f"{COVER}\n{PAPER}")
        assert "multi-disciplinary open access" not in cleaned
        assert "Introduction" in cleaned

    def test_keeps_text_that_has_no_cover_sheet(self) -> None:
        assert strip_cover_page(PAPER) == PAPER

    def test_strips_through_the_last_marker_not_the_first(self) -> None:
        """The cover sheet's French half follows its English half."""
        cleaned = strip_cover_page(f"{COVER}\n{PAPER}")
        assert "To cite this version" not in cleaned


class TestDropRunningHeads:
    def test_removes_a_repeated_journal_footer(self) -> None:
        lines = ["Revue, 26 | 2021", "Du texte.", "Revue, 26 | 2021", "Plus.", "Revue, 26 | 2021"]
        assert drop_running_heads(lines) == ["Du texte.", "Plus."]

    def test_keeps_lines_below_the_threshold(self) -> None:
        lines = ["Introduction", "Du texte.", "Introduction"]
        assert drop_running_heads(lines) == lines

    def test_ignores_blank_lines(self) -> None:
        assert drop_running_heads(["", "", "", "Du texte."]) == ["", "", "", "Du texte."]


class TestHeaderSection:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("Introduction", Section.INTRODUCTION),
            ("Méthodes", Section.METHODS),
            ("MÉTHODES", Section.METHODS),
            ("Résultats", Section.RESULTS),
            ("Discussion", Section.DISCUSSION),
            ("Conclusion", Section.CONCLUSION),
            ("Résumé", Section.ABSTRACT),
        ],
    )
    def test_recognises_french_headers(self, line: str, expected: Section) -> None:
        assert header_section(line) == expected

    @pytest.mark.parametrize(
        "line", ["2. Méthodes", "3.1 Résultats", "4.2.1 Discussion", "IV. Conclusion"]
    )
    def test_strips_section_numbering(self, line: str) -> None:
        """normalise_header handles accents but not numbering."""
        assert header_section(line) is not None

    def test_strips_trailing_punctuation(self) -> None:
        assert header_section("Méthodes :") == Section.METHODS

    def test_descriptive_headings_are_not_headers(self) -> None:
        """A humanities heading carries no section signal."""
        assert header_section("1. Autour de l'e-cigarette : succès et débats") is None

    def test_long_lines_are_body_text(self) -> None:
        assert header_section("Introduction " * 20) is None

    def test_blank_line_is_not_a_header(self) -> None:
        assert header_section("   ") is None

    def test_bare_numbering_is_not_a_header(self) -> None:
        assert header_section("3.") is None


class TestSegment:
    def test_finds_every_canonical_section(self) -> None:
        sections = [s.section for s in segment(PAPER)]
        assert sections == [
            Section.INTRODUCTION,
            Section.METHODS,
            Section.RESULTS,
            Section.CONCLUSION,
        ]

    def test_body_text_follows_its_header(self) -> None:
        methods = next(s for s in segment(PAPER) if s.section is Section.METHODS)
        assert "modele multilingue" in methods.body
        assert "travaux anterieurs" not in methods.body

    def test_header_is_recorded_without_numbering(self) -> None:
        results = next(s for s in segment(PAPER) if s.section is Section.RESULTS)
        assert results.header == "Resultats"

    def test_prologue_before_the_first_header_is_dropped(self) -> None:
        text = f"{COVER}\n{PAPER}"
        assert all("cite this version" not in s.body for s in segment(text))

    def test_empty_sections_are_not_emitted(self) -> None:
        assert segment("Introduction\n\nMéthodes\n") == []

    def test_text_with_no_headers_yields_nothing(self) -> None:
        assert segment("Juste du texte sans aucun titre de section.") == []


class TestLooksLikeHeader:
    """Used as a section *terminator*, so both directions matter."""

    @pytest.mark.parametrize(
        "line",
        [
            "Introduction",
            "1.2. L'e-cigarette et les reseaux sociaux",
            "2. Cadre theorique et remarques methodologiques",
            "Methodes :",
        ],
    )
    def test_recognises_headings(self, line: str) -> None:
        assert looks_like_header(line)

    @pytest.mark.parametrize(
        "line",
        [
            "Cette etude porte sur la classification des sections.",
            "les travaux anterieurs se concentrent sur l'anglais",
            "",
            "3.",
            "Un titre beaucoup trop long pour etre un en-tete de section credible dans un article",
        ],
    )
    def test_rejects_body_text(self, line: str) -> None:
        assert not looks_like_header(line)

    def test_sentence_punctuation_is_the_decisive_signal(self) -> None:
        assert looks_like_header("Resultats et discussion")
        assert not looks_like_header("Resultats et discussion.")


class TestSectionTermination:
    """A heading we cannot label must end the section it follows.

    Without this, prose under a descriptive heading is appended to the
    previous canonical section — which relabelled whole discussion
    sections as Introduction in the first real scrape.
    """

    TEXT = (
        "Introduction\n"
        "Cette premiere phrase appartient bien a l'introduction du papier.\n"
        "3. Autour d'un sujet particulier\n"
        "Cette seconde phrase ne doit surtout pas etre etiquetee introduction.\n"
    )

    def test_content_under_an_unlabelled_heading_is_dropped(self) -> None:
        intro = next(s for s in segment(self.TEXT) if s.section is Section.INTRODUCTION)
        assert "premiere phrase" in intro.body
        assert "seconde phrase" not in intro.body

    def test_only_the_labelled_section_survives(self) -> None:
        assert [s.section for s in segment(self.TEXT)] == [Section.INTRODUCTION]


class TestIsUsable:
    def test_two_distinct_sections_is_enough(self) -> None:
        assert is_usable(segment(PAPER))

    def test_a_single_section_is_rejected(self) -> None:
        """One stray header is usually a false positive."""
        assert not is_usable([Segment("Introduction", Section.INTRODUCTION, "Du texte.")])

    def test_repeats_of_one_section_do_not_count_twice(self) -> None:
        segments = [
            Segment("Introduction", Section.INTRODUCTION, "A."),
            Segment("Introduction", Section.INTRODUCTION, "B."),
        ]
        assert not is_usable(segments)


class TestDetectLanguage:
    def test_identifies_french_prose(self) -> None:
        text = (
            "Nous avons entraine un modele sur des donnees qui sont issues de la "
            "litterature scientifique, et les resultats que nous obtenons dans "
            "cette etude sont plus fiables que ceux de la reference."
        )
        assert detect_language(text) == "fr"

    def test_identifies_english_prose(self) -> None:
        text = (
            "We trained a model on the data that are drawn from the scientific "
            "literature, and the results which we report in this study are more "
            "reliable than those of the baseline."
        )
        assert detect_language(text) == "en"

    def test_too_short_to_judge_returns_none(self) -> None:
        assert detect_language("Trop court.") is None

    def test_ambiguous_text_returns_none(self) -> None:
        """Better to drop a row than to mislabel the language column."""
        assert detect_language("alpha beta gamma delta " * 10) is None


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self) -> None:
        body = (
            "Cette premiere phrase est suffisamment longue pour passer le filtre. "
            "Et cette seconde phrase est egalement assez longue pour passer."
        )
        assert len(split_sentences(body)) == 2

    def test_drops_fragments_that_are_too_short(self) -> None:
        assert split_sentences("Oui. Non. Peut-etre.") == []

    def test_drops_passages_that_are_too_long(self) -> None:
        assert split_sentences("mot " * 400) == []


class TestToRecords:
    def test_language_is_decided_per_document_not_per_sentence(self) -> None:
        """A lone sentence rarely has enough function words to judge."""
        segments = segment(PAPER)
        one_sentence = split_sentences(segments[0].body)[0]
        assert detect_language(one_sentence) is None
        assert list(to_records(segments, docid="hal-1"))

    def test_undetectable_language_yields_nothing(self) -> None:
        segments = [Segment("Introduction", Section.INTRODUCTION, "alpha beta. " * 30)]
        assert list(to_records(segments, docid="hal-1")) == []

    def test_explicit_language_overrides_detection(self) -> None:
        records = list(to_records(segment(PAPER), docid="hal-1", language="en"))
        assert {r["language"] for r in records} == {"en"}

    def test_emits_the_contract_load_hal_expects(self) -> None:
        records = list(to_records(segment(PAPER), docid="hal-1"))
        assert records
        for record in records:
            assert set(record) == {"text", "header", "language", "docid"}
            assert record["language"] in ("fr", "en")
            assert record["docid"] == "hal-1"

    def test_french_body_text_is_labelled_french(self) -> None:
        records = list(to_records(segment(PAPER), docid="hal-1"))
        assert {r["language"] for r in records} == {"fr"}

    def test_records_load_through_the_dataset_loader(self) -> None:
        """The whole point: these rows must parse as training records."""
        from src.datasets import load_hal

        lines = [json.dumps(r) for r in to_records(segment(PAPER), docid="hal-1")]
        loaded = list(load_hal(lines))
        assert loaded
        assert {r.source for r in loaded} == {"hal"}
        assert Section.METHODS in {r.label for r in loaded}
