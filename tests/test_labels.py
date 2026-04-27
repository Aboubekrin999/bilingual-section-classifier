"""Tests for the unified section-label schema and source-label mappings."""

import pytest

from src.labels import (
    CSABSTRUCT_TO_CANONICAL,
    HAL_HEADER_PATTERNS,
    PUBMED_RCT_TO_CANONICAL,
    Section,
    all_canonical_labels,
    hal_header_to_section,
    normalise_header,
)


def test_canonical_labels_unique():
    """Label string values are distinct — guards against future enum drift."""
    labels = all_canonical_labels()
    assert len(labels) == len(set(labels))


def test_pubmed_rct_covers_full_label_set():
    """The full PubMed-RCT label vocabulary is mapped."""
    expected = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}
    assert set(PUBMED_RCT_TO_CANONICAL.keys()) == expected


def test_pubmed_rct_targets_are_canonical():
    """All PubMed-RCT mappings land in the canonical Section enum."""
    for src, target in PUBMED_RCT_TO_CANONICAL.items():
        assert isinstance(target, Section), f"{src} → {target!r} is not canonical"


def test_csabstruct_targets_are_canonical():
    """All CSAbstruct mappings land in the canonical Section enum."""
    for src, target in CSABSTRUCT_TO_CANONICAL.items():
        assert isinstance(target, Section), f"{src} → {target!r} is not canonical"


def test_hal_patterns_target_canonical():
    """All HAL header patterns map to canonical sections."""
    for header, target in HAL_HEADER_PATTERNS.items():
        assert isinstance(target, Section), f"{header} → {target!r} is not canonical"


@pytest.mark.parametrize(
    "raw,expected_normalised",
    [
        ("Résumé", "resume"),
        ("RESUME", "resume"),
        ("  resume ", "resume"),
        ("Méthodes", "methodes"),
        ("État de l'art", "etat de l'art"),
    ],
)
def test_normalise_header_strips_case_whitespace_and_diacritics(
    raw: str, expected_normalised: str
):
    assert normalise_header(raw) == expected_normalised


@pytest.mark.parametrize(
    "raw,expected_section",
    [
        ("Résumé", Section.ABSTRACT),
        ("Méthodes", Section.METHODS),
        ("Résultats", Section.RESULTS),
        ("Travaux connexes", Section.RELATED_WORK),
        ("Conclusion", Section.CONCLUSION),
        ("Methods", Section.METHODS),
        ("Related Work", Section.RELATED_WORK),
        ("Acknowledgements", Section.OTHER),  # unmapped → OTHER
    ],
)
def test_hal_header_to_section_bilingual(raw: str, expected_section: Section):
    assert hal_header_to_section(raw) is expected_section
