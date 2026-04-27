"""
Unified section-label schema for the bilingual section classifier.

Each source dataset uses a different label vocabulary. PubMed-RCT calls
the methods section "METHODS" but tags introductions "BACKGROUND" or
"OBJECTIVE". CSAbstruct uses a smaller schema. HAL French papers expose
LaTeX section markers like `Méthodes`, `Résultats`, etc.

This module defines the canonical schema used during training and the
mappings from each source to the canonical labels. The schema is the
most-disputed design decision in the project — centralising it lets us
version it, test mappings, and re-run training when it changes (see
ADR-002 in docs/DECISIONS.md).
"""

from __future__ import annotations

from enum import StrEnum


class Section(StrEnum):
    """Canonical section labels. Order is intentional — used as the
    fixed label index for the trained classifier head."""

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    RELATED_WORK = "related_work"
    CONCLUSION = "conclusion"
    OTHER = "other"


# PubMed-RCT exposes 5 labels at the sentence level. BACKGROUND and
# OBJECTIVE both fold into INTRODUCTION here — the distinction is
# domain-specific to clinical abstracts and doesn't generalise.
PUBMED_RCT_TO_CANONICAL: dict[str, Section] = {
    "BACKGROUND": Section.INTRODUCTION,
    "OBJECTIVE": Section.INTRODUCTION,
    "METHODS": Section.METHODS,
    "RESULTS": Section.RESULTS,
    "CONCLUSIONS": Section.CONCLUSION,
}


# CSAbstruct uses computer-science abstract labels; "other" is preserved.
CSABSTRUCT_TO_CANONICAL: dict[str, Section] = {
    "background": Section.INTRODUCTION,
    "objective": Section.INTRODUCTION,
    "method": Section.METHODS,
    "result": Section.RESULTS,
    "other": Section.OTHER,
}


# HAL French papers expose LaTeX/PDF section headers verbatim. We match
# normalised (lowercase, accent-stripped) headers to canonical labels.
# Both languages live in this map because HAL hosts EN and FR papers.
HAL_HEADER_PATTERNS: dict[str, Section] = {
    # English
    "abstract": Section.ABSTRACT,
    "introduction": Section.INTRODUCTION,
    "method": Section.METHODS,
    "methods": Section.METHODS,
    "methodology": Section.METHODS,
    "results": Section.RESULTS,
    "experimental results": Section.RESULTS,
    "discussion": Section.DISCUSSION,
    "related work": Section.RELATED_WORK,
    "background": Section.RELATED_WORK,
    "conclusion": Section.CONCLUSION,
    "conclusions": Section.CONCLUSION,
    # French (accent-stripped at lookup time — see normalise_header)
    "resume": Section.ABSTRACT,
    "methodes": Section.METHODS,
    "methode": Section.METHODS,
    "methodologie": Section.METHODS,
    "resultats": Section.RESULTS,
    "resultats experimentaux": Section.RESULTS,
    "travaux connexes": Section.RELATED_WORK,
    "etat de l art": Section.RELATED_WORK,
    "etat de lart": Section.RELATED_WORK,
}


def all_canonical_labels() -> list[str]:
    """Ordered list of label string values, used as the fixed index for
    the model head."""
    return [s.value for s in Section]


def normalise_header(raw: str) -> str:
    """Lowercase, strip whitespace, and remove combining diacritics so
    `Résumé`, `RESUME`, and `  resume ` all map to `resume`."""
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", raw.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def hal_header_to_section(raw: str) -> Section:
    """Map a raw section header from a HAL paper to a canonical label.
    Falls back to OTHER for unrecognised headers."""
    return HAL_HEADER_PATTERNS.get(normalise_header(raw), Section.OTHER)
