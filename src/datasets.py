"""
Source-dataset loaders that emit a unified ``Record`` stream.

Three sources feed the bilingual section classifier — PubMed-RCT (English
sentence-level labels), CSAbstruct (English abstract-level labels), and a
custom JSONL of HAL French paper paragraphs labelled by section header.
Each source has its own format and label vocabulary; this module is the
boundary that translates them into the canonical schema in ``labels.py``.

Loaders are streaming: they accept text/iterables, not file paths, so the
heavy I/O (``open``, ``gzip.open``, S3, etc.) lives in the caller and the
parsing logic stays trivially testable. Records with source labels that
don't map to a canonical section are skipped silently except for HAL,
where unrecognised headers fall through to ``Section.OTHER`` — that's the
realistic noise floor for a header-based heuristic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from src.labels import (
    CSABSTRUCT_TO_CANONICAL,
    PUBMED_RCT_TO_CANONICAL,
    Section,
    hal_header_to_section,
)

Language = Literal["en", "fr"]


@dataclass(frozen=True)
class Record:
    """A unified training example emitted by every source loader."""

    text: str
    language: Language
    label: Section
    source: Literal["pubmed_rct", "csabstruct", "hal"]


def load_pubmed_rct(lines: Iterable[str]) -> Iterator[Record]:
    """
    Parse PubMed-RCT's `LABEL<TAB>sentence` line format.

    The dataset uses ``###<PMID>`` lines to separate documents and blank
    lines between blocks; both are skipped. Lines whose label is not in
    the known PubMed-RCT vocabulary are dropped — a safety net for
    encoding glitches or future label additions.
    """
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line or line.startswith("###"):
            continue
        if "\t" not in line:
            continue
        label_str, _, text = line.partition("\t")
        text = text.strip()
        if not text:
            continue
        canonical = PUBMED_RCT_TO_CANONICAL.get(label_str.strip())
        if canonical is None:
            continue
        yield Record(
            text=text,
            language="en",
            label=canonical,
            source="pubmed_rct",
        )


def load_csabstruct(lines: Iterable[str]) -> Iterator[Record]:
    """
    Parse CSAbstruct's JSONL: one abstract per line, with parallel
    ``sentences`` and ``labels`` arrays.

    Each sentence becomes its own Record. Mismatched array lengths in a
    document raise — that's a corrupt source file, not noise to silently
    swallow.
    """
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        doc = json.loads(line)
        sentences = doc["sentences"]
        labels = doc["labels"]
        if len(sentences) != len(labels):
            raise ValueError(
                f"CSAbstruct document has mismatched lengths: "
                f"{len(sentences)} sentences vs {len(labels)} labels"
            )
        for sentence, source_label in zip(sentences, labels, strict=True):
            text = sentence.strip()
            if not text:
                continue
            canonical = CSABSTRUCT_TO_CANONICAL.get(source_label.strip().lower())
            if canonical is None:
                continue
            yield Record(
                text=text,
                language="en",
                label=canonical,
                source="csabstruct",
            )


def load_hal(lines: Iterable[str]) -> Iterator[Record]:
    """
    Parse the HAL JSONL produced by the scrape pipeline.

    Each line is a JSON object with ``text``, ``header``, and
    ``language`` fields. ``language`` must be ``"en"`` or ``"fr"`` —
    detected upstream by langdetect or fasttext at scrape time, not
    here. Headers are mapped via ``hal_header_to_section`` so unknown
    headers fall through to ``Section.OTHER`` rather than being dropped.
    """
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        doc = json.loads(line)
        text = doc["text"].strip()
        if not text:
            continue
        language = doc["language"]
        if language not in ("en", "fr"):
            raise ValueError(
                f"HAL record has unsupported language {language!r}; "
                "expected 'en' or 'fr'"
            )
        canonical = hal_header_to_section(doc["header"])
        yield Record(
            text=text,
            language=language,
            label=canonical,
            source="hal",
        )
