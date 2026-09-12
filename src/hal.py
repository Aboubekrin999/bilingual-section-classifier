"""
HAL scrape pipeline — French scientific paragraphs labelled by section.

This is the source that makes the classifier bilingual. PubMed-RCT and
CSAbstruct supply English only; every French row, and every example of
``abstract`` / ``discussion`` / ``related_work``, comes from here.

The pipeline is four pure functions plus an injectable fetcher, mirroring
``src.download``, so everything except the network is testable offline:

    search()          HAL API query  -> document metadata
    extract_text()    PDF bytes      -> raw text
    segment()         raw text       -> [(header, body)]
    to_records()      segments       -> JSONL-ready dicts

**HAL is multi-disciplinary**, so a large share of its French corpus is
humanities work with descriptive rather than IMRaD headers. Rather than
guess a discipline filter, ``segment`` keeps only what it can recognise
and ``is_usable`` rejects papers that yielded too few canonical sections.
That gate does the discipline selection implicitly, and honestly.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from src.labels import Section, hal_header_to_section, normalise_header

HAL_API = "https://api.archives-ouvertes.fr/search/"

#: Identify the scraper. HAL asks that bulk users be reachable.
USER_AGENT = (
    "bilingual-section-classifier/0.1 "
    "(research; https://github.com/Aboubekrin999/bilingual-section-classifier)"
)

#: Papers older than this are frequently image-only scans that extract as
#: noise. 2010 keeps ~100k French open-access articles, which is ample.
DEFAULT_MIN_YEAR = 2010

#: HAL's French open-access corpus is ~80% humanities and social sciences
#: (112k of 140k articles), and that literature does not use IMRaD
#: headings — it yields Introduction and Conclusion and almost no Methods
#: or Results. Restricting to the disciplines that do structure papers as
#: IMRaD still leaves ~28k articles, which is far more than this corpus
#: needs, and it is the difference between a classifier that has seen
#: "Résultats" and one that has not.
IMRAD_DOMAINS = ("sdv", "sde", "info", "spi", "chim", "phys")

#: A header line is short. Body prose that survives the other filters is
#: usually longer than this.
MAX_HEADER_CHARS = 70

#: Leading section numbering: "3.", "3.1", "IV." and friends.
_NUMBERING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+", re.IGNORECASE)

#: A line repeated this often is a running head or a journal footer.
_RUNNING_HEAD_THRESHOLD = 3

#: Everything above this line count in the HAL cover sheet is boilerplate.
_COVER_MARKERS = (
    "hal is a multi-disciplinary open access archive",
    "l'archive ouverte pluridisciplinaire",
    "to cite this version",
)

#: Rough sentence boundary. Deliberately simple: the downstream model sees
#: short passages, and an occasional bad split is noise, not a bug.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂÎÔÛÇ«\"])")

#: Function words that separate French from English scientific prose.
_FR_MARKERS = frozenset(
    "le la les des une dun dans est sont cette nous que qui pour par sur "
    "avec plus ainsi entre leur ses aux ont été être cet elle".split()
)
_EN_MARKERS = frozenset(
    "the of and to in is are this that we for with as by be have has "
    "these those their from was were which".split()
)

Fetcher = Callable[[str], bytes]


class HALError(RuntimeError):
    """A HAL request failed, or returned something unusable."""


@dataclass(frozen=True)
class Segment:
    """One canonical section of one paper, with its body text."""

    header: str
    section: Section
    body: str


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def http_fetch(url: str) -> bytes:
    """Single GET with the project User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as exc:
        raise HALError(f"{url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HALError(f"could not reach {url}: {exc.reason}") from exc


def build_search_url(
    *,
    rows: int = 100,
    start: int = 0,
    min_year: int = DEFAULT_MIN_YEAR,
    domains: tuple[str, ...] = IMRAD_DOMAINS,
) -> str:
    """URL for one page of French, open-access, full-text articles."""
    if not 1 <= rows <= 1000:
        raise ValueError(f"rows must be between 1 and 1000, got {rows}")
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")

    params = [
        ("q", "*:*"),
        ("fq", "language_s:fr"),
        ("fq", "openAccess_bool:true"),
        ("fq", "docType_s:ART"),
        ("fq", f"producedDateY_i:[{min_year} TO *]"),
    ]
    if domains:
        params.append(("fq", f"level0_domain_s:({' OR '.join(domains)})"))
    params += [
        ("fl", "docid,fileMain_s,producedDateY_i"),
        ("rows", str(rows)),
        ("start", str(start)),
        ("sort", "docid asc"),
        ("wt", "json"),
    ]
    return f"{HAL_API}?{urllib.parse.urlencode(params)}"


def search(
    *,
    rows: int = 100,
    start: int = 0,
    min_year: int = DEFAULT_MIN_YEAR,
    domains: tuple[str, ...] = IMRAD_DOMAINS,
    fetch: Fetcher = http_fetch,
) -> list[dict]:
    """One page of results, with documents missing a PDF dropped."""
    body = fetch(
        build_search_url(
            rows=rows, start=start, min_year=min_year, domains=domains
        )
    )
    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise HALError("HAL search returned non-JSON") from exc

    try:
        docs = payload["response"]["docs"]
    except (KeyError, TypeError) as exc:
        raise HALError(f"unexpected HAL response shape: {body[:200]!r}") from exc

    return [d for d in docs if d.get("fileMain_s")]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def sanitise(text: str) -> str:
    """Drop characters that cannot survive a round-trip to UTF-8.

    Malformed embedded fonts make pypdf emit lone surrogates in the
    U+DC80-U+DCFF range. They compare fine as ``str`` and then raise
    ``UnicodeEncodeError`` at the point of writing JSONL, which is a long
    way from the cause. Removing them at extraction keeps the failure
    where it belongs.
    """
    return text.encode("utf-8", "ignore").decode("utf-8")


def extract_text(pdf_bytes: bytes) -> str:
    """Concatenate every page's text. Empty string if nothing extracts."""
    from pypdf import PdfReader  # noqa: PLC0415 - keeps import cost off the API path

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # pypdf raises a wide variety on malformed files
        raise HALError(f"could not read PDF: {exc}") from exc
    return sanitise(pages)


def looks_like_text(text: str, *, min_chars: int = 2000, min_alpha: float = 0.6) -> bool:
    """Reject scans and extraction failures before they reach segmentation.

    Image-only PDFs extract to near-nothing; bad OCR extracts to a soup of
    punctuation and ligature artefacts with a low alphabetic ratio.
    """
    if len(text) < min_chars:
        return False
    alpha = sum(c.isalpha() for c in text)
    return alpha / len(text) >= min_alpha


def strip_cover_page(text: str) -> str:
    """Drop HAL's deposit cover sheet.

    Every HAL PDF is prefixed with a generated page describing the archive
    in English and French. Left in, it contributes English boilerplate to
    what is supposed to be the French half of the corpus.
    """
    lines = text.split("\n")
    last_marker = -1
    for i, line in enumerate(lines[:80]):
        lowered = line.strip().lower()
        if any(marker in lowered for marker in _COVER_MARKERS):
            last_marker = i
    return "\n".join(lines[last_marker + 1 :]) if last_marker >= 0 else text


def drop_running_heads(lines: list[str]) -> list[str]:
    """Remove journal footers and page headers repeated across pages."""
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1
    return [
        line
        for line in lines
        if counts.get(line.strip(), 0) < _RUNNING_HEAD_THRESHOLD
    ]


def header_section(line: str) -> Section | None:
    """Canonical section for ``line``, or ``None`` if it isn't a header.

    ``normalise_header`` handles case and accents but not the numbering
    that real papers put in front of a heading, so that is stripped first.
    Headers mapping to ``OTHER`` are treated as non-headers: a descriptive
    humanities heading carries no section signal, and admitting it would
    let arbitrary prose through as an ``OTHER`` example.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADER_CHARS:
        return None

    without_number = _NUMBERING.sub("", stripped)
    # Trailing punctuation is common in "Méthodes :" style headings.
    candidate = without_number.strip().rstrip(":.").strip()
    if not candidate:
        return None

    section = hal_header_to_section(candidate)
    return section if section is not Section.OTHER else None


def looks_like_header(line: str) -> bool:
    """Whether ``line`` is structurally a heading, canonical or not.

    Needed as a *terminator*. HAL's corpus is full of descriptive headings
    ("3. Autour de l'e-cigarette") that carry no section signal. Without
    treating them as boundaries, everything under them is appended to
    whichever canonical section came last — which silently relabels whole
    discussion sections as Introduction.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADER_CHARS:
        return False

    without_number = _NUMBERING.sub("", stripped).strip()
    if not without_number:
        return False
    if len(without_number.split()) > 10:
        return False
    if not (without_number[0].isupper() or without_number[0].isdigit()):
        return False
    # The decisive signal: prose terminates with sentence punctuation and a
    # heading does not. ":" is allowed — "Méthodes :" is a common style.
    return without_number[-1] not in ".,;!?"


def segment(text: str) -> list[Segment]:
    """Split a paper into canonical sections.

    Everything before the first recognised header is discarded — that span
    is the cover sheet, title block, and author list, none of which is a
    labelled section.
    """
    lines = drop_running_heads(strip_cover_page(text).split("\n"))

    segments: list[Segment] = []
    header: str | None = None
    section: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        if header is not None and section is not None:
            body = " ".join(" ".join(buffer).split())
            if body:
                segments.append(Segment(header=header, section=section, body=body))

    for line in lines:
        found = header_section(line)
        if found is not None:
            flush()
            header = _NUMBERING.sub("", line.strip()).strip().rstrip(":.").strip()
            section = found
            buffer = []
        elif looks_like_header(line):
            # A heading we cannot label ends the section it follows.
            flush()
            header, section, buffer = None, None, []
        elif section is not None:
            buffer.append(line)

    flush()
    return segments


def is_usable(segments: list[Segment], *, min_sections: int = 2) -> bool:
    """Whether a paper contributed enough structure to be worth keeping.

    A paper with one recognised header is usually a false positive — a
    stray "Introduction" in a reference list, say. Requiring two distinct
    canonical sections is what selects IMRaD-shaped papers out of HAL's
    multi-disciplinary corpus without hard-coding a discipline filter.
    """
    return len({s.section for s in segments}) >= min_sections


# ---------------------------------------------------------------------------
# Language + record emission
# ---------------------------------------------------------------------------


def detect_language(text: str) -> str | None:
    """``"fr"``, ``"en"``, or ``None`` when neither clearly dominates.

    HAL's ``language_s`` is deposit metadata and is wrong often enough to
    matter — French-tagged records regularly carry English bodies. A
    function-word ratio is crude but decisive on scientific prose, and
    keeps this module dependency-free.
    """
    words = re.findall(r"[a-zà-öø-ÿ']+", text.lower())
    if len(words) < 20:
        return None

    french = sum(w in _FR_MARKERS for w in words)
    english = sum(w in _EN_MARKERS for w in words)
    if french == english:
        return None
    return "fr" if french > english else "en"


def split_sentences(body: str, *, min_chars: int = 40, max_chars: int = 600) -> list[str]:
    """Section body to sentence-scale passages.

    The English sources label individual abstract sentences, so HAL
    paragraphs are split to roughly match that granularity rather than
    handing the model whole sections it would never see at inference.
    """
    pieces = _SENTENCE_END.split(body)
    return [
        piece.strip()
        for piece in pieces
        if min_chars <= len(piece.strip()) <= max_chars
    ]


def to_records(
    segments: Iterable[Segment],
    *,
    docid: str,
    language: str | None = None,
) -> Iterator[dict]:
    """JSONL records matching the contract ``src.datasets.load_hal`` expects.

    Language is decided once for the whole paper, not per sentence. A
    single scientific sentence rarely carries twenty function words, so
    per-sentence detection abstains constantly and would silently discard
    most of the corpus. Papers whose language cannot be determined at
    document scale yield nothing, which is the honest outcome — the
    ``language`` column is the axis the evaluation is stratified on, so a
    guess there is worse than a gap.
    """
    segments = list(segments)
    if language is None:
        language = detect_language(" ".join(seg.body for seg in segments))
    if language is None:
        return

    for seg in segments:
        for sentence in split_sentences(seg.body):
            yield {
                "text": sentence,
                "header": seg.header,
                "language": language,
                "docid": docid,
            }
