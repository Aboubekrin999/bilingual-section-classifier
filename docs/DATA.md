# Datasets

The classifier learns from three sources, each with its own format and label vocabulary, harmonised through the canonical schema in [`src/labels.py`](../src/labels.py).

## Sources

### 1. PubMed-RCT (English, biomedical abstracts)

- **Variant:** `PubMed_20k_RCT` — 20,000 randomized controlled trial abstracts, sentence-level labels.
- **Source:** [Franck-Dernoncourt/pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct), pinned to a specific commit sha in [`src/download.py`](../src/download.py).
- **Format:** plain text. Lines are `LABEL\tsentence`; abstracts are separated by blank lines and `###PMID` markers.
- **Labels:** `BACKGROUND`, `OBJECTIVE`, `METHODS`, `RESULTS`, `CONCLUSIONS`. The first two collapse to `Section.INTRODUCTION` in our schema.
- **License:** [original repo terms](https://github.com/Franck-Dernoncourt/pubmed-rct) — academic / research use.

### 2. CSAbstruct (English, computer-science abstracts)

- **Source:** [allenai/sequential_sentence_classification](https://github.com/allenai/sequential_sentence_classification), pinned commit in [`src/download.py`](../src/download.py).
- **Format:** JSONL. One abstract per line with parallel `sentences` and `labels` arrays.
- **Labels:** `background`, `objective`, `method`, `result`, `other`. `background` and `objective` collapse to `INTRODUCTION`; `other` is preserved.
- **License:** Apache-2.0 per the source repo.

### 3. HAL French paragraphs (French, mixed-domain)

- **Status:** scraper not yet authored — placeholder format documented for forward compatibility.
- **Format:** JSONL with `text`, `header`, `language` per record. The HAL header (`Méthodes`, `Résultats`, …) is mapped via `hal_header_to_section` in [`src/labels.py`](../src/labels.py); unknown headers fall through to `Section.OTHER`.
- **Why HAL:** open-access French scientific papers, broad domain coverage (engineering, social science, humanities), gives the classifier real bilingual signal beyond translated CS abstracts.

## Build pipeline

```
data/
  pubmed_rct/      ← downloaded splits
  csabstruct/      ← downloaded splits
  hal/             ← scraper output (when available)
  built/
    train.parquet
    val.parquet
    test.parquet
```

Two CLI scripts reproduce the corpus from scratch:

```bash
python -m scripts.download_data        # → data/pubmed_rct/, data/csabstruct/
python -m scripts.build_dataset        # → data/built/{train,val,test}.parquet
```

The build is fully deterministic given a seed (see `stratified_split`'s `seed` argument). Splits are stratified by `(label, language)` so every split preserves both class and language balance — critical because the headline metric is **macro-F1 averaged over languages**, not pooled accuracy.

## Canonical label schema

Defined as `Section` in [`src/labels.py`](../src/labels.py):

| Canonical | Sources that map here |
| --- | --- |
| `abstract` | HAL `Résumé` / `Abstract` |
| `introduction` | PubMed-RCT `BACKGROUND`/`OBJECTIVE`, CSAbstruct `background`/`objective`, HAL `Introduction` |
| `methods` | PubMed-RCT `METHODS`, CSAbstruct `method`, HAL `Méthodes`/`Methods`/`Methodology` |
| `results` | PubMed-RCT `RESULTS`, CSAbstruct `result`, HAL `Résultats`/`Results` |
| `discussion` | HAL `Discussion` |
| `related_work` | HAL `Travaux connexes` / `Related Work` / `Background` |
| `conclusion` | PubMed-RCT `CONCLUSIONS`, HAL `Conclusion` |
| `other` | CSAbstruct `other`, unrecognised HAL headers |

The schema is the most-disputed design decision in the project; centralising it lets us version it, test mappings, and re-run training when it changes.

## Known noise and limitations

- **Genre skew.** PubMed-RCT and CSAbstruct are *abstracts*, not full paper sections. Sentence boundaries and section vocabulary inside an abstract differ from inside a full paper — the model may underperform on full-paper inference until HAL adds full-paper signal.
- **Language imbalance.** Until HAL lands, the corpus is overwhelmingly English. The stratified splitter preserves whatever balance the source data has; it doesn't *create* balance. Read per-language F1, not pooled accuracy.
- **Header heuristic on HAL.** Section attribution is by header match; subsections (`5.1 Implementation`) inherit from the nearest preceding canonical header. This is approximate by design — the goal is "section the model would have predicted from text alone," not ground truth from authors.

## What's intentionally not done here

- No data augmentation. The model is fine-tuned on the raw distribution and evaluated on it; if we shift the distribution at training time we have to ship with that augmentation in production too.
- No deduplication across sources. The three sources have functionally zero overlap; running near-duplicate detection burns time without changing metrics.
