# Bilingual Paper Section Classifier (EN + FR)

[![CI](https://github.com/Aboubekrin999/bilingual-section-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Aboubekrin999/bilingual-section-classifier/actions/workflows/ci.yml)

> Fine-tune a small multilingual encoder to classify scientific text passages by section type (Abstract / Introduction / Methods / Results / Discussion / Related Work / Conclusion). Targets both English and French academic prose.

**Status:** Data pipeline and training script complete · 94 tests green · the training run itself has not been executed. Paused May 2026; see [What's built today](#whats-built-today).

---

## The problem

Scientific papers are structured, but most ingestion pipelines treat them as flat text. A passage from "Methods" should be retrieved differently than a passage from "Related Work" depending on the question:

- *"What did they actually do?"* → Methods chunks
- *"What is the prior art?"* → Related Work chunks
- *"What did they find?"* → Results / Discussion chunks

Most off-the-shelf classifiers are English-only. **French scientific writing — HAL, INRIA, French university theses — is underserved.** A bilingual classifier closes that gap.

## Why this matters for [paper-companion](https://github.com/Aboubekrin999/paper-companion)

This classifier is the smarter chunker for the [paper-companion](https://github.com/Aboubekrin999/paper-companion) RAG system. With predicted section labels on each chunk, retrieval can filter by section type and dramatically improve answer relevance for "method-flavored" vs "result-flavored" questions.

The two repos ship independently but compose.

## Approach

| Step | Choice | Rationale |
|---|---|---|
| Base model | `xlm-roberta-base` (270M params) | Strong multilingual encoder, well-supported, fits a single consumer GPU |
| Backup model | `Qwen/Qwen2.5-0.5B` | Smaller; tested if XLM-R underfits |
| Dataset (EN) | [PubMed-RCT](https://github.com/Franck-Dernoncourt/pubmed-rct) + [CSAbstruct](https://github.com/allenai/sequential_sentence_classification) | Section-tagged, well-known baselines |
| Dataset (FR) | [HAL](https://hal.science/) open-access papers, segmented by LaTeX/PDF section markers (~5–10k labeled sentences) | Custom — fills the gap |
| Eval split | Stratified 80/10/10, **language-stratified test set** | Catches asymmetric performance |
| Metrics | Macro-F1 overall · F1 per class · F1 per language · confusion matrix | A single headline number hides per-class and per-language failure |
| Tracking | [Weights & Biases](https://wandb.ai/) | Reproducible experiments |
| Demo | Gradio on Hugging Face Spaces | Lets anyone test the model without cloning the repo |

Detailed reasoning in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## What's built today

Honest state of the repo, so you can tell the code from the plan.

| Area | State |
|---|---|
| **Label schema** — unified section taxonomy with EN + FR source mappings | Built, tested |
| **Source loaders** — PubMed-RCT, CSAbstruct, HAL adapters | Built, tested |
| **Splitter** — stratified 80/10/10 with a language-stratified test set | Built, tested |
| **Dataset build** — download → normalize → Parquet, documented in [`docs/DATA.md`](docs/DATA.md) | Built, tested |
| **Metrics** — macro-F1, per-class F1, per-language F1, confusion matrix | Built, tested |
| **Training script** — XLM-R fine-tune with per-language eval | Written, not yet run |
| **Trained model on the Hub** | Not built |
| **Gradio demo on Spaces** | Not built |
| **Results write-up** | Not built — there are no results to report yet |

94 tests pass (`pytest`). CI runs ruff and the test suite on every PR, deliberately without the heavy ML dependencies so it stays fast; training runs on Colab.

Work paused in May 2026 while client delivery took priority.

## What v1 will ship

- Trained model on Hugging Face Hub: `Aboubekrin999/bilingual-section-classifier`
- Live Gradio demo on Hugging Face Spaces
- Write-up: methodology, results, error analysis, language-stratified breakdown, limitations
- Reproducibility: data prep notebooks + training script + eval harness

## Tech stack

- Python 3.11 · PyTorch · Hugging Face `transformers` + `datasets`
- `wandb` for experiment tracking
- `gradio` for the demo
- Trained on Colab Pro / single consumer GPU

## Roadmap

3-week plan in [`docs/ROADMAP.md`](docs/ROADMAP.md): dataset build, then training, then reproducible eval and the demo.

## Local development

Running the tests needs nothing heavier than pytest, ruff and pyarrow — the
same three packages CI installs. Torch and transformers are only needed to
train.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "pytest>=8.3" "ruff>=0.6" "pyarrow>=18"

pytest                  # 94 passed — no network, no GPU, no ML dependencies
ruff check src tests scripts
```

The suite runs on fixtures, so it works on a fresh clone in under a second.

### Building the dataset

The full stack — torch, transformers, datasets, wandb, gradio — installs with
the package itself:

```bash
pip install -e .
python scripts/download_data.py     # PubMed-RCT, CSAbstruct, HAL
python scripts/build_dataset.py     # normalize → label → stratify → Parquet
```

Source-by-source provenance, licensing, and the French segmentation approach are in [`docs/DATA.md`](docs/DATA.md).

### Training

```bash
python scripts/train.py             # XLM-R fine-tune, logs to W&B
python scripts/evaluate.py          # macro-F1 + per-class + per-language breakdown
```

Written for a single consumer GPU or Colab. Not yet executed — no checkpoint or results are published.

## Author

**Aboubekrin Mohamed Salem** — software engineer, Paris, working in English and French.

Built to make retrieval in [paper-companion](https://github.com/Aboubekrin999/paper-companion) section-aware, and to learn transformer fine-tuning properly: constructing a dataset rather than downloading one, designing a label schema that holds across two languages, and evaluating in a way that exposes per-language weakness instead of averaging it away. Reasoning in [`docs/DECISIONS.md`](docs/DECISIONS.md).

GitHub: [@Aboubekrin999](https://github.com/Aboubekrin999)
