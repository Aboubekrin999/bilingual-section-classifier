# Bilingual Paper Section Classifier (EN + FR)

> Fine-tune a small multilingual encoder to classify scientific text passages by section type (Abstract / Introduction / Methods / Results / Discussion / Related Work / Conclusion). Targets both English and French academic prose.

**Status:** Planning — training begins May 2026

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
| Metrics | Macro-F1 overall · F1 per class · F1 per language · confusion matrix | Honest reporting beats single-number bragging |
| Tracking | [Weights & Biases](https://wandb.ai/) | Reproducible experiments |
| Demo | Gradio on Hugging Face Spaces | Recruiter-clickable |

Detailed reasoning in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## What v1 ships

- Trained model on Hugging Face Hub: `Aboubekrin999/bilingual-section-classifier`
- Live Gradio demo on Hugging Face Spaces
- 5-page write-up: methodology, results, error analysis, language-stratified breakdown, limitations
- Reproducibility: data prep notebooks + training script + eval harness

## Tech stack

- Python 3.11 · PyTorch · Hugging Face `transformers` + `datasets`
- `wandb` for experiment tracking
- `gradio` for the demo
- Trained on Colab Pro / single consumer GPU

## Roadmap

3-week plan in [`docs/ROADMAP.md`](docs/ROADMAP.md). Starts after [paper-companion](https://github.com/Aboubekrin999/paper-companion) v1 is live (early May 2026).

## Local development

> Coming as data prep and training scripts land. Will document `data/` build (week 1), training (week 2), and reproducible eval (week 3).

## Author

**Aboubekrin Mohamed Salem** — AI Master's student, working in English and French. Building this as the depth piece of a job-search portfolio.

GitHub: [@Aboubekrin999](https://github.com/Aboubekrin999)
