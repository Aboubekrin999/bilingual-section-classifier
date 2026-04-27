# Architecture Decisions

Lightweight ADRs. Each entry: context → decision → consequences.

---

## ADR-001 — XLM-RoBERTa-base over mBERT or larger XLM-R variants

**Date:** 2026-04-27
**Status:** Accepted

**Context.** Several multilingual encoders are viable: mBERT, XLM-R-base, XLM-R-large, mDeBERTa-v3.

**Decision.** Start with `xlm-roberta-base` (270M).

**Why.**
- Strong baseline for FR + EN on classification tasks.
- 270M fits a single consumer GPU with batch size 16+ at 512 tokens — viable on Colab Pro.
- Larger XLM-R variants (550M+) would need either smaller batch or gradient accumulation, slowing iteration.
- mBERT is older and weaker on French.
- mDeBERTa-v3 is competitive but the FR pretraining mix is less documented.

**Consequences.** If macro-F1 falls short of the published baselines on PubMed-RCT, switch to XLM-R-large or mDeBERTa-v3 in week 2.

---

## ADR-002 — Combine PubMed-RCT, CSAbstruct, and HAL-derived FR set rather than build from scratch

**Date:** 2026-04-27
**Status:** Accepted

**Context.** Could build a fully custom dataset from arXiv + HAL.

**Decision.** Use PubMed-RCT and CSAbstruct as the EN backbone; build a smaller FR set from HAL.

**Why.**
- Existing benchmarks have known label schemas, removes annotation noise.
- Custom data collection is the long pole. HAL FR scrape is targeted; the EN side is downloaded.
- Builds on top of established baselines so results are interpretable.

**Consequences.** Label schema must be unified across the three sources — a small amount of label-mapping work. Documented in week-1 data prep.

---

## ADR-003 — Language-stratified test set

**Date:** 2026-04-27
**Status:** Accepted

**Context.** Standard 80/10/10 split could underrepresent French in eval if FR is the smaller portion.

**Decision.** Stratify by language *and* by class, ensuring the test set has balanced FR + EN coverage with minimum N per class per language.

**Why.** Honest evaluation requires per-language F1. Imbalanced test sets hide language-specific failures. This is the single most defensible methodological choice in the project; recruiters and reviewers look for it.

**Consequences.** Slightly smaller effective training set if FR is rare. Acceptable.

---

## ADR-004 — Hugging Face Hub for model + Spaces for demo, not a custom hosting setup

**Date:** 2026-04-27
**Status:** Accepted

**Context.** Could self-host the demo on Vercel/Railway like the paper-companion API.

**Decision.** Push the trained model to Hugging Face Hub; deploy a Gradio demo on Hugging Face Spaces.

**Why.**
- Zero hosting cost.
- HF Hub model card is the standard artifact ML hiring managers look for.
- Spaces gives a live, public, clickable demo that survives long after this project is "done."
- Matches ecosystem expectations — recruiters know what to look for.

**Consequences.** Model and demo are coupled to HF's platform availability. Acceptable trade.

---

## ADR-005 — Track everything with Weights & Biases

**Date:** 2026-04-27
**Status:** Accepted

**Context.** Could use TensorBoard or just log JSON.

**Decision.** Use W&B for runs, sweeps, and the final report.

**Why.** Recruiter-readable shareable dashboards. Sweep support if hyperparameter search is needed in week 2. Public run links can be embedded in the write-up.

**Consequences.** Free tier is sufficient for personal projects.
