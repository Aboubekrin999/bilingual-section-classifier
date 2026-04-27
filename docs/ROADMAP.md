# 3-Week Roadmap to v1

Target: trained model on Hugging Face Hub + live Spaces demo + 5-page write-up.
Budget: ~10 hours per week, starting after [paper-companion](https://github.com/Aboubekrin999/paper-companion) v1 ships.

Each week ends in a checkpoint that can be defended in an interview.

---

## Week 1 — Data
*Tentative: May 11 – May 17*

- [ ] Download and inspect PubMed-RCT and CSAbstruct
- [ ] Define unified label schema across both datasets (Abstract, Introduction, Methods, Results, Discussion, Related Work, Conclusion)
- [ ] Scrape ~5–10k French sentences from HAL open-access PDFs, segmented by LaTeX section markers
- [ ] Manual spot-check on 200 FR sentences — flag noise rate, fix segmentation rules
- [ ] Stratified 80/10/10 split by class + language → final train/val/test parquet files
- [ ] Data card in `docs/DATA.md`: sources, label distribution, language balance, known noise

**Checkpoint.** A clean, versioned dataset on disk with a documented schema. Notebook reproduces the build from raw downloads.

---

## Week 2 — Training
*Tentative: May 18 – May 24*

- [ ] XLM-RoBERTa-base baseline: full fine-tune, default hyperparameters
- [ ] Hyperparameter sweep (learning rate, warmup, weight decay, batch size) via W&B Sweeps
- [ ] Best run: log macro-F1, per-class F1, per-language F1, confusion matrix
- [ ] If macro-F1 < 0.75 on test: retry with XLM-R-large or mDeBERTa-v3
- [ ] Push best model to Hugging Face Hub with a thorough model card

**Checkpoint.** Trained model on HF Hub with reproducible training script and W&B run links.

---

## Week 3 — Demo, write-up, polish
*Tentative: May 25 – May 31*

- [ ] Gradio demo on Hugging Face Spaces (paste a paragraph → see predicted section + confidence)
- [ ] Error analysis: 50 worst-confidence test examples, categorize failure modes
- [ ] 5-page write-up `docs/REPORT.md`: motivation, methods, results, error analysis, limitations, future work
- [ ] Update README header with Spaces demo link + final metrics table
- [ ] (Stretch) Plug the classifier into [paper-companion](https://github.com/Aboubekrin999/paper-companion)'s chunker as section metadata

**Checkpoint.** Recruiter clicks the README, reads a clean writeup, opens the Spaces demo, types a sentence, sees a prediction. Project shippable.

---

## After v1

- Per-section retrieval improvements in paper-companion
- Add Arabic if a small labeled set becomes available (extending the multilingual angle)
- Compare against frozen-encoder + linear-probe baseline for cost/quality trade-off
