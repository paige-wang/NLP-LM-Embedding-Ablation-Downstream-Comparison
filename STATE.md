# STATE.md — Dynamic Project State

Cross-references: [CLAUDE.md](CLAUDE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [EXPERIMENTS.md](EXPERIMENTS.md)

**Last updated:** 2026-03-01

---

## Current Phase

**Phase 4 — Core Implementation Complete + Smoke Validation**

---

## Completed

- [x] Read and aligned implementation with `CLAUDE.md` and `PROJECT_BLUEPRINT.md`
- [x] Implemented data modules: tokenizer, vocabulary, LM dataset, downstream dataset, dataset pipeline
- [x] Implemented models: NGram, RNN, LSTM, custom Transformer (causal mask)
- [x] Implemented embedding ablation modules: trainable, fixed-self (Word2Vec), fixed-pretrained (GloVe)
- [x] Implemented downstream modules: intent classifier and trainer
- [x] Implemented scripts: `train_lm.py`, `ablation.py`, `downstream.py`
- [x] Added experiment auto-logging to `outputs/EXPERIMENTS.md`
- [x] Added checkpoint/log/figure output paths and generation
- [x] Verified all required command paths via lightweight smoke runs

---

## In Progress

- [ ] Full-scale server training runs for final convergence baselines
- [ ] Populate root `EXPERIMENTS.md` summary tables with full-run metrics
- [ ] Prepare final report figures/tables from server outputs

---

## Notes

- Local validation used reduced settings (`epochs=1`, limited steps) to verify engineering correctness.
- Offline fallback datasets are included so scripts remain runnable without network access.
- Git commit completed locally on `main`:
  - `646e18d feat: complete Phase 1 data pipeline, model architectures, and smoke tests`
- Remote appears configured as:
  - `https://github.com/paige-wang/NLP-LM-Embedding-Ablation-Downstream-Comparison.git`
- Upload workflow validated:
  - `git push` succeeded for `main` (`9e22eda..646e18d`) after running outside sandbox network restrictions.
