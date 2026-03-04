# STATE.md — Dynamic Project State

Cross-references: [CLAUDE.md](CLAUDE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [EXPERIMENTS.md](EXPERIMENTS.md)

**Last updated:** 2026-03-04

---

## Current Phase

**Phase 5 — Bug Fixes + New Experiments + Awaiting Server Results**

---

## Completed

- [x] Read and aligned implementation with `CLAUDE.md` and `PROJECT_BLUEPRINT.md`
- [x] Implemented data modules: tokenizer, vocabulary, LM dataset, downstream dataset, dataset pipeline
- [x] Implemented models: NGram, RNN, LSTM, custom Transformer (causal mask)
- [x] Implemented embedding ablation modules: trainable, fixed-self (Word2Vec), fixed-pretrained (GloVe)
- [x] Implemented downstream modules: intent classifier and trainer
- [x] Implemented scripts: `train_lm.py`, `ablation.py`, `downstream.py`
- [x] Added one-click full-run script: `scripts/run_all_experiments.sh`
- [x] Added experiment auto-logging to `outputs/EXPERIMENTS.md`
- [x] Added checkpoint/log/figure output paths and generation
- [x] Verified all required command paths via lightweight smoke runs
- [x] **[2026-03-04] Bug fix:** `fixed_pretrained.py` now raises `FileNotFoundError` when GloVe file missing (was silently producing `glove_coverage=0.0000` and random-noise embeddings)
- [x] **[2026-03-04]** `ablation.py`: added `--glove_path` CLI arg, passes to `EmbeddingConfig`
- [x] **[2026-03-04]** `config.py`: `glove_path` default changed to `""` (force explicit user input)
- [x] **[2026-03-04]** `downstream.py`: added `--embed_init` (random/glove/pretrained) and `--glove_path` for Part III initialization strategy comparison experiments
- [x] **[2026-03-04]** `run_all_experiments.sh`: Stage 2 passes `--glove_path`; Stage 3 expanded to 4 runs (random/glove/pretrained frozen + finetune)
- [x] **[2026-03-04]** New `scripts/visualize.py`: unified figure generation (convergence, coverage, t-SNE, cosine, transfer, downstream subcommands)
- [x] **[2026-03-04]** `EXPERIMENTS.md` and `STATE.md` updated with corrected dataset info, bug note, new experiment design
- [x] **[2026-03-04]** Report draft (`report_for_overleaf.tex`) committed: Part I complete with real data; Part II/III placeholder sections ready for post-server fill-in
- [x] All changes pushed to GitHub (`main` branch, commit `3c3d899`)

---

## In Progress / Blocked

- [ ] **[BLOCKED: GloVe download]** Part II GloVe rows — need `data/raw/glove.6B.300d.txt` on server
  - Download: `wget https://nlp.stanford.edu/data/glove.6B.zip && unzip -j glove.6B.zip glove.6B.300d.txt -d data/raw/`
- [ ] Full-scale server training runs (Part I 30 epochs, Part II 3×3 grid, Part III 4 runs)
- [ ] Populate `EXPERIMENTS.md` summary tables with full-run metrics
- [ ] Generate figures via `scripts/visualize.py all`
- [ ] Write Part II + Part III + Discussion sections in `report_for_overleaf.tex`

---

## Next Actions (server-side)

```bash
git pull                          # get latest code
# download GloVe
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip -j glove.6B.zip glove.6B.300d.txt -d data/raw/
# run all experiments
export GLOVE_PATH=data/raw/glove.6B.300d.txt
export CKPT_PATH=outputs/checkpoints/<best_transformer>/model.pt
bash scripts/run_all_experiments.sh
# generate figures
python scripts/visualize.py all --glove_path $GLOVE_PATH --ckpt_path $CKPT_PATH
```

---

## Notes

- Local validation used reduced settings (`epochs=1`, limited steps) to verify engineering correctness.
- Offline fallback datasets are included so scripts remain runnable without network access.
- Remote: `https://github.com/paige-wang/NLP-LM-Embedding-Ablation-Downstream-Comparison.git`
- Latest push: `main` at commit `3c3d899` (2026-03-04)
