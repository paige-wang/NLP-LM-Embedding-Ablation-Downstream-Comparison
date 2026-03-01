# CLAUDE.md — Global Conventions

Authoritative reference for all working conventions in this repository.
Read before touching any code or docs.

Cross-references:
- Module layout & class designs → [ARCHITECTURE.md](ARCHITECTURE.md)
- Task status & backlog         → [STATE.md](STATE.md)
- Experiment results & configs  → [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 1. Project Overview

**Course:** ST5230 Applied Natural Language Processing
**Assignment:** 1 — Language Model Embedding Ablation & Downstream Comparison
**Deadline:** [TBD]
**Deliverables:** PDF report (≤6 pages) + code (GitHub link or zip)
**Grading target:** High Credit (A range) — careful control, insightful comparisons, clear interpretation

| Part | Points | Scope |
|------|--------|-------|
| I    | 40     | Train & compare n-gram, RNN, LSTM, Transformer LMs |
| II   | 30     | Embedding ablation: trainable / fixed-self-trained / fixed-pretrained |
| III  | 30     | Downstream task (sentiment) with frozen vs. fine-tuned LM representations |

See `requirements.md` for the full rubric.

---

## 2. Tech Stack

```
python        >= 3.10
torch         >= 2.2.0        # training framework
transformers  >= 4.40.0       # pretrained models & tokenizers
datasets      >= 2.19.0       # dataset loading
tokenizers    >= 0.19.0       # fast tokenizers
gensim        >= 4.3.0        # Word2Vec / GloVe training
numpy         >= 1.26.0
scikit-learn  >= 1.4.0        # downstream metrics
matplotlib    >= 3.8.0        # plotting
tqdm          >= 4.66.0       # progress bars
```

**Device:** CUDA GPU preferred; CPU fallback must work.
**Package manager:** pip + `requirements.txt` in project root.

---

## 3. Repository Layout

```
.
├── CLAUDE.md              ← this file
├── ARCHITECTURE.md
├── STATE.md
├── EXPERIMENTS.md
├── requirements.md        ← assignment spec (read-only reference)
├── requirements.txt       ← pip dependencies
├── data/
│   ├── raw/               ← original downloaded data
│   └── processed/         ← tokenised, split, cached tensors
├── src/
│   ├── data/              ← dataset, tokenizer, vocab
│   ├── models/            ← ngram, rnn, lstm, transformer LMs
│   ├── embeddings/        ← embedding factory & three variants
│   ├── downstream/        ← classifier head & fine-tune loop
│   └── utils/             ← metrics, logging, seed, config
├── scripts/
│   ├── train_lm.py        ← Part I driver
│   ├── ablation.py        ← Part II driver
│   └── downstream.py      ← Part III driver
├── notebooks/             ← EDA and result visualization
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

---

## 4. Code Conventions

- **Style:** PEP 8; max line length 100. Run `black` + `isort` before committing.
- **Naming:** `snake_case` for variables/functions/modules; `PascalCase` for classes.
- **Type hints:** Required on all public function signatures.
- **Docstrings:** Google-style on all public classes and functions.
- **Config:** All hyperparameters live in dataclass configs or YAML files under `src/utils/config.py`; no magic numbers in model code.
- **Logging:** Use Python `logging` module; write to `outputs/logs/<run_name>.log`.

---

## 5. Reproducibility Principles

1. **Set global seed** (`src/utils/seed.py`) at the top of every training script: `set_seed(42)`.
2. **Log all hyperparameters** at the start of each run.
3. **Save checkpoints** to `outputs/checkpoints/<model>_<run_id>/`.
4. **Never commit raw data** — document download instructions in `data/README.md`.
5. **Pin library versions** in `requirements.txt`.

---

## 6. Experiment Tracking

- All results recorded in `EXPERIMENTS.md` immediately after a run.
- Update `STATE.md` task status after each milestone.
- One experiment = one row in the relevant results table in `EXPERIMENTS.md`.
