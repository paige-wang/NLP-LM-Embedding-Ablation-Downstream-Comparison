# EXPERIMENTS.md — Experiment Matrix

Cross-references: [CLAUDE.md](CLAUDE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [STATE.md](STATE.md)

**Last updated:** 2026-03-04

---

## Dataset

| Property | Value |
|----------|-------|
| LM Corpus | WikiText-2 |
| Domain | Wikipedia articles |
| Total tokens (train) | ~2.1M |
| Vocabulary size | 26,006 (joint with SNIPS) |
| Train / Val / Test | Standard WikiText-2 splits |
| Downstream dataset | SNIPS (7-class intent classification) |
| SNIPS train / val / test | 13,784 / 700 / 700 |
| Tokenization | Whitespace (SimpleTokenizer), lowercase |
| Download source | `datasets` HuggingFace hub (wikitext, SNIPS) |

---

## Global Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| Random seed | 42 | All runs |
| Batch size (LM) | 64 | |
| Batch size (downstream) | 32 | |
| Sequence length (LM) | 64 | Sliding window, stride 1 |
| Sequence length (downstream) | 32 | |
| Optimizer | AdamW | weight_decay=0.01 |
| LM learning rate | 3e-4 | Linear warmup 1,000 steps |
| Downstream backbone LR | 5e-5 | Fine-tune mode |
| Downstream head LR | 5e-4 | = backbone × 10 |
| Frozen mode LR | 1e-3 | Head only |
| Max epochs (LM) | 30 | |
| Max epochs (downstream) | 20 | Early stopping patience=5 |
| Gradient clip | 1.0 | All runs |
| Device | CUDA / CPU fallback | |

---

## Part I Results — Language Model Comparison

Results from 30-epoch full runs on server (to be filled after server run).

| Model | Params | Epoch Time | Inference | Train Loss | Val PPL |
|-------|--------|-----------|-----------|-----------|---------|
| Trigram (n=3) | — | <1s | <1ms | — | 25.15 |
| RNN | 1.12M | 0.33s | 4.2ms | 3.23 | 27.49 |
| LSTM | 3.94M | 0.48s | 5.8ms | 3.22 | 25.67 |
| Transformer | 4.35M | 0.51s | 2.1ms | 3.40 | **17.87** |

*Timing measured on NVIDIA RTX 4090D, batch size 64.*

**Model Hyperparameters (Part I):**

| Model | Specific Params |
|-------|----------------|
| Trigram | n=3, Laplace α=1.0 |
| RNN | hidden=512, layers=2, dropout=0.5 |
| LSTM | hidden=512, layers=2, dropout=0.5 |
| Transformer | d_model=300, nhead=6, layers=4, ffn=1200, Pre-Norm, GELU, sinusoidal PE |

---

## Part II Results — Embedding Ablation

**3×3 grid**: RNN / LSTM / Transformer × Trainable / Word2Vec / GloVe

> **Bug note (2026-03-04 fixed):** Previous runs showed `glove_coverage=0.0000` because
> `data/raw/glove.6B.300d.txt` was missing and the code silently fell back to random noise.
> `fixed_pretrained.py` now raises `FileNotFoundError`. All GloVe rows below require
> re-running with `--glove_path data/raw/glove.6B.300d.txt` after downloading GloVe 6B.

| Model | Embedding | Val PPL | Δ vs Trainable | GloVe Coverage | Status |
|-------|-----------|---------|---------------|----------------|--------|
| RNN | Trainable | TBD | baseline | — | pending server run |
| RNN | Word2Vec | TBD | TBD | ~N/A | pending server run |
| RNN | GloVe | TBD | TBD | ~60-80% (expected) | pending GloVe download |
| LSTM | Trainable | TBD | baseline | — | pending server run |
| LSTM | Word2Vec | TBD | TBD | — | pending server run |
| LSTM | GloVe | TBD | TBD | ~60-80% (expected) | pending GloVe download |
| Transformer | Trainable | TBD | baseline | — | pending server run |
| Transformer | Word2Vec | TBD | TBD | — | pending server run |
| Transformer | GloVe | TBD | TBD | ~60-80% (expected) | pending GloVe download |

**Embedding Hyperparameters:**

| Setting | Params |
|---------|--------|
| Trainable | dim=300, jointly optimized |
| Word2Vec | dim=300, window=5, min_count=2, epochs=10 |
| GloVe | GloVe-6B-300d, source=nlp.stanford.edu/data/glove.6B.zip |

**Server commands to reproduce:**
```bash
export GLOVE_PATH=data/raw/glove.6B.300d.txt
for model in rnn lstm transformer; do
  for embed in trainable fixed_self fixed_pretrained; do
    python scripts/ablation.py --model $model --embed $embed \
      --glove_path $GLOVE_PATH --epochs 30
  done
done
```

---

## Part III Results — Downstream Task

**Task:** 7-class Intent Classification
**Dataset:** SNIPS (13,784 train / 700 val / 700 test)
**Backbone:** Transformer LM pretrained on WikiText-2 (best checkpoint from Part I)
**Pooling:** Last-token pooling → Linear(300 → 7)

### Frozen vs Fine-tuned (existing results)

| Setting | Init | Test Acc | Macro-F1 | Epochs | Notes |
|---------|------|---------|---------|--------|-------|
| Frozen | WikiText pretrained | 91.57% | 91.55% | — | high linear separability |
| Fine-tuned | WikiText pretrained | **97.07%** | **97.05%** | 17 | early stopping; differential LR |

### Initialization Strategy Comparison — NEW (pending server run)

Isolates the contribution of embedding initialization vs. full pretraining.
All three rows use **Frozen** mode (backbone weights not updated during downstream training).

| Init Strategy | Embedding Source | Test Acc | Macro-F1 | Notes |
|--------------|----------------|---------|---------|-------|
| Random | Random N(0, 0.01) | TBD | TBD | baseline — no prior knowledge |
| GloVe Init | GloVe-6B-300d (embed layer only) | TBD | TBD | general-domain semantic prior |
| WikiText Pretrained | Full LM checkpoint | 91.57% | 91.55% | domain-adapted representations |
| Fine-tuned | WikiText Pretrained | 97.07% | 97.05% | upper bound |

**Research question:** Does downstream performance come from the model architecture
or from WikiText pretraining? Can GloVe's general-domain embeddings substitute
for task-domain pretraining?

**Server commands to reproduce:**
```bash
export CKPT_PATH=outputs/checkpoints/<best_transformer_run>/model.pt
export GLOVE_PATH=data/raw/glove.6B.300d.txt

python scripts/downstream.py --mode frozen --embed_init random
python scripts/downstream.py --mode frozen --embed_init glove --glove_path $GLOVE_PATH
python scripts/downstream.py --mode frozen --ckpt_path $CKPT_PATH
python scripts/downstream.py --mode finetune --ckpt_path $CKPT_PATH
```

---

## Key Findings

*(Updated as experiments complete)*

- **Part I:** Transformer achieves lowest Val PPL (17.87) with fastest inference (2.1ms). LSTM > RNN > Trigram. Trigram (25.15) is competitive with LSTM (25.67) — local bigram statistics capture substantial signal.
- **Part II:** Pending corrected server run (GloVe bug fixed 2026-03-04). Expected: trainable > fixed_self > fixed_pretrained for in-domain perplexity; GloVe may offer faster convergence.
- **Part III:** Frozen backbone achieves 91.57% — representations are strongly linearly separable. Fine-tuning raises to 97.07%. Initialization strategy comparison pending.

---

## Visualization Scripts (2026-03-04)

```bash
# After server runs, generate all figures:
python scripts/visualize.py all \
  --glove_path data/raw/glove.6B.300d.txt \
  --ckpt_path outputs/checkpoints/<best_run>/model.pt \
  --output_dir outputs/figures/

# Or individual subcommands:
python scripts/visualize.py convergence   # Part I & II curves
python scripts/visualize.py coverage     # vocab coverage bars
python scripts/visualize.py tsne         # embedding space
python scripts/visualize.py cosine       # similarity heatmaps
python scripts/visualize.py transfer     # init strategy bars
python scripts/visualize.py downstream   # downstream curves
```

---

## Smoke Validation (2026-03-01)

These are lightweight functional checks (short runs) to verify code paths and artifact generation.

| Command | Status | Notes |
|---------|--------|-------|
| `python scripts/train_lm.py --model ngram` | Pass | Logged val/test PPL |
| `python scripts/train_lm.py --model rnn --epochs 1 --max_steps 2 --batch_size 8` | Pass | Checkpoint + log emitted |
| `python scripts/train_lm.py --model lstm --epochs 1 --max_steps 2 --batch_size 8` | Pass | Checkpoint + log emitted |
| `python scripts/train_lm.py --model transformer --epochs 1 --max_steps 2 --batch_size 8` | Pass | Checkpoint + log emitted |
| `python scripts/ablation.py --model rnn --embed trainable --epochs 1 --max_steps 2 --batch_size 8` | Pass | Part II row appended |
| `python scripts/ablation.py --model transformer --embed fixed_pretrained --epochs 1 --max_steps 2 --batch_size 8` | Pass | GloVe coverage logged |
| `python scripts/downstream.py --mode frozen --epochs 1 --max_steps 2 --batch_size 8` | Pass | Confusion matrix saved |
| `python scripts/downstream.py --mode finetune --epochs 1 --max_steps 2 --batch_size 8` | Pass | Confusion matrix saved |
