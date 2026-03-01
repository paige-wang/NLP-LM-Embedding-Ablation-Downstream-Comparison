# EXPERIMENTS.md — Experiment Matrix

Cross-references: [CLAUDE.md](CLAUDE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [STATE.md](STATE.md)

**Last updated:** 2026-02-28

---

## Dataset

| Property | Value |
|----------|-------|
| Name | TBD |
| Domain | TBD |
| Total tokens | TBD |
| Vocabulary size | TBD |
| Train split | TBD |
| Validation split | TBD |
| Test split | TBD |
| Tokenization | TBD |
| Download source | TBD |

---

## Global Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| Random seed | 42 | Fixed for all runs |
| Batch size | TBD | |
| Sequence length | TBD | |
| Optimizer | AdamW | |
| Learning rate | TBD | |
| LR scheduler | TBD | |
| Max epochs | TBD | |
| Early stopping patience | TBD | |
| Gradient clip | TBD | |
| Device | CUDA / CPU | |

---

## Part I Results — Language Model Comparison

| Model | Embedding | Train Time | Inference Time | Val PPL | Test PPL | Sample Quality |
|-------|-----------|-----------|---------------|---------|---------|----------------|
| N-gram (n=?) | N/A (count-based) | — | — | — | — | — |
| RNN | Trainable | — | — | — | — | — |
| LSTM | Trainable | — | — | — | — | — |
| Transformer | Trainable | — | — | — | — | — |

**Model Hyperparameters (Part I):**

| Model | Specific Params |
|-------|----------------|
| N-gram | n=TBD, smoothing=Laplace |
| RNN | hidden=TBD, layers=TBD |
| LSTM | hidden=TBD, layers=TBD, dropout=TBD |
| Transformer | d_model=TBD, nhead=TBD, layers=TBD, ffn=TBD |

---

## Part II Results — Embedding Ablation

Base model for ablation: **RNN** and **LSTM** (neural LMs only)

| Model | Embedding Setting | Train Time | Val PPL | Test PPL | Convergence Epochs | Stability |
|-------|------------------|-----------|---------|---------|-------------------|-----------|
| RNN | Trainable (scratch) | — | — | — | — | — |
| RNN | Fixed self-trained (Word2Vec) | — | — | — | — | — |
| RNN | Fixed pretrained (GloVe) | — | — | — | — | — |
| LSTM | Trainable (scratch) | — | — | — | — | — |
| LSTM | Fixed self-trained (Word2Vec) | — | — | — | — | — |
| LSTM | Fixed pretrained (GloVe) | — | — | — | — | — |

**Embedding Hyperparameters:**

| Setting | Params |
|---------|--------|
| Trainable | dim=TBD |
| Word2Vec | dim=TBD, window=TBD, min_count=TBD, epochs=TBD |
| GloVe | source=TBD, dim=TBD |

---

## Part III Results — Downstream Task

**Task:** Sentiment Analysis (binary)
**Dataset:** TBD (IMDb / SST-2)
**LM source:** Best model from Part I/II (TBD)

| Setting | Accuracy | F1 | Precision | Recall | Training Epochs |
|---------|----------|----|-----------|--------|----------------|
| Frozen LM + linear head | — | — | — | — | — |
| Fine-tuned LM + linear head | — | — | — | — | — |

---

## Key Findings

*(Populated as experiments complete)*

- **Part I:** TBD
- **Part II:** TBD
- **Part III:** TBD

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
