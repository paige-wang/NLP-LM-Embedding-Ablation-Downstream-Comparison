# ST5230 Assignment 1

Student ID: `E1597490`

This repository contains the code, report source, and final experiment artifacts for ST5230 Assignment 1 on language modeling, embedding ablation, and downstream transfer.

## Repository Contents

- `src/`: model, embedding, data, downstream, and utility modules
- `scripts/`: runnable entry points for Part I, Part II, Part III, and result summarization
- `data/`: cached processed LM data and SNIPS splits used in the experiments
- `final_results_20260305/`: final Part I and Part III artifacts used in the report
- `part2_results_full_20260307/`: final Part II `3 x 3` budget-controlled rerun artifacts from AutoDL
- `report_for_overleaf.tex`: final LaTeX source of the report
- `requirements.md`: original assignment requirements
- `requirements.txt`: Python dependencies

## Assignment Coverage

The submission covers all three required parts.

### Part I: Language Model Training and Comparison

Trained and compared:

- `n`-gram LM
- RNN LM
- LSTM LM
- decoder-only Transformer LM

Compared with:

- validation perplexity
- training time
- inference latency
- generated text samples

### Part II: Embedding Variants and Ablation

Compared three embedding settings in neural LMs:

- trainable embeddings learned from scratch
- fixed self-trained embeddings (`Word2Vec`)
- fixed pretrained embeddings (`GloVe-6B-300d`)

The final reported Part II matrix is a full `3 x 3` comparison over:

- architectures: `RNN`, `LSTM`, `Transformer`
- embeddings: `trainable`, `fixed_self`, `fixed_pretrained`

Because the complete grid was run on a rented AutoDL server, the final Part II study is reported as a unified-compute comparison:

- `10` epochs
- at most `400` optimization steps per epoch
- batch size `128`
- NVIDIA RTX `4090D 24GB`

This design keeps all nine runs directly comparable under limited compute.

### Part III: Downstream Task

Built a simple SNIPS intent classifier from Transformer representations:

- last-token pooling
- single linear classification head

Compared:

- frozen WikiText-pretrained Transformer
- frozen GloVe-initialized model
- full fine-tuning

## Main Final Results

### Part I

- Best LM backbone: `Transformer`
- Best validation perplexity: `17.87`

### Part II

Final validation perplexity under the unified compute budget:

| Model | Trainable | Fixed Self | GloVe |
|---|---:|---:|---:|
| RNN | 183.27 | **165.95** | 187.01 |
| LSTM | 336.58 | **304.21** | 403.49 |
| Transformer | 213.45 | **174.43** | 187.11 |

Additional Part II fact:

- `glove_coverage = 0.9199`

### Part III

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen WikiText | 91.57 | 91.55 |
| Frozen GloVe | 95.36 | 95.35 |
| Fine-tuned WikiText | **97.07** | **97.05** |

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

### Part I

Train a language model:

```bash
python scripts/train_lm.py --model ngram
python scripts/train_lm.py --model rnn --embed trainable
python scripts/train_lm.py --model lstm --embed trainable
python scripts/train_lm.py --model transformer --embed trainable
```

### Part II

Single run:

```bash
python scripts/train_lm.py --model transformer --embed fixed_self
python scripts/train_lm.py --model transformer --embed fixed_pretrained --glove_path data/raw/glove.6B.300d.txt
```

Full grid:

```bash
bash scripts/run_part2_grid.sh
python scripts/summarize_part2.py --min_epochs 10
```

### Part III

Frozen transfer:

```bash
python scripts/downstream.py --mode frozen --embed_init pretrained --ckpt_path <transformer_checkpoint>
python scripts/downstream.py --mode frozen --embed_init glove --glove_path data/raw/glove.6B.300d.txt
```

Fine-tuning:

```bash
python scripts/downstream.py --mode finetune --embed_init pretrained --ckpt_path <transformer_checkpoint>
```

## Data Notes

- WikiText-2 and SNIPS are used in this project.
- `data/snips/` is included.
- `data/processed/` contains cached token IDs and vocabulary built for the experiments.
- `data/raw/` is not included; if GloVe is needed, place `glove.6B.300d.txt` at `data/raw/glove.6B.300d.txt`.
- Additional download details are in `data/README.md`.

## Final Artifact Locations

- Report source: `report_for_overleaf.tex`
- Part I / Part III artifacts: `final_results_20260305/`
- Part II full server rerun artifacts: `part2_results_full_20260307/`

Important Part II files:

- `part2_results_full_20260307/outputs/part2_summary.md`
- `part2_results_full_20260307/outputs/logs/`
- `part2_results_full_20260307/outputs/checkpoints/`

## AutoDL Work Log Summary

The final Part II matrix was completed on AutoDL with an RTX 4090D 24GB GPU.

Server-side work included:

- hardening `scripts/run_part2_grid.sh` for CUDA execution and resumable grid runs
- adding backward-compatible CLI support in `scripts/train_lm.py`
- fixing `fixed_self` embedding training so `Word2Vec` explicitly builds vocabulary before training
- rerunning the full Part II `3 x 3` matrix under a controlled compute budget
- downloading the complete logs, checkpoints, and summaries back into this repository

## Submission Note

For course submission, the final deliverable folder is:

- `ST5230_Assignment1_E1597490/`

It contains the cleaned submission-ready copy of the code, report source, and final experiment artifacts needed for grading, excluding AI planning files and intermediate project-management notes. To avoid duplicating very large binaries, the submission folder keeps the final Part II logs and summary but does not duplicate the full checkpoint archive that is already preserved in the repository root.
