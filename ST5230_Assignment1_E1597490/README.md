# ST5230 Assignment 1 Submission

Student ID: `E1597490`

This folder is the cleaned submission package for `ST5230 Assignment 1`. It contains the final report source, the code used for all experiments, the datasets required to reproduce the runs, and the compact result summaries used in the report.

## What To Submit

The assignment asks for:

- a PDF report with a maximum length of 6 pages
- the code as either a GitHub repository or a zip file

This folder is prepared for the code submission. The final report source is included as:

- `report_for_overleaf.tex`

The final PDF should be exported from Overleaf and submitted separately.

## What Is Included

- `src/`
  - implementation of the language models, embedding variants, downstream classifier, and data pipeline
- `scripts/`
  - runnable scripts for Part I, Part II, Part III, and Part II summarization
- `data/`
  - SNIPS splits and cached WikiText-2 tokenized arrays used in the experiments
- `final_results_20260305/`
  - compact final artifacts used for Part I and Part III in the report
- `part2_results_full_20260307/`
  - compact summary files for the full Part II `3 x 3` study
- `requirements.txt`
  - Python dependency list

## What Is Excluded

To keep the submission package compact and readable, this folder does not duplicate:

- intermediate planning notes
- AI workflow files
- large model checkpoints
- raw server logs that are not needed for grading

The report tables and conclusions are still fully supported by the included summary files.

## Assignment Requirement Coverage

### Part I: Language Model Training and Comparison

Implemented and trained on the same WikiText-2 corpus:

- `n`-gram LM
- RNN LM
- LSTM LM
- Transformer LM

Compared in the report using:

- validation perplexity
- training time
- inference latency
- qualitative generated text samples

Relevant code:

- `scripts/train_lm.py`
- `src/models/`

Relevant artifacts:

- `final_results_20260305/EXPERIMENTS.md`
- `final_results_20260305/outputs/figures/`

### Part II: Embedding Variants and Ablation

Compared three embedding settings in neural language models:

- trainable embeddings learned from scratch
- fixed self-trained embeddings (`Word2Vec`)
- fixed pretrained embeddings (`GloVe-6B-300d`)

Compared across three model families:

- `RNN`
- `LSTM`
- `Transformer`

The final reported Part II study is a full `3 x 3` matrix under a unified compute budget:

- `10` epochs
- up to `400` optimization steps per epoch
- batch size `128`

This budget-controlled setting was used so that all nine runs remained directly comparable under limited compute.

Relevant code:

- `scripts/run_part2_grid.sh`
- `scripts/summarize_part2.py`
- `src/embeddings/`

Relevant artifacts:

- `part2_results_full_20260307/outputs/part2_summary.md`
- `part2_results_full_20260307/outputs/EXPERIMENTS.md`

### Part III: Downstream Task with Learned Representations

Built a simple intent classifier on SNIPS using Transformer representations:

- last-token pooling
- single linear classification head

Compared three transfer settings:

- frozen WikiText-pretrained Transformer
- frozen GloVe-initialized model
- fine-tuned WikiText-pretrained Transformer

Relevant code:

- `scripts/downstream.py`
- `src/downstream/`

Relevant artifacts:

- `final_results_20260305/EXPERIMENTS.md`
- `final_results_20260305/outputs/figures/`

## Main Results Used In The Report

### Part I

- best backbone: `Transformer`
- best validation perplexity: `17.87`

### Part II

Validation perplexity under the unified compute budget:

| Model | Trainable | Fixed Self | Fixed Pretrained |
|---|---:|---:|---:|
| RNN | 183.27 | **165.95** | 187.01 |
| LSTM | 336.58 | **304.21** | 403.49 |
| Transformer | 213.45 | **174.43** | 187.11 |

Additional result:

- `glove_coverage = 0.9199`

### Part III

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Frozen WikiText | 91.57 | 91.55 |
| Frozen GloVe | 95.36 | 95.35 |
| Fine-tuned WikiText | **97.07** | **97.05** |

## How To Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the Part I models:

```bash
python scripts/train_lm.py --model ngram
python scripts/train_lm.py --model rnn --embed trainable
python scripts/train_lm.py --model lstm --embed trainable
python scripts/train_lm.py --model transformer --embed trainable
```

Run the Part II grid:

```bash
bash scripts/run_part2_grid.sh
python scripts/summarize_part2.py --min_epochs 10
```

Run the Part III downstream experiments:

```bash
python scripts/downstream.py --mode frozen --embed_init pretrained --ckpt_path <transformer_checkpoint>
python scripts/downstream.py --mode frozen --embed_init glove --glove_path data/raw/glove.6B.300d.txt
python scripts/downstream.py --mode finetune --embed_init pretrained --ckpt_path <transformer_checkpoint>
```

If `GloVe-6B-300d` is needed, place:

- `glove.6B.300d.txt`

at:

- `data/raw/glove.6B.300d.txt`

## Compute Work Completed

In addition to the local experiments, substantial server-side work was required to complete the final Part II matrix.

Completed work included:

- adapting `scripts/run_part2_grid.sh` for stable CUDA execution on AutoDL
- making `scripts/train_lm.py` compatible with the required CLI variants
- fixing the self-trained embedding pipeline so `Word2Vec` explicitly builds vocabulary before training
- running the full Part II `3 x 3` matrix on an RTX `4090D 24GB` server
- collecting the result summaries back into this project for reporting

These steps are reflected in the final code under `scripts/` and `src/embeddings/`.
