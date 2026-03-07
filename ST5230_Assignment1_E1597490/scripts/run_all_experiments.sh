#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# ── User-configurable paths ──────────────────────────────────────────────────
GLOVE_PATH="${GLOVE_PATH:-data/raw/glove.6B.300d.txt}"
CKPT_PATH="${CKPT_PATH:-}"  # Set to best Transformer checkpoint path

echo "===================="
echo "Stage 1: Train Base Language Models"
echo "===================="

for model in ngram rnn lstm transformer; do
  echo "[Stage 1] Running train_lm for model=${model}"
  python scripts/train_lm.py --model "${model}" || echo "Error in execution, continuing..."
done

echo "===================="
echo "Stage 1 Completed"
echo "===================="

echo "===================="
echo "Stage 2: Embedding Ablation (3 models x 3 embeddings)"
echo "===================="

for model in rnn lstm transformer; do
  for embed in trainable fixed_self fixed_pretrained; do
    echo "[Stage 2] Running ablation for model=${model}, embed=${embed}"
    python scripts/ablation.py --model "${model}" --embed "${embed}" --glove_path "${GLOVE_PATH}" || echo "Error in execution, continuing..."
  done
done

echo "===================="
echo "Stage 2 Completed"
echo "===================="

echo "===================="
echo "Stage 3: Downstream Intent Classification"
echo "===================="

# 3a: Frozen with different initializations
echo "[Stage 3] Frozen — Random Init"
python scripts/downstream.py --mode frozen --embed_init random || echo "Error, continuing..."

echo "[Stage 3] Frozen — GloVe Init"
python scripts/downstream.py --mode frozen --embed_init glove --glove_path "${GLOVE_PATH}" || echo "Error, continuing..."

if [ -n "${CKPT_PATH}" ]; then
  echo "[Stage 3] Frozen — WikiText Pretrained"
  python scripts/downstream.py --mode frozen --ckpt_path "${CKPT_PATH}" || echo "Error, continuing..."

  # 3b: Fine-tune with pretrained checkpoint
  echo "[Stage 3] Fine-tune — WikiText Pretrained"
  python scripts/downstream.py --mode finetune --ckpt_path "${CKPT_PATH}" || echo "Error, continuing..."
else
  echo "[Stage 3] Skipping pretrained runs: CKPT_PATH not set"
fi

echo "===================="
echo "Stage 3 Completed"
echo "===================="

echo "===================="
echo "All experiment stages finished."
echo "===================="
