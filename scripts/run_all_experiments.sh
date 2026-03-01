#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

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
    python scripts/ablation.py --model "${model}" --embed "${embed}" || echo "Error in execution, continuing..."
  done
done

echo "===================="
echo "Stage 2 Completed"
echo "===================="

echo "===================="
echo "Stage 3: Downstream Intent Classification"
echo "===================="

for mode in frozen finetune; do
  echo "[Stage 3] Running downstream mode=${mode}"
  python scripts/downstream.py --mode "${mode}" || echo "Error in execution, continuing..."
done

echo "===================="
echo "Stage 3 Completed"
echo "===================="

echo "===================="
echo "All experiment stages finished."
echo "===================="
