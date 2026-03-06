#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

GLOVE_PATH="${GLOVE_PATH:-data/raw/glove.6B.300d.txt}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-}"

MODELS=(rnn lstm transformer)
EMBEDS=(trainable fixed_self fixed_pretrained)

echo "===================="
echo "Part II corrected grid rerun"
echo "repo: ${REPO_ROOT}"
echo "python: ${PYTHON_BIN}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "max_steps: ${MAX_STEPS}"
echo "glove_path: ${GLOVE_PATH}"
echo "data_dir: ${DATA_DIR:-<default>}"
echo "models: ${MODELS[*]}"
echo "embeds: ${EMBEDS[*]}"
echo "===================="

if [[ " ${EMBEDS[*]} " == *" fixed_pretrained "* ]] && [[ ! -f "${GLOVE_PATH}" ]]; then
  echo "ERROR: GloVe file not found at ${GLOVE_PATH}" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  for embed in "${EMBEDS[@]}"; do
    echo "[Part II] model=${model} embed=${embed}"
    cmd=(
      "${PYTHON_BIN}" scripts/train_lm.py
      --model "${model}" \
      --embed "${embed}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_steps "${MAX_STEPS}" \
      --glove_path "${GLOVE_PATH}"
    )
    if [[ -n "${DATA_DIR}" ]]; then
      cmd+=(--data_dir "${DATA_DIR}")
    fi
    "${cmd[@]}"
  done
done

echo "===================="
echo "Part II reruns completed"
echo "Next: python scripts/summarize_part2.py"
echo "===================="
