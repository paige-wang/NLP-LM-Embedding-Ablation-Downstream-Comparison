#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

GLOVE_PATH="${GLOVE_PATH:-data/raw/glove.6B.300d.txt}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

MODELS=(rnn lstm transformer)
EMBEDS=(trainable fixed_self fixed_pretrained)

mkdir -p outputs/logs outputs/checkpoints

current_model=""
current_embed=""

trap 'echo "[Part II] failed at model=${current_model:-<unset>} embed=${current_embed:-<unset>}" >&2' ERR

echo "===================="
echo "CUDA preflight"
echo "===================="
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; aborting Part II rerun.")
print("torch.cuda.current_device =", torch.cuda.current_device())
print("torch.cuda.get_device_name =", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

is_completed() {
  local model="$1"
  local embed="$2"
  local target_epochs="$3"
  "${PYTHON_BIN}" - "$model" "$embed" "$target_epochs" <<'PY'
from pathlib import Path
import re
import sys

model, embed, target_epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
pattern = f"{model}_{embed}_*.log"
epoch_re = re.compile(r"epoch=(\d+)\s+train_loss=")
best = 0
for path in Path("outputs/logs").glob(pattern):
    epochs = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = epoch_re.search(line)
                if match:
                    epochs = max(epochs, int(match.group(1)))
    except OSError:
        continue
    best = max(best, epochs)
sys.exit(0 if best >= target_epochs else 1)
PY
}

echo "===================="
echo "Part II corrected grid rerun"
echo "repo: ${REPO_ROOT}"
echo "python: ${PYTHON_BIN}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
echo "epochs: ${EPOCHS}"
echo "batch_size: ${BATCH_SIZE}"
echo "max_steps: ${MAX_STEPS}"
echo "glove_path: ${GLOVE_PATH}"
echo "data_dir: ${DATA_DIR:-<default>}"
echo "skip_completed: ${SKIP_COMPLETED}"
echo "models: ${MODELS[*]}"
echo "embeds: ${EMBEDS[*]}"
echo "===================="

if [[ " ${EMBEDS[*]} " == *" fixed_pretrained "* ]] && [[ ! -f "${GLOVE_PATH}" ]]; then
  echo "ERROR: GloVe file not found at ${GLOVE_PATH}" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  for embed in "${EMBEDS[@]}"; do
    current_model="${model}"
    current_embed="${embed}"
    if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed "${model}" "${embed}" "${EPOCHS}"; then
      echo "[Part II] skip completed model=${model} embed=${embed}"
      continue
    fi
    echo "[Part II] model=${model} embed=${embed}"
    cmd=(
      "${PYTHON_BIN}" scripts/train_lm.py
      --model "${model}"
      --embed "${embed}"
      --epochs "${EPOCHS}"
      --batch_size "${BATCH_SIZE}"
      --max_steps "${MAX_STEPS}"
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
