#!/usr/bin/env bash
# Step 3: Generate GPT prompt/target pairs from the Korean processed dataset.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경을 활성화하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
TRAIN_OUTPUT_NAME="${TRAIN_OUTPUT_NAME:-gpt_pairs_train.jsonl}"
VAL_OUTPUT_NAME="${VAL_OUTPUT_NAME:-gpt_pairs_val.jsonl}"
PAIRS_PER_TARGET="${PAIRS_PER_TARGET:-2}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-1}"
MIN_CODE_LEN="${MIN_CODE_LEN:-1}"
MAX_PAIRS="${MAX_PAIRS:-0}"
SEED="${SEED:-2025}"
FORCE="${FORCE:-0}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/generate_gpt_pairs.py"
  --dataset "${DATASET_DIR}"
  --train-output-name "${TRAIN_OUTPUT_NAME}"
  --val-output-name "${VAL_OUTPUT_NAME}"
  --pairs-per-target "${PAIRS_PER_TARGET}"
  --min-text-len "${MIN_TEXT_LEN}"
  --min-code-len "${MIN_CODE_LEN}"
  --seed "${SEED}"
)

if [[ "${MAX_PAIRS}" -gt 0 ]]; then
  CMD+=(--max-pairs "${MAX_PAIRS}")
fi
if [[ "${FORCE}" == "1" ]]; then
  CMD+=(--force)
fi

echo "[KO-STEP3] dataset-dir=${DATASET_DIR}"
echo "[KO-STEP3] outputs=${TRAIN_OUTPUT_NAME}, ${VAL_OUTPUT_NAME}"

if [[ "${TIMEOUT_SECS}" -gt 0 ]]; then
  timeout "${TIMEOUT_SECS}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "[KO-STEP3] GPT pairs generation complete."
