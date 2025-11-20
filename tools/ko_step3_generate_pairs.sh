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

usage() {
  cat <<'USAGE'
Usage: ko_step3_generate_pairs.sh [--force]

Environment variables can also be used to tweak behavior:
  DATASET_DIR, TRAIN_OUTPUT_NAME, VAL_OUTPUT_NAME, PAIRS_PER_TARGET,
  MIN_TEXT_LEN, MIN_CODE_LEN, MAX_PAIRS, SEED, FORCE, TIMEOUT_SECS, PYTHON_BIN

Flags:
  --force     Overwrite existing output files (equivalent to FORCE=1)
  -h, --help  Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

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
