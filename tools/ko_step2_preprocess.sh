#!/usr/bin/env bash
# Step 2: Preprocess Korean dataset with the newly trained tokenizer.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경을 활성화하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
GPT_CHECKPOINT="${GPT_CHECKPOINT:-/mnt/sda1/models/IndexTTS-2/gpt.pth}"
LANGUAGE_HINT="${LANGUAGE_HINT:-ko}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
VAL_RATIO="${VAL_RATIO:-0.01}"
DEVICE="${DEVICE:-cuda}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/preprocess_data.py"
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --tokenizer "${TOKENIZER_MODEL}"
  --config "${CONFIG_PATH}"
  --gpt-checkpoint "${GPT_CHECKPOINT}"
  --language "${LANGUAGE_HINT}"
  --batch-size "${BATCH_SIZE}"
  --workers "${NUM_WORKERS}"
  --val-ratio "${VAL_RATIO}"
  --device "${DEVICE}"
  --skip-existing
)

if [[ "${MAX_SAMPLES}" -gt 0 ]]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

echo "[KO-STEP2] manifest=${MANIFEST}"
echo "[KO-STEP2] output-dir=${OUTPUT_DIR}"
echo "[KO-STEP2] tokenizer=${TOKENIZER_MODEL}"

if [[ "${TIMEOUT_SECS}" -gt 0 ]]; then
  timeout "${TIMEOUT_SECS}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "[KO-STEP2] preprocessing complete."
