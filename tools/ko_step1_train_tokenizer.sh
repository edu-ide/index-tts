#!/usr/bin/env bash
# Step 1: Train a Korean-only SentencePiece tokenizer (Jmica 방식 준용).

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경을 활성화하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MANIFEST="${MANIFEST:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe}"
RAW_PREFIX="${RAW_PREFIX:-${OUTPUT_PREFIX}_raw}"
VOCAB_SIZE="${VOCAB_SIZE:-16000}"
CHAR_COVERAGE="${CHAR_COVERAGE:-0.9995}"
MODEL_TYPE="${MODEL_TYPE:-bpe}"
BYTE_FALLBACK="${BYTE_FALLBACK:-1}"
LANGUAGE="${LANGUAGE:-ko}"
USER_SYMBOLS="${USER_SYMBOLS:-}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/sda1/models/IndexTTS-2/bpe.model}"
RETAIN_COUNT="${RETAIN_COUNT:-0}"
TARGET_SIZE="${TARGET_SIZE:-16000}"
KEEP_RAW="${KEEP_RAW:-0}"

IFS=',' read -ra USER_SYMBOL_ARRAY <<< "${USER_SYMBOLS}"
USER_SYMBOL_FLAGS=()
for symbol in "${USER_SYMBOL_ARRAY[@]}"; do
  trimmed="$(echo "${symbol}" | xargs)"
  [[ -z "${trimmed}" ]] && continue
  USER_SYMBOL_FLAGS+=("--user-defined-symbol" "${trimmed}")
done

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/tokenizer/train_bpe.py"
  --manifest "${MANIFEST}"
  --output-prefix "${RAW_PREFIX}"
  --vocab-size "${VOCAB_SIZE}"
  --character-coverage "${CHAR_COVERAGE}"
  --model-type "${MODEL_TYPE}"
  --language "${LANGUAGE}"
)

if [[ "${BYTE_FALLBACK}" == "1" ]]; then
  CMD+=(--byte-fallback)
fi

CMD+=("${USER_SYMBOL_FLAGS[@]}")

echo "[KO-STEP1] manifest=${MANIFEST}"
echo "[KO-STEP1] output-prefix=${OUTPUT_PREFIX}"
echo "[KO-STEP1] vocab-size=${VOCAB_SIZE}"

if [[ "${TIMEOUT_SECS}" -gt 0 ]]; then
  timeout "${TIMEOUT_SECS}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

RAW_MODEL="${RAW_PREFIX}.model"
RAW_VOCAB="${RAW_PREFIX}.vocab"
FINAL_MODEL="${OUTPUT_PREFIX}.model"
FINAL_VOCAB="${OUTPUT_PREFIX}.vocab"

if (( RETAIN_COUNT > 0 )); then
  echo "[KO-STEP1] rebuilding tokenizer with base retention (retain=${RETAIN_COUNT}, target=${TARGET_SIZE})"
  REBUILD_CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/tokenizer/rebuild_bpe_with_base.py"
    --base-model "${BASE_MODEL_PATH}"
    --new-model "${RAW_MODEL}"
    --output-model "${FINAL_MODEL}"
    --output-vocab "${FINAL_VOCAB}"
    --retain-count "${RETAIN_COUNT}"
    --target-size "${TARGET_SIZE}"
  )
  "${REBUILD_CMD[@]}"
  if [[ "${KEEP_RAW}" != "1" ]]; then
    rm -f "${RAW_MODEL}" "${RAW_VOCAB}"
  fi
  echo "[KO-STEP1] tokenizer training and rebuild complete -> ${FINAL_MODEL}"
else
  echo "[KO-STEP1] retain_count=0, using raw tokenizer as final output."
  mv -f "${RAW_MODEL}" "${FINAL_MODEL}"
  mv -f "${RAW_VOCAB}" "${FINAL_VOCAB}"
  echo "[KO-STEP1] tokenizer training complete -> ${FINAL_MODEL}"
fi
