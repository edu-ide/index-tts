#!/usr/bin/env bash
# Convenience wrapper: resume fused AdamW from latest_full.pth (keeps optimizer/scheduler by default).
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer full checkpoint to keep optimizer/scheduler; default to latest_full.pth
DEFAULT_FULL="/mnt/sda1/models/index-tts-ko/checkpoints/latest_full.pth"
DEFAULT_LIGHT="/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth"
if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "${DEFAULT_FULL}" ]]; then
    CKPT="${DEFAULT_FULL}"
  else
    CKPT="${DEFAULT_LIGHT}"
  fi
fi
# Default: resume optimizer/scheduler state
FRESH_OPT="${FRESH_OPT:-0}"

# Training defaults (override as needed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LR="${LR:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-4}"  # Batch 8 * 4 = 32
VAL_INTERVAL="${VAL_INTERVAL:-100}" # Validate every 100 steps
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"
NUM_WORKERS="${NUM_WORKERS:-32}"

# Strip accidental surrounding quotes
CKPT="${CKPT%\"}"
CKPT="${CKPT#\"}"
CKPT="${CKPT%\'}"
CKPT="${CKPT#\'}"

env \
  CKPT="${CKPT}" \
  FRESH_OPT="${FRESH_OPT}" \
  LR="${LR}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACC="${GRAD_ACC}" \
  VAL_INTERVAL="${VAL_INTERVAL}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  TOKENIZER_MODEL="${TOKENIZER_MODEL}" \
  "${SCRIPT_DIR}/resume_ko_fused_adamw.sh"
