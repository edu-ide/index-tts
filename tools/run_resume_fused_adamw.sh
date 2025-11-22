#!/usr/bin/env bash
# Convenience wrapper: resume fused AdamW from latest_full.pth (keeps optimizer/scheduler by default).
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer full checkpoint to keep optimizer/scheduler; fall back to latest.pth only if full is missing
DEFAULT_FULL="/mnt/sda1/models/index-tts-ko/checkpoints/latest_full.pth"
DEFAULT_LIGHT="/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth"

if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "${DEFAULT_FULL}" ]]; then
    CKPT="${DEFAULT_FULL}"
  else
    CKPT="${DEFAULT_LIGHT}"
  fi
fi
FRESH_OPT="${FRESH_OPT:-0}"   # 0: resume optimizer/scheduler state

# Strip accidental surrounding quotes
CKPT="${CKPT%\"}"
CKPT="${CKPT#\"}"
CKPT="${CKPT%\'}"
CKPT="${CKPT#\'}"

env \
  CKPT="${CKPT}" \
  FRESH_OPT="${FRESH_OPT}" \
  "${SCRIPT_DIR}/resume_ko_fused_adamw.sh"
