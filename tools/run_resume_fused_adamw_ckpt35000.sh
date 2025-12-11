#!/usr/bin/env bash
# Convenience wrapper: resume fused AdamW from model_step35000.pth with existing optimizer/scheduler (FRESH_OPT=0).
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:-/mnt/sda1/models/index-tts-ko/checkpoints/model_step35000.pth}"
FRESH_OPT="${FRESH_OPT:-0}"   # keep optimizer/scheduler state

# Strip accidental surrounding quotes in CKPT
CKPT="${CKPT%\"}"
CKPT="${CKPT#\"}"
CKPT="${CKPT%\'}"
CKPT="${CKPT#\'}"

env \
  CKPT="${CKPT}" \
  FRESH_OPT="${FRESH_OPT}" \
  "${SCRIPT_DIR}/resume_ko_fused_adamw.sh"
