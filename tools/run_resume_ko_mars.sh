#!/usr/bin/env bash
# Convenience wrapper: resume MARS with preset hyperparams (no manual env typing).
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults (edit here if 필요) ----
# [Recovery Plan: Restart from Base]
# Step 351k was already overfitted.
# We restart from the clean Base Model (IndexTTS-2/gpt.pth).
CKPT="${CKPT:-/mnt/sda1/models/IndexTTS-2/gpt.pth}"
FRESH_OPT="${FRESH_OPT:-1}"          # 1: Must reset optimizer for Base Model
LR="${LR:-5e-6}"                     # Safe LR (1/4 of previous failed run)
MAX_STEPS="${MAX_STEPS:-0}"          # Unlimited
WSD_WARMUP="${WSD_WARMUP:-2000}"     # Warmup to adapt to MARS
WSD_STABLE_RATIO="${WSD_STABLE_RATIO:-0.9}" # Long stable phase
WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO:-0.0}"
OPTIMIZER="${OPTIMIZER:-mars}"
SCHEDULER="${SCHEDULER:-wsd}"
# Allow disabling Aim with NO_AIM=1
NO_AIM="${NO_AIM:-0}"
# CPU checkpoint load (1 to enable)
CPU_CKPT_LOAD="${CPU_CKPT_LOAD:-1}"
# 나머지는 resume_ko_mars.sh 기본값(배치8/acc1/grad_clip0.5 등) 사용
# ------------------------------------

env \
  CKPT="${CKPT}" \
  FRESH_OPT="${FRESH_OPT}" \
  LR="${LR}" \
  MAX_STEPS="${MAX_STEPS}" \
  WSD_WARMUP="${WSD_WARMUP}" \
  WSD_STABLE_RATIO="${WSD_STABLE_RATIO}" \
  WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO}" \
  OPTIMIZER="${OPTIMIZER}" \
  SCHEDULER="${SCHEDULER}" \
  NO_AIM="${NO_AIM}" \
  CPU_CKPT_LOAD="${CPU_CKPT_LOAD}" \
  "${SCRIPT_DIR}/resume_ko_mars.sh"
