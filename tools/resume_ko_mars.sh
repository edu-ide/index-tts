#!/usr/bin/env bash
# Resume training with MARS optimizer.
# If FRESH_OPT=1, load only model weights (reset optimizer/scheduler) so new LR/WSD params apply.
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
CKPT="${CKPT:-/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth}"
FRESH_OPT="${FRESH_OPT:-0}"
OPTIMIZER="${OPTIMIZER:-mars}"
SCHEDULER="${SCHEDULER:-wsd}"
LR="${LR:-3e-5}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-1}"
AMP="${AMP:-1}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-0}"
WSD_STABLE_RATIO="${WSD_STABLE_RATIO:-0.9}"
WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO:-0.05}"
DURATION_CONDITIONING="${DURATION_CONDITIONING:-length}"
NO_AIM="${NO_AIM:-0}"
# Force CPU checkpoint load? (1 to enable)
CPU_CKPT_LOAD="${CPU_CKPT_LOAD:-1}"

# Allow alias WSD_WARMUP
if [[ -n "${WSD_WARMUP:-}" ]]; then
  WARMUP_STEPS="${WSD_WARMUP}"
fi

echo "Resuming from checkpoint: ${CKPT}"
if [[ "${FRESH_OPT}" == "1" ]]; then
  echo "  -> FRESH_OPT=1: optimizer/scheduler will NOT be resumed (model weights only)."
else
  echo "  -> FRESH_OPT=0: optimizer/scheduler will be resumed."
fi

CMD_ENV=(
  OPTIMIZER="${OPTIMIZER}"
  SCHEDULER="${SCHEDULER}"
  LR="${LR}"
  GRAD_CLIP="${GRAD_CLIP}"
  BATCH_SIZE="${BATCH_SIZE}"
  GRAD_ACC="${GRAD_ACC}"
  NUM_WORKERS="${NUM_WORKERS}"
  LOG_INTERVAL="${LOG_INTERVAL}"
  VAL_INTERVAL="${VAL_INTERVAL}"
  WEIGHT_DECAY="${WEIGHT_DECAY}"
  WARMUP_STEPS="${WARMUP_STEPS}"
  MAX_STEPS="${MAX_STEPS}"
  WSD_STABLE_RATIO="${WSD_STABLE_RATIO}"
  WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO}"
  BASE_CHECKPOINT="${CKPT}"
  RESUME="$( [[ \"${FRESH_OPT}\" == \"1\" ]] && echo \"\" || echo \"${CKPT}\" )"
  AMP="${AMP}"
  DURATION_CONDITIONING="${DURATION_CONDITIONING}"
  CPU_CKPT_LOAD="${CPU_CKPT_LOAD}"
)

EXTRA_ARGS=()
if [[ "${NO_AIM}" == "1" ]]; then
  EXTRA_ARGS+=(--no-aim)
fi
if [[ "${CPU_CKPT_LOAD}" == "1" ]]; then
  EXTRA_ARGS+=(--cpu-ckpt-load)
fi

env "${CMD_ENV[@]}" "${SCRIPT_DIR}/ko_step4_train_gpt.sh" "${EXTRA_ARGS[@]}"
