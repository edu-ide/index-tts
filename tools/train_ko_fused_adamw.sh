#!/usr/bin/env bash
# Train with fused AdamW (no scheduler by default)
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 기본값: IndexTTS-2 gpt.pth (완전 새 시작). 재개하려면 CKPT=/path/to/latest.pth 로 override.
CKPT="${CKPT:-/mnt/sda1/models/IndexTTS-2/gpt.pth}"

OPTIMIZER="${OPTIMIZER:-adamw}"
SCHEDULER="${SCHEDULER:-wsd}"           # 논문 기반 WSD 추천
LR="${LR:-3e-5}"                        # 8x1 배치에 맞춘 보수적 LR
GRAD_CLIP="${GRAD_CLIP:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-1}"               # 실효 배치 8 (요청: 8x1)
AMP="${AMP:-1}"
NUM_WORKERS="${NUM_WORKERS:-12}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-10000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"    # 배치 축소 시 1k 워밍업
MAX_STEPS="${MAX_STEPS:-0}"             # 0이면 epoch 기반에서 자동 계산
WSD_STABLE_RATIO="${WSD_STABLE_RATIO:-0.9}"
WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO:-0.05}"
# Duration conditioning: 논문 방식 length tie
DURATION_CONDITIONING="${DURATION_CONDITIONING:-length}"

echo "Using base checkpoint: ${CKPT}"
echo "  (override with CKPT=/path/to/checkpoint.pth)"

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
  AMP="${AMP}"
  DURATION_CONDITIONING="${DURATION_CONDITIONING}"
)

env "${CMD_ENV[@]}" "${SCRIPT_DIR}/ko_step4_train_gpt.sh" --no-aim
