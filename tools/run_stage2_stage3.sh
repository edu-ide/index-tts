#!/usr/bin/env bash
# Convenience: run Stage 2 (GRL disentanglement) followed by Stage 3 (conditioner freeze) sequentially.
# Uses existing train_ko_stage2.sh / train_ko_stage3.sh with sane defaults; override via env vars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "🚦 IndexTTS2 Stage 2 ➜ Stage 3 자동 실행"
echo "================================================================"
echo ""
echo "환경변수로 조정 가능:"
echo "  STAGE1_CHECKPOINT, SPEAKER_MAPPING, DATASET_DIR, CHECKPOINT_DIR_STAGE2, CHECKPOINT_DIR_STAGE3"
echo "  BATCH_SIZE, GRAD_ACC, LR, WARMUP_STEPS, EPOCHS, GRAD_CLIP, GRL_LAMBDA, SPEAKER_LOSS_WEIGHT"
echo ""

read -p "Stage 2 → Stage 3 순차 실행을 시작할까요? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

CHECKPOINT_DIR_STAGE2="${CHECKPOINT_DIR_STAGE2:-/mnt/sda1/models/index-tts-ko/stage2}"
CHECKPOINT_DIR_STAGE3="${CHECKPOINT_DIR_STAGE3:-/mnt/sda1/models/index-tts-ko/stage3}"

# Stage 2
ENV_STAGE2=(
  "CHECKPOINT_DIR=${CHECKPOINT_DIR_STAGE2}"
)
echo ">>> Stage 2 시작..."
(
  cd "${SCRIPT_DIR}"
  env "${ENV_STAGE2[@]}" ./train_ko_stage2.sh
)

# Stage 3
ENV_STAGE3=(
  "CHECKPOINT_DIR=${CHECKPOINT_DIR_STAGE3}"
  "BASE_CHECKPOINT=${CHECKPOINT_DIR_STAGE2}/best_model.pth"
)
echo ">>> Stage 3 시작..."
(
  cd "${SCRIPT_DIR}"
  env "${ENV_STAGE3[@]}" ./train_ko_stage3.sh
)

echo "✅ Stage 2 → Stage 3 완료"
