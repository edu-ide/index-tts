#!/usr/bin/env bash
# Phase 2: Phase 1 성공 후 계속 학습
# 조건: Phase 1에서 loss가 감소 추세를 보인 경우

set -euo pipefail

echo "================================================================"
echo "Phase 2: 계속 학습 (LR 점진 증가)"
echo "================================================================"
echo ""
echo "Phase 1 결과:"
echo "  - loss가 감소 추세를 보임"
echo "  - 초저 LR(1e-6)로 모델 회복 가능 확인"
echo ""
echo "Phase 2 전략:"
echo "  - LR: 2e-6 (Phase 1의 2배, 원래의 1/10)"
echo "  - Warmup: 5000 step"
echo "  - Max steps: 제한 없음 (1 epoch 완료 목표)"
echo "  - 자동 저장: 1000 step마다"
echo ""
echo "목표:"
echo "  - 최소 1 epoch 완료 (548만 샘플 전부)"
echo "  - text_loss < 0.9 달성"
echo "  - mel_loss < 3.5 달성"
echo ""
echo "================================================================"

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="/mnt/sda1/models/index-tts-ko/checkpoints"

# 최신 체크포인트 찾기
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pth"

if [[ ! -f "${LATEST_CHECKPOINT}" ]]; then
  echo "[ERROR] 체크포인트를 찾을 수 없습니다: ${LATEST_CHECKPOINT}" >&2
  exit 1
fi

echo "체크포인트: ${LATEST_CHECKPOINT}"
echo "TensorBoard: http://localhost:6006"
echo ""

# 사용자 확인
read -p "Phase 2를 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

echo "학습을 시작합니다..."
echo ""

# Phase 2 학습 시작
SKIP_DATA_CHECK=1 \
LR=2e-6 \
WARMUP_STEPS=5000 \
MAX_STEPS=0 \
GRAD_CLIP=0.5 \
BATCH_SIZE=4 \
LOG_INTERVAL=100 \
VAL_INTERVAL=1000 \
EPOCHS=1 \
RESUME="${LATEST_CHECKPOINT}" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

echo ""
echo "================================================================"
echo "Phase 2 완료!"
echo "================================================================"
