#!/usr/bin/env bash
# Phase 1 실패 시: Base 모델부터 재학습
# 조건: Phase 1에서도 loss가 계속 증가한 경우

set -euo pipefail

echo "================================================================"
echo "Base 모델부터 재학습"
echo "================================================================"
echo ""
echo "Phase 1 결과:"
echo "  - 초저 LR에서도 loss 증가"
echo "  - 기존 체크포인트 사용 불가 판단"
echo ""
echo "재학습 전략:"
echo "  - Base 모델: /mnt/sda1/models/IndexTTS-2/gpt.pth"
echo "  - LR: 5e-6 (보수적으로 시작)"
echo "  - Warmup: 10000 step (충분한 warmup)"
echo "  - Batch size: 8 (기존 4의 2배)"
echo "  - Gradient clip: 0.5"
echo ""
echo "예상 시간: 50-70시간 (1 epoch 기준)"
echo ""
echo "================================================================"

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth"
BACKUP_DIR="/mnt/sda1/models/index-tts-ko/checkpoints_backup_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "[ERROR] Base 체크포인트를 찾을 수 없습니다: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

# 기존 체크포인트 백업 확인
echo "⚠️  경고: 기존 학습 데이터를 백업합니다."
echo "백업 위치: ${BACKUP_DIR}"
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

# 백업
echo "기존 체크포인트 백업 중..."
mkdir -p "${BACKUP_DIR}"
cp -r /mnt/sda1/models/index-tts-ko/checkpoints/* "${BACKUP_DIR}/" || true
echo "백업 완료: ${BACKUP_DIR}"
echo ""

echo "학습을 시작합니다..."
echo "TensorBoard: http://localhost:6006"
echo ""

# 재학습 시작 (RESUME 플래그 없음)
LR=5e-6 \
WARMUP_STEPS=10000 \
MAX_STEPS=0 \
GRAD_CLIP=0.5 \
BATCH_SIZE=8 \
LOG_INTERVAL=100 \
VAL_INTERVAL=1000 \
EPOCHS=2 \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

echo ""
echo "================================================================"
echo "재학습 완료!"
echo "================================================================"
