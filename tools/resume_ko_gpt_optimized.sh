#!/usr/bin/env bash
# 최적화된 한국어 GPT 재개 스크립트 (Phase 1/Phase 2 전략)
#
# 2단계 학습 전략:
# Phase 1 (0→240k): GRAD_ACC=1, LR=2e-5 (빠른 수렴)
# Phase 2 (240k→끝): GRAD_ACC=8, LR=5e-6 (안정화)
#
# 이 스크립트는 checkpoint step에 따라 자동으로 Phase 선택:
# - Step < 240k: Phase 1 설정 사용
# - Step >= 240k: Phase 2 설정 사용
#
# Phase 1: 빠른 수렴
# - LR: 2e-5 (공격적)
# - GRAD_ACC: 1 (빠른 학습)
# - MAX_STEPS: 240000
#
# Phase 2: 안정화
# - LR: 5e-6 (안정적)
# - GRAD_ACC: 8 (batch 크기 증가)
# - EPOCHS: 3

set -euo pipefail

echo "================================================================"
echo "🔄 최적화된 한국어 GPT 재개 - Transformer 논문 기반"
echo "================================================================"
echo ""

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
RESUME_PATH="${RESUME_PATH:-${CHECKPOINT_DIR}/latest.pth}"
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
RAW_MANIFEST="${RAW_MANIFEST:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"

SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-1}"

# 체크포인트 검증
if [[ ! -f "${RESUME_PATH}" ]]; then
  echo "[ERROR] 체크포인트를 찾을 수 없습니다: ${RESUME_PATH}" >&2
  exit 1
fi

CKPT_SIZE=$(stat -c%s "${RESUME_PATH}" 2>/dev/null || stat -f%z "${RESUME_PATH}" 2>/dev/null)
CKPT_SIZE_GB=$((CKPT_SIZE / 1024 / 1024 / 1024))

echo "📁 체크포인트 정보:"
echo "  - 경로: ${RESUME_PATH}"
echo "  - 크기: ${CKPT_SIZE_GB}GB"
echo "  - 수정: $(ls -lh "${RESUME_PATH}" | awk '{print $6, $7, $8}')"
echo ""

# GPU 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "✅ GPU: ${GPU_NAME}"
echo "✅ VRAM: ${GPU_MEM}MB"
echo ""

# 데이터 무결성 체크 (옵션)
if [[ "${SKIP_DATA_CHECK}" != "1" ]]; then
  echo "🔍 데이터 무결성 검사 중..." >&2
  DATASET_DIR="${DATASET_DIR}" RAW_MANIFEST="${RAW_MANIFEST}" \
    "${SCRIPT_DIR}/ko_step2_fix_broken.sh" --scan-empty --scan-missing
else
  echo "⏭️  데이터 검사 생략 (SKIP_DATA_CHECK=1)" >&2
fi

# NUM_WORKERS 자동 설정 (최대 성능 - CPU 코어 수 사용)
if [[ -z "${NUM_WORKERS:-}" ]]; then
  CPU_CORES=$(nproc)
  export NUM_WORKERS=${CPU_CORES}
  echo "⚙️  NUM_WORKERS 자동 설정: ${NUM_WORKERS} (CPU 코어 수)" >&2
else
  echo "⚙️  NUM_WORKERS 사용자 설정: ${NUM_WORKERS}" >&2
fi

# 체크포인트에서 step 추출
CKPT_STEP=0
if [[ "${RESUME_PATH}" =~ model_step([0-9]+)\.pth ]]; then
  CKPT_STEP="${BASH_MATCH[1]}"
elif python3 -c "import torch; print(torch.load('${RESUME_PATH}', map_location='cpu').get('step', 0))" 2>/dev/null | grep -q '^[0-9]\+$'; then
  CKPT_STEP=$(python3 -c "import torch; print(torch.load('${RESUME_PATH}', map_location='cpu').get('step', 0))")
fi

echo "📊 체크포인트 Step: ${CKPT_STEP}"
echo ""

# Phase 자동 선택
PHASE_THRESHOLD=240000
if [[ ${CKPT_STEP} -lt ${PHASE_THRESHOLD} ]]; then
  PHASE="Phase 1"
  LR_VAL=2e-5
  GRAD_ACC_VAL=1
  GRAD_CLIP_VAL=1.0
  WARMUP_VAL=1000
  MAX_STEPS_VAL=240000
  EPOCHS_VAL=999
  echo "🚀 Phase 1: 빠른 수렴 (Step 0 → 240k)"
  echo "  - LR: ${LR_VAL} (공격적)"
  echo "  - GRAD_ACC: ${GRAD_ACC_VAL} (빠른 학습)"
  echo "  - GRAD_CLIP: ${GRAD_CLIP_VAL} (LR에 맞춘 안전장치)"
  echo "  - Warmup: ${WARMUP_VAL}"
  echo "  - Max Steps: ${MAX_STEPS_VAL}"
else
  PHASE="Phase 2"
  LR_VAL=5e-6
  GRAD_ACC_VAL=8
  GRAD_CLIP_VAL=0.5
  WARMUP_VAL=1000
  MAX_STEPS_VAL=0
  EPOCHS_VAL=3
  echo "🎯 Phase 2: 안정화 (Step 240k → 끝)"
  echo "  - LR: ${LR_VAL} (안정적)"
  echo "  - GRAD_ACC: ${GRAD_ACC_VAL} (batch 크기 증가)"
  echo "  - GRAD_CLIP: ${GRAD_CLIP_VAL} (보수적 안전장치)"
  echo "  - Warmup: ${WARMUP_VAL}"
  echo "  - Epochs: ${EPOCHS_VAL}"
fi

echo ""
echo "⚙️  공통 설정:"
echo "  - Batch Size: 8 (RTX 4090 검증됨)"
echo ""

# 사용자 확인 스킵 (자동 실행)
echo "자동으로 ${PHASE} 학습을 재개합니다..."

echo ""
echo "🎬 학습을 재개합니다..."
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 체크포인트: ${CHECKPOINT_DIR}/"
echo ""

# Phase별 최적화 설정으로 재개
SKIP_DATA_CHECK=1 \
LR=${LR_VAL} \
WARMUP_STEPS=${WARMUP_VAL} \
BATCH_SIZE=8 \
GRAD_ACC=${GRAD_ACC_VAL} \
GRAD_CLIP=${GRAD_CLIP_VAL} \
LOG_INTERVAL=100 \
VAL_INTERVAL=10000 \
MAX_STEPS=${MAX_STEPS_VAL} \
EPOCHS=${EPOCHS_VAL} \
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth" \
RESUME="${RESUME_PATH}" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh" "$@"

echo ""
echo "================================================================"
echo "✅ 학습 재개 완료!"
echo "================================================================"
echo ""
echo "📁 저장된 체크포인트:"
echo "  - 최고 성능: ${CHECKPOINT_DIR}/best_model.pth"
echo "  - 최신: ${CHECKPOINT_DIR}/latest.pth"
echo ""
if [[ -f "${CHECKPOINT_DIR}/best_loss.txt" ]]; then
  echo "🎯 Best text_loss: $(cat "${CHECKPOINT_DIR}/best_loss.txt")"
fi
echo ""
echo "다음 단계:"
echo "  1. TensorBoard로 학습 곡선 확인"
echo "  2. ${PHASE} 완료 후 다음 Phase 진행 (해당시)"
echo "  3. best_model.pth로 음성 생성 테스트"
echo ""
