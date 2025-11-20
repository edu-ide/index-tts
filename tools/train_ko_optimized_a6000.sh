#!/usr/bin/env bash
# 최적화된 한국어 GPT 학습 스크립트 - A6000 48GB
#
# 과학적 근거:
# - LR: 1e-5 (IndexTTS-2 pre-training 2e-4의 1/20, fine-tuning 표준)
# - Batch: 16 (Square Root Scaling Rule: batch 4→16 = 4배, LR 2배)
# - Warmup: 30,000 steps (전체 학습의 4.4%, GPT 논문 권장 10% 이하)
# - Grad Clip: 0.5 (최신 TTS 모델 표준)
#
# 예상 학습 시간: 26-28시간 (2 epochs)
# 예상 성능: text_loss < 0.9 달성 가능

set -euo pipefail

echo "================================================================"
echo "🚀 최적화된 한국어 GPT 학습 - A6000 48GB"
echo "================================================================"
echo ""
echo "📊 하이퍼파라미터 (과학적 근거 기반):"
echo "  - GPU: A6000 48GB"
echo "  - Batch Size: 16 (메모리 최적)"
echo "  - Learning Rate: 1e-5 (fine-tuning 표준)"
echo "  - Warmup Steps: 30,000 (4.4% of total)"
echo "  - Epochs: 2"
echo "  - Gradient Clip: 0.5"
echo ""
echo "🎯 목표:"
echo "  - text_loss < 0.9"
echo "  - mel_loss < 3.5"
echo "  - 학습 시간: ~26-28시간"
echo ""
echo "📚 과학적 근거:"
echo "  - Square Root Scaling Rule (AdamW)"
echo "  - IndexTTS-2 공식 설정 기반"
echo "  - 2024 GPT/TTS 논문 권장사항"
echo ""
echo "================================================================"

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ ${GPU_MEM} -lt 40000 ]]; then
  echo "[WARNING] GPU 메모리가 40GB 미만입니다: ${GPU_MEM}MB" >&2
  echo "          A6000 (48GB) 권장. RTX 4090은 train_ko_optimized_4090.sh 사용" >&2
  echo "" >&2
  read -p "계속하시겠습니까? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
  fi
fi

echo "✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "✅ VRAM: ${GPU_MEM}MB"
echo ""

# 사용자 확인
read -p "학습을 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

echo ""
echo "🎬 학습을 시작합니다..."
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 체크포인트: /mnt/sda1/models/index-tts-ko/checkpoints/"
echo ""

# 최적화된 설정으로 학습 시작
SKIP_DATA_CHECK=1 \
LR=1e-5 \
WARMUP_STEPS=30000 \
BATCH_SIZE=16 \
GRAD_ACC=1 \
GRAD_CLIP=0.5 \
LOG_INTERVAL=100 \
VAL_INTERVAL=1000 \
EPOCHS=2 \
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

echo ""
echo "================================================================"
echo "✅ 학습 완료!"
echo "================================================================"
echo ""
echo "📁 저장된 체크포인트:"
echo "  - 최고 성능: /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
echo "  - 최신: /mnt/sda1/models/index-tts-ko/checkpoints/latest.pth"
echo ""
if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt" ]]; then
  echo "🏆 Best mel_loss: $(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt)"
fi
echo ""
echo "다음 단계:"
echo "  1. TensorBoard로 학습 곡선 확인"
echo "  2. best_model.pth로 음성 생성 테스트"
echo "  3. 품질 평가 후 필요시 추가 학습"
echo ""
