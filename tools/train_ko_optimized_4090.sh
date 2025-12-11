#!/usr/bin/env bash
# 최적화된 한국어 GPT 학습 스크립트 - RTX 4090 24GB (Fine-tuning)
# Phase 1: 빠른 수렴 (Step 0 → 240k)
#
# 과학적 근거 (이전 학습 데이터 분석 기반):
# - LR: 2e-5 (공격적, 빠른 수렴)
# - Batch: 8 (RTX 4090 24GB 검증된 안전값)
# - Grad Accumulation: 1 (빠른 학습, batch 8)
# - Warmup: 1,000 steps
# - Grad Clip: 1.0 (LR에 맞춘 안전장치, 이전 대비 수렴 속도 향상)
#
# 이전 학습 결과 (GRAD_ACC=1, LR=2e-5):
#   - Step 247800: Loss=0.9648 (최저점 달성!)
#   - Step 250k 이후: Overfitting 시작
#
# Phase 1 목표: Step 240k에서 Loss < 1.0 달성
# Phase 2 전환: Step 240k에서 GRAD_ACC=8, LR=5e-6으로 안정화
#
# 예상 학습 시간: ~20시간 (240k steps, GRAD_ACC=1로 빠름)
#
# 주의: Cross-lingual fine-tuning (English base → Korean)
#       Phase 1: 빠른 수렴 (현재)
#       Phase 2: 안정화 및 일반화 (Step 240k 이후)

set -euo pipefail

echo "================================================================"
echo "🚀 한국어 GPT Fine-tuning - RTX 4090 24GB"
echo "📍 Phase 1: 빠른 수렴 (Step 0 → 240k)"
echo "================================================================"
echo ""
echo "📊 Phase 1 하이퍼파라미터:"
echo "  - GPU: RTX 4090 24GB"
echo "  - Batch Size: 8"
echo "  - Gradient Accumulation: 1 (빠른 학습)"
echo "  - Learning Rate: 2e-5 (공격적)"
echo "  - Warmup Steps: 1,000"
echo "  - Max Steps: 240,000"
echo "  - Gradient Clip: 1.0 (LR에 맞춘 안전장치)"
echo ""
echo "💡 2단계 학습 전략:"
echo "  - Phase 1 (0→240k): GRAD_ACC=1, LR=2e-5 (빠른 수렴)"
echo "  - Phase 2 (240k→끝): GRAD_ACC=8, LR=5e-6 (안정화)"
echo "  - 이전 결과: Step 247k에서 Loss=0.96 달성"
echo "  - 목표: Overfitting 방지하며 최저점 도달"
echo ""
echo "🎯 목표:"
echo "  - text_loss < 0.9"
echo "  - mel_loss < 3.5"
echo "  - 학습 시간: ~45시간 (3 epochs)"
echo ""
echo "📚 참고문헌:"
echo "  - IndexTTS2 (arXiv:2506.21619v2) - Base architecture"
echo "  - 2024-2025 Fine-tuning research - LR scheduling"
echo "  - Warmup-Stable-Decay (2024) - Modern LR schedule"
echo ""
echo "⚠️  주의사항:"
echo "  - Cross-lingual fine-tuning (not from scratch)"
echo "  - Phase 1: Baseline 확보"
echo "  - Phase 2: GRL emotion disentanglement 예정"
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
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "✅ GPU: ${GPU_NAME}"
echo "✅ VRAM: ${GPU_MEM}MB"
echo ""

if [[ ${GPU_MEM} -gt 40000 ]]; then
  echo "[INFO] 48GB+ GPU 감지되었습니다." >&2
  echo "      더 빠른 학습을 위해 train_ko_optimized_a6000.sh 사용 권장" >&2
  echo "" >&2
fi

# 사용자 확인 스킵 (자동 실행)
echo "자동으로 학습을 시작합니다..."

echo ""
echo "🎬 Phase 1 학습을 시작합니다..."
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 체크포인트: /mnt/sda1/models/index-tts-ko/checkpoints/"
echo ""
echo "⚡ Phase 1 전략:"
echo "   - GRAD_ACC=1로 빠른 학습"
echo "   - Step 240k에서 자동 종료"
echo "   - 이후 Phase 2로 수동 전환 필요"
echo "   - Phase 2: GRAD_ACC=8, LR=5e-6"
echo ""

# Phase 1: 빠른 수렴 (Step 0 → 240k)
# 이전 학습 데이터 기반 전략:
#   - GRAD_ACC=1로 빠르게 수렴
#   - Step 247k에서 Loss=0.96 달성
#   - Step 240k에서 멈추고 Phase 2로 전환 예정
SKIP_DATA_CHECK=1 \
LR=2e-5 \
WARMUP_STEPS=1000 \
BATCH_SIZE=8 \
GRAD_ACC=1 \
GRAD_CLIP=1.0 \
LOG_INTERVAL=100 \
VAL_INTERVAL=10000 \
MAX_STEPS=240000 \
EPOCHS=999 \
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

echo ""
echo "================================================================"
echo "✅ Phase 1 학습 완료!"
echo "================================================================"
echo ""
echo "📁 저장된 체크포인트:"
echo "  - 최고 성능: /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
echo "  - 최신: /mnt/sda1/models/index-tts-ko/checkpoints/latest.pth"
echo ""
if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt" ]]; then
  echo "🎯 Best text_loss: $(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt)"
fi
echo ""
echo "다음 단계:"
echo "  1. TensorBoard로 학습 곡선 확인"
echo "  2. Phase 2 시작: tools/train_ko_optimized_4090_phase2.sh"
echo "  3. best_model.pth로 음성 생성 테스트"
echo ""
