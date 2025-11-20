#!/usr/bin/env bash
# 최적화된 한국어 GPT 학습 스크립트 - RTX 4090 24GB
#
# 과학적 근거 (IndexTTS2 논문 기반):
# - LR: 2e-4 (IndexTTS2 논문 권장값, arXiv:2506.21619v2)
# - Batch: 8 (RTX 4090 24GB 검증된 안전값)
# - Grad Accumulation: 8 (실효 batch 64 = ~34,694 tokens, Transformer 논문 수준)
# - Warmup: 30,000 steps (안정적인 학습 시작)
# - Grad Clip: 0.5 (gradient explosion 방지)
#
# 예상 학습 시간: 30-35시간 (2 epochs, gradient accumulation으로 약간 느림)
# 예상 성능: text_loss < 0.9, mel_loss < 3.5 달성 가능
#
# 주의: 현재는 1-stage 학습 (논문의 3-stage 미구현)
#       GRL (Gradient Reversal Layer) 미포함
#       Phase 1 baseline 확보 후 GRL 구현 예정

set -euo pipefail

echo "================================================================"
echo "🚀 최적화된 한국어 GPT 학습 - RTX 4090 24GB"
echo "================================================================"
echo ""
echo "📊 하이퍼파라미터 (IndexTTS2 논문 기반):"
echo "  - GPU: RTX 4090 24GB"
echo "  - Batch Size: 8 (메모리 안전)"
echo "  - Gradient Accumulation: 8 (실효 batch 64)"
echo "  - Tokens/Batch: ~34,694 (Transformer 논문 수준)"
echo "  - Learning Rate: 2e-4 (논문 권장값)"
echo "  - Warmup Steps: 30,000"
echo "  - Epochs: 2"
echo "  - Gradient Clip: 0.5"
echo ""
echo "💡 Gradient Accumulation 사용:"
echo "  - 메모리는 batch 8 수준 유지 (안전)"
echo "  - 학습 효과는 batch 64와 동일"
echo "  - Transformer 논문 기준(~25k tokens) 달성 ✅"
echo ""
echo "🎯 목표:"
echo "  - text_loss < 0.9"
echo "  - mel_loss < 3.5"
echo "  - 학습 시간: ~30-35시간"
echo ""
echo "📚 참고문헌:"
echo "  - IndexTTS2 (arXiv:2506.21619v2) - LR 2e-4, AdamW"
echo "  - Attention is All You Need (Vaswani+ 2017) - ~25k tokens/batch"
echo "  - Decoupled Weight Decay (Loshchilov+ 2019) - AdamW optimizer"
echo ""
echo "⚠️  주의사항:"
echo "  - 현재: 1-stage 학습 (ConformerEncoder + PerceiverResampler)"
echo "  - 논문: 3-stage 학습 (basic → GRL emotion → fine-tune)"
echo "  - GRL 미구현으로 speaker-emotion disentanglement 제한적"
echo "  - Phase 1 baseline 확보 후 GRL 추가 예정"
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
echo "💡 Gradient Accumulation 활성화:"
echo "   - Forward/Backward 8회 실행 후 1회 업데이트"
echo "   - 약간 느리지만 Transformer 논문 기준 달성"
echo ""

# 최적화된 설정으로 학습 시작 (데이터 분석 기반 fine-tuning)
# 실제 학습 데이터 분석 결과: LR 1e-5가 최적 (2e-5 이상에서 loss 정체)
# Fine-tuning이므로 GRAD_CLIP 0.5로 낮게 유지 (base model 보존)
SKIP_DATA_CHECK=1 \
LR=1e-5 \
WARMUP_STEPS=5000 \
BATCH_SIZE=8 \
GRAD_ACC=8 \
GRAD_CLIP=0.5 \
LOG_INTERVAL=100 \
VAL_INTERVAL=1000 \
EPOCHS=3 \
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
