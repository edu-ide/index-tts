#!/usr/bin/env bash
# IndexTTS2 Stage 3: Fine-tuning with Frozen Feature Conditioners
#
# Stage 3 설정 (IndexTTS2 논문 기반):
# - Speaker perceiver conditioner: FROZEN (Stage 1에서 학습됨)
# - Emotion perceiver conditioner: FROZEN (Stage 2에서 학습됨)
# - GPT backbone: TRAINABLE
# - GRL: DISABLED (Stage 2에서만 사용)
# - Learning Rate: 1e-4 (Stage 1/2보다 낮음)
# - 전체 데이터셋 사용
#
# 목적:
#   - Stage 1, 2에서 학습된 feature를 고정
#   - GPT의 생성 품질만 개선
#   - Overfitting 방지
#
# 참고문헌:
# - IndexTTS2 (arXiv:2506.21619v2)

set -euo pipefail

echo "================================================================"
echo "🎯 IndexTTS2 Stage 3: Fine-tuning (Frozen Conditioners)"
echo "================================================================"
echo ""

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ============================================================
# Stage 3 Configuration
# ============================================================

# Paths
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/sda1/models/index-tts-ko/stage3}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth}"

# Model config
CONFIG="${CONFIG:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer.model}"

# Training hyperparameters (Stage 3 specific)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-1e-4}"  # Lower than Stage 1/2
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EPOCHS="${EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"

# Logging
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"

# ============================================================
# Validation
# ============================================================

echo "📂 Paths:"
echo "  - Dataset: ${DATASET_DIR}"
echo "  - Output: ${CHECKPOINT_DIR}"
echo "  - Stage 2 checkpoint: ${STAGE2_CHECKPOINT}"
echo ""

# Check Stage 2 checkpoint
if [[ ! -f "${STAGE2_CHECKPOINT}" ]]; then
  echo "[ERROR] Stage 2 checkpoint not found: ${STAGE2_CHECKPOINT}" >&2
  echo "" >&2
  echo "Stage 3는 Stage 2 완료 후 실행해야 합니다:" >&2
  echo "  1. Stage 2 학습 먼저 실행:" >&2
  echo "     ./tools/train_ko_stage2.sh" >&2
  echo "  2. Stage 2 checkpoint 확인:" >&2
  echo "     ls -lh /mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth" >&2
  echo "  3. Stage 3 학습 실행:" >&2
  echo "     ./tools/train_ko_stage3.sh" >&2
  echo "" >&2
  exit 1
fi

# GPU 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo "✅ GPU: ${GPU_NAME}"
echo "✅ VRAM: ${GPU_MEM}MB"
echo ""

# ============================================================
# Stage 3 Configuration Summary
# ============================================================

echo "📊 Stage 3 하이퍼파라미터:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACC} (실효 batch ${((BATCH_SIZE * GRAD_ACC))})"
echo "  - Learning Rate: ${LR} (Stage 1/2보다 낮음)"
echo "  - Warmup Steps: ${WARMUP_STEPS}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Gradient Clip: ${GRAD_CLIP}"
echo ""
echo "🔒 Freezing:"
echo "  - Speaker Perceiver: FROZEN (Stage 1에서 학습완료)"
echo "  - Emotion Perceiver: FROZEN (Stage 2에서 학습완료)"
echo "  - GPT Backbone: TRAINABLE"
echo "  - GRL: DISABLED (Stage 2에서만 사용)"
echo ""
echo "📚 참고:"
echo "  - Stage 3는 feature conditioner를 고정하고 GPT만 fine-tuning합니다"
echo "  - Stage 1, 2에서 학습된 speaker/emotion feature를 보존합니다"
echo "  - Overfitting을 방지하고 생성 품질만 개선합니다"
echo ""
echo "================================================================"
echo ""

# 사용자 확인
read -p "Stage 3 학습을 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

echo ""
echo "🎬 Stage 3 학습을 시작합니다..."
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 체크포인트: ${CHECKPOINT_DIR}/"
echo ""

# ============================================================
# Stage 3 Training
# ============================================================

echo ""
echo "================================================================"
echo "✅ Stage 3 학습 시작 (Frozen Conditioners Fine-tuning)"
echo "================================================================"
echo ""
echo "📊 주요 설정:"
echo "  ✅ Feature conditioners frozen"
echo "  ✅ GPT backbone trainable"
echo "  ✅ GRL disabled"
echo "  ✅ Lower learning rate for fine-tuning"
echo ""
echo "================================================================"
echo ""

# Stage 3 Training with Frozen Conditioners
python trainers/train_gpt_v2.py \
    --train-manifest ${DATASET_DIR}/train_manifest.jsonl \
    --val-manifest ${DATASET_DIR}/val_manifest.jsonl \
    --tokenizer ${TOKENIZER} \
    --config ${CONFIG} \
    --base-checkpoint ${STAGE2_CHECKPOINT} \
    --output-dir ${CHECKPOINT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --grad-accumulation ${GRAD_ACC} \
    --learning-rate ${LR} \
    --warmup-steps ${WARMUP_STEPS} \
    --epochs ${EPOCHS} \
    --grad-clip ${GRAD_CLIP} \
    --freeze-conditioners \
    --log-interval ${LOG_INTERVAL} \
    --val-interval ${VAL_INTERVAL}

echo ""
echo "================================================================"
echo "✅ Stage 3 학습이 완료되었습니다"
echo "================================================================"
