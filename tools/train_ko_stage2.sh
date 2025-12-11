#!/usr/bin/env bash
# IndexTTS2 Stage 2: Emotion Disentanglement with GRL
#
# Stage 2 설정 (IndexTTS2 논문 기반):
# - Speaker perceiver conditioner: FROZEN (Stage 1에서 학습됨)
# - Emotion perceiver conditioner: TRAINABLE
# - GRL + Speaker Classifier: ENABLED
# - Learning Rate: 2e-4 (논문 권장값)
# - 감정 데이터셋: 135시간 (논문), 현재는 전체 데이터 사용
#
# Loss Function:
#   LAR = TTS_loss + α * Speaker_classification_loss
#   (GRL이 gradient를 자동으로 역전시킴)
#
# 참고문헌:
# - IndexTTS2 (arXiv:2506.21619v2)
# - Ganin et al. 2016 (JMLR) - GRL

set -euo pipefail

echo "================================================================"
echo "🎭 IndexTTS2 Stage 2: Emotion Disentanglement (GRL)"
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
# Stage 2 Configuration
# ============================================================

# Paths
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/sda1/models/index-tts-ko/stage2}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth}"
SPEAKER_MAPPING="${SPEAKER_MAPPING:-/mnt/sda1/models/index-tts-ko/speaker_mapping.json}"

# Model config
CONFIG="${CONFIG:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

# Training hyperparameters (IndexTTS2 논문 기반)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
EPOCHS="${EPOCHS:-2}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"

# Stage 2 specific
GRL_LAMBDA="${GRL_LAMBDA:-1.0}"
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"
GRL_SCHEDULE="${GRL_SCHEDULE:-exponential}"

# Logging
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-32}"

# ============================================================
# Validation
# ============================================================

echo "📂 Paths:"
echo "  - Dataset: ${DATASET_DIR}"
echo "  - Output: ${CHECKPOINT_DIR}"
echo "  - Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
echo "  - Speaker mapping: ${SPEAKER_MAPPING}"
echo ""

# Check Stage 1 checkpoint
if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
  echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
  echo "" >&2
  echo "Stage 2는 Stage 1 완료 후 실행해야 합니다:" >&2
  echo "  1. Stage 1 학습 먼저 실행:" >&2
  echo "     ./tools/train_ko_optimized_4090.sh" >&2
  echo "  2. Stage 1 checkpoint 확인:" >&2
  echo "     ls -lh /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth" >&2
  echo "  3. Stage 2 학습 실행:" >&2
  echo "     ./tools/train_ko_stage2.sh" >&2
  echo "" >&2
  exit 1
fi

# Check speaker mapping
if [[ ! -f "${SPEAKER_MAPPING}" ]]; then
  echo "[ERROR] Speaker mapping not found: ${SPEAKER_MAPPING}" >&2
  echo "" >&2
  echo "Speaker mapping을 먼저 생성해야 합니다:" >&2
  echo "  python tools/build_speaker_mapping.py \\" >&2
  echo "    --manifest ${DATASET_DIR}/gpt_pairs_train.jsonl \\" >&2
  echo "    --output ${SPEAKER_MAPPING} \\" >&2
  echo "    --top-k 500 \\" >&2
  echo "    --min-samples 50" >&2
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
# Stage 2 Configuration Summary
# ============================================================

echo "📊 Stage 2 하이퍼파라미터:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACC} (실효 batch $((BATCH_SIZE * GRAD_ACC)))"
echo "  - Learning Rate: ${LR}"
echo "  - Warmup Steps: ${WARMUP_STEPS}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Gradient Clip: ${GRAD_CLIP}"
echo ""
echo "🎭 GRL 설정:"
echo "  - GRL Lambda: ${GRL_LAMBDA}"
echo "  - Speaker Loss Weight: ${SPEAKER_LOSS_WEIGHT}"
echo "  - Lambda Schedule: ${GRL_SCHEDULE}"
echo ""
echo "🔒 Freezing:"
echo "  - Speaker Perceiver: FROZEN (Stage 1에서 학습완료)"
echo "  - Emotion Perceiver: TRAINABLE"
echo ""
echo "📚 참고:"
echo "  - IndexTTS2 Stage 2는 speaker-emotion disentanglement를 수행합니다"
echo "  - GRL은 emotion vector에서 speaker identity를 제거합니다"
echo "  - 결과적으로 speaker와 무관한 순수한 감정 제어가 가능합니다"
echo ""
echo "================================================================"
echo ""

# 사용자 확인
read -p "Stage 2 학습을 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

echo ""
echo "🎬 Stage 2 학습을 시작합니다..."
echo "📊 TensorBoard: http://localhost:6006"
echo "📁 체크포인트: ${CHECKPOINT_DIR}/"
echo ""

# ============================================================
# Stage 2 Training
# ============================================================

echo ""
echo "================================================================"
echo "✅ Stage 2 학습 시작 (GRL + Pre-computed Emo-Vec)"
echo "================================================================"
echo ""
echo "📊 주요 변경사항:"
echo "  ✅ GRL + Speaker Classifier 통합 완료"
echo "  ✅ Speaker classification loss 추가 완료"
echo "  ✅ Speaker mapping 로드 완료 (500 speakers)"
echo ""
echo "🎯 GRL 방식:"
echo "  - Pre-computed emo_vec 사용 (Wav2Vec2-BERT 피처)"
echo "  - GRL이 emo_vec에서 speaker identity 제거"
echo "  - Speaker classifier로 adversarial training"
echo ""
echo "================================================================"
echo ""

# Stage 2 Training with GRL and Real-time Mel Computation (Paper Approach)
# IndexTTS2 논문 방식: Audio → Mel → emo_conditioning_encoder → GRL
# 이 방식은 gradient가 emo encoder를 통해 흐르므로 proper adversarial training 가능
python "${PROJECT_ROOT}/trainers/train_gpt_v2.py" \
    --train-manifest ${DATASET_DIR}/gpt_pairs_train.jsonl \
    --val-manifest ${DATASET_DIR}/gpt_pairs_val.jsonl \
    --tokenizer ${TOKENIZER} \
    --config ${CONFIG} \
    --base-checkpoint ${STAGE1_CHECKPOINT} \
    --output-dir ${CHECKPOINT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --grad-accumulation ${GRAD_ACC} \
    --learning-rate ${LR} \
    --warmup-steps ${WARMUP_STEPS} \
    --epochs ${EPOCHS} \
    --grad-clip ${GRAD_CLIP} \
    --enable-grl \
    --speaker-mapping ${SPEAKER_MAPPING} \
    --grl-lambda ${GRL_LAMBDA} \
    --speaker-loss-weight ${SPEAKER_LOSS_WEIGHT} \
    --enable-stage2-realtime-emo \
    --emo-mel-input-size 80 \
    --log-interval ${LOG_INTERVAL} \
    --val-interval ${VAL_INTERVAL} \
    --num-workers ${NUM_WORKERS}

echo ""
echo "================================================================"
echo "✅ Stage 2 학습이 완료되었습니다"
echo "================================================================"
