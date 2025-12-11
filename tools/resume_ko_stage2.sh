#!/usr/bin/env bash
# Resume Stage 2 training from an existing checkpoint.
# If FRESH_OPT=1, load only model weights and reset optimizer/scheduler.
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Stage 2 checkpoint directory
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2}"

# Auto-detect checkpoint: prefer full checkpoint
DEFAULT_FULL="${STAGE2_DIR}/latest_full.pth"
DEFAULT_LIGHT="${STAGE2_DIR}/latest.pth"
if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "${DEFAULT_FULL}" ]]; then
    CKPT="${DEFAULT_FULL}"
  elif [[ -f "${DEFAULT_LIGHT}" ]]; then
    CKPT="${DEFAULT_LIGHT}"
  else
    echo "[ERROR] No Stage 2 checkpoint found in ${STAGE2_DIR}" >&2
    echo "  Expected: ${DEFAULT_FULL} or ${DEFAULT_LIGHT}" >&2
    exit 1
  fi
fi

# Sanitize possible quoted CKPT
CKPT="${CKPT%\"}"
CKPT="${CKPT#\"}"
CKPT="${CKPT%\'}"
CKPT="${CKPT#\'}"

# Default: resume optimizer/scheduler state
FRESH_OPT="${FRESH_OPT:-0}"

echo "================================================================"
echo "🎭 IndexTTS2 Stage 2 Resume"
echo "================================================================"
echo ""
echo "📂 Resuming from: ${CKPT}"
if [[ "${FRESH_OPT}" == "1" ]]; then
  echo "  -> FRESH_OPT=1: optimizer/scheduler will NOT be resumed (model weights only)."
else
  echo "  -> FRESH_OPT=0: optimizer/scheduler will be resumed."
fi
echo ""

# Ensure checkpoint exists
if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT not found: ${CKPT}" >&2
  exit 1
fi

# Paths
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
SPEAKER_MAPPING="${SPEAKER_MAPPING:-/mnt/sda1/models/index-tts-ko/speaker_mapping.json}"
CONFIG="${CONFIG:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

# Training hyperparameters (can override via env)
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
EPOCHS="${EPOCHS:-2}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-32}"

# Stage 2 specific (GRL)
GRL_LAMBDA="${GRL_LAMBDA:-1.0}"
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"

# Logging
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"

# CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "📊 Stage 2 하이퍼파라미터:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACC} (실효 batch $((BATCH_SIZE * GRAD_ACC)))"
echo "  - Learning Rate: ${LR}"
echo "  - GRL Lambda: ${GRL_LAMBDA}"
echo "  - Speaker Loss Weight: ${SPEAKER_LOSS_WEIGHT}"
echo ""

# Determine resume argument
if [[ "${FRESH_OPT}" == "1" ]]; then
  RESUME_ARG=""
  BASE_CKPT_ARG="--base-checkpoint ${CKPT}"
else
  RESUME_ARG="--resume ${CKPT}"
  BASE_CKPT_ARG=""
fi

# Stage 2 Training with GRL and Real-time Mel Computation (Paper Approach)
# IndexTTS2 논문 방식: Audio → Mel → emo_conditioning_encoder → GRL
python "${PROJECT_ROOT}/trainers/train_gpt_v2.py" \
    --train-manifest ${DATASET_DIR}/gpt_pairs_train.jsonl \
    --val-manifest ${DATASET_DIR}/gpt_pairs_val.jsonl \
    --tokenizer ${TOKENIZER} \
    --config ${CONFIG} \
    ${BASE_CKPT_ARG} \
    ${RESUME_ARG} \
    --output-dir ${STAGE2_DIR} \
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
echo "✅ Stage 2 Resume 완료"
echo "================================================================"
