#!/usr/bin/env bash
# Automated Stage 2 Pipeline: rsync → preprocess mel → train
# Run this and go to sleep!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration
SRC_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
DST_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed_ssd"
NUM_WORKERS="${NUM_WORKERS:-32}"

# Stage 2 training config
STAGE2_DIR="/mnt/sda1/models/index-tts-ko/stage2"
STAGE1_CHECKPOINT="/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
SPEAKER_MAPPING="/mnt/sda1/models/index-tts-ko/speaker_mapping.json"
CONFIG="/mnt/sda1/models/IndexTTS-2/config.yaml"
TOKENIZER="/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model"

echo "================================================================"
echo "🚀 Automated Stage 2 Pipeline"
echo "================================================================"
echo ""
echo "📋 Plan:"
echo "  1. rsync: HDD → SSD"
echo "  2. Preprocess mel-spectrograms"
echo "  3. Start Stage 2 training"
echo ""
echo "You can go to sleep now! 😴"
echo ""
echo "================================================================"
echo ""

# Kill any existing mel preprocessing
pkill -f "preprocess_mel_for_stage2.py" 2>/dev/null || true

# ============================================================
# Step 1: rsync to SSD
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 [1/3] Copying data to SSD..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Source: ${SRC_DIR}"
echo "Destination: ${DST_DIR}"
echo ""

mkdir -p "${DST_DIR}"

rsync -av --progress \
    --exclude='mel/' \
    "${SRC_DIR}/" "${DST_DIR}/"

echo ""
echo "✅ rsync complete!"
echo ""

# ============================================================
# Step 2: Preprocess mel-spectrograms
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎵 [2/3] Preprocessing mel-spectrograms on SSD..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Train data
echo "📊 [2a/3] Processing TRAIN data..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DST_DIR}/gpt_pairs_train.jsonl" \
    --output-manifest "${DST_DIR}/gpt_pairs_train_mel.jsonl" \
    --data-dir "${DST_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "📊 [2b/3] Processing VAL data..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DST_DIR}/gpt_pairs_val.jsonl" \
    --output-manifest "${DST_DIR}/gpt_pairs_val_mel.jsonl" \
    --data-dir "${DST_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "✅ Mel preprocessing complete!"
echo ""

# ============================================================
# Step 3: Stage 2 Training
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎭 [3/3] Starting Stage 2 Training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
EPOCHS="${EPOCHS:-2}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
GRL_LAMBDA="${GRL_LAMBDA:-1.0}"
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "📊 Hyperparameters:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACC}"
echo "  - Learning Rate: ${LR}"
echo "  - GRL Lambda: ${GRL_LAMBDA}"
echo ""

# Use pre-computed mel (Option A - faster)
python "${PROJECT_ROOT}/trainers/train_gpt_v2.py" \
    --train-manifest "${DST_DIR}/gpt_pairs_train_mel.jsonl" \
    --val-manifest "${DST_DIR}/gpt_pairs_val_mel.jsonl" \
    --tokenizer "${TOKENIZER}" \
    --config "${CONFIG}" \
    --base-checkpoint "${STAGE1_CHECKPOINT}" \
    --output-dir "${STAGE2_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accumulation "${GRAD_ACC}" \
    --learning-rate "${LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --epochs "${EPOCHS}" \
    --grad-clip "${GRAD_CLIP}" \
    --enable-grl \
    --speaker-mapping "${SPEAKER_MAPPING}" \
    --grl-lambda "${GRL_LAMBDA}" \
    --speaker-loss-weight "${SPEAKER_LOSS_WEIGHT}" \
    --log-interval "${LOG_INTERVAL}" \
    --val-interval "${VAL_INTERVAL}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "================================================================"
echo "✅ Stage 2 Training Complete!"
echo "================================================================"
echo ""
echo "Checkpoint: ${STAGE2_DIR}"
echo ""
