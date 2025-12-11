#!/usr/bin/env bash
# Automated Stage 2 Pipeline with Subset (500K samples)
# No rsync needed - runs directly on HDD
# Run this and go to sleep!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration
DATASET_DIR="${DATASET_DIR:-/mnt/sdb1/emilia-yodas/KO_preprocessed}"
NUM_WORKERS="${NUM_WORKERS:-32}"
SUBSET_SIZE="${SUBSET_SIZE:-500000}"

# Stage 2 training config
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth}"
SPEAKER_MAPPING="${SPEAKER_MAPPING:-/mnt/sda1/models/index-tts-ko/speaker_mapping_500k.json}"
CONFIG="${CONFIG:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

echo "================================================================"
echo "Automated Stage 2 Pipeline (${SUBSET_SIZE} samples)"
echo "================================================================"
echo ""
echo "Plan:"
echo "  1. Create subset manifests (if needed)"
echo "  2. Preprocess mel-spectrograms"
echo "  3. Start Stage 2 training"
echo ""
echo "You can go to sleep now!"
echo ""
echo "================================================================"
echo ""

# Kill any existing mel preprocessing
pkill -f "preprocess_mel_for_stage2.py" 2>/dev/null || true

# ============================================================
# Step 1: Create subset manifests (if not exist)
# ============================================================
TRAIN_SUBSET="${DATASET_DIR}/gpt_pairs_train_${SUBSET_SIZE}.jsonl"
VAL_SUBSET="${DATASET_DIR}/gpt_pairs_val_10k.jsonl"

echo "[1/3] Checking subset manifests..."

if [[ ! -f "${TRAIN_SUBSET}" ]]; then
    echo "Creating train subset: ${TRAIN_SUBSET}"
    head -n "${SUBSET_SIZE}" "${DATASET_DIR}/gpt_pairs_train.jsonl" > "${TRAIN_SUBSET}"
fi

if [[ ! -f "${VAL_SUBSET}" ]]; then
    echo "Creating val subset: ${VAL_SUBSET}"
    head -n 10000 "${DATASET_DIR}/gpt_pairs_val.jsonl" > "${VAL_SUBSET}"
fi

echo "Train subset: $(wc -l < "${TRAIN_SUBSET}") samples"
echo "Val subset: $(wc -l < "${VAL_SUBSET}") samples"
echo ""

# ============================================================
# Step 2: Preprocess mel-spectrograms
# ============================================================
echo "[2/3] Preprocessing mel-spectrograms..."
echo ""

TRAIN_MEL="${DATASET_DIR}/gpt_pairs_train_${SUBSET_SIZE}_mel.jsonl"
VAL_MEL="${DATASET_DIR}/gpt_pairs_val_10k_mel.jsonl"

# Train data
echo "[2a/3] Processing TRAIN data..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${TRAIN_SUBSET}" \
    --output-manifest "${TRAIN_MEL}" \
    --data-dir "${DATASET_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "[2b/3] Processing VAL data..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${VAL_SUBSET}" \
    --output-manifest "${VAL_MEL}" \
    --data-dir "${DATASET_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "Mel preprocessing complete!"
echo ""

# ============================================================
# Step 3: Stage 2 Training
# ============================================================
echo "[3/3] Starting Stage 2 Training..."
echo ""

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EPOCHS="${EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
GRL_LAMBDA="${GRL_LAMBDA:-1.0}"
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Hyperparameters:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACC} (effective batch: $((BATCH_SIZE * GRAD_ACC)))"
echo "  - Learning Rate: ${LR}"
echo "  - GRL Lambda: ${GRL_LAMBDA}"
echo "  - Train samples: ${SUBSET_SIZE}"
echo ""

# Use pre-computed mel (faster)
python "${PROJECT_ROOT}/trainers/train_gpt_v2.py" \
    --train-manifest "${TRAIN_MEL}" \
    --val-manifest "${VAL_MEL}" \
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
echo "Stage 2 Training Complete!"
echo "================================================================"
echo ""
echo "Checkpoint: ${STAGE2_DIR}"
echo ""
