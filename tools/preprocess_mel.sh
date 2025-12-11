#!/usr/bin/env bash
# Stage 2 Mel-spectrogram Preprocessing
# Pre-compute mel-spectrograms from audio files for faster training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
NUM_WORKERS="${NUM_WORKERS:-32}"

echo "================================================================"
echo "🎵 Stage 2 Mel-spectrogram Preprocessing"
echo "================================================================"
echo ""
echo "📂 Dataset: ${DATASET_DIR}"
echo "👷 Workers: ${NUM_WORKERS}"
echo ""

# Train data
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [1/2] Processing TRAIN data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DATASET_DIR}/gpt_pairs_train.jsonl" \
    --output-manifest "${DATASET_DIR}/gpt_pairs_train_mel.jsonl" \
    --data-dir "${DATASET_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 [2/2] Processing VAL data..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DATASET_DIR}/gpt_pairs_val.jsonl" \
    --output-manifest "${DATASET_DIR}/gpt_pairs_val_mel.jsonl" \
    --data-dir "${DATASET_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "================================================================"
echo "✅ Mel preprocessing complete!"
echo "================================================================"
echo ""
echo "Output files:"
echo "  - ${DATASET_DIR}/gpt_pairs_train_mel.jsonl"
echo "  - ${DATASET_DIR}/gpt_pairs_val_mel.jsonl"
echo "  - ${DATASET_DIR}/mel/*.npy"
echo ""
echo "Next step: Run Stage 2 training with pre-computed mel"
