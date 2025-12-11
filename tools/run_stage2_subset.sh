#!/usr/bin/env bash
# Stage 2 Subset Training Automation Script
# 
# Strategy:
# 1. Create a subset of the training manifest (first 430k samples) which already have pre-computed mel spectrograms.
# 2. Train on this subset using SSD/HDD data.
# 3. Force MAX_STEPS to the value expected for the FULL dataset (5.5M).
#    This prevents the WSD scheduler from decaying the learning rate prematurely.
#    (Scheduler thinks we are at step 0-20k of 257k, so it keeps LR stable/high)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration
HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
FULL_MANIFEST="${HDD_DIR}/gpt_pairs_train_mel.jsonl"
SUBSET_MANIFEST="${HDD_DIR}/gpt_pairs_train_mel_subset.jsonl"
LOG_FILE="/mnt/sda1/models/index-tts-ko/stage2/train_subset.log"
SUBSET_SIZE=430000

# 1. Create Subset Manifest
echo "----------------------------------------------------------------"
echo "[1/2] Creating subset manifest (Top ${SUBSET_SIZE} samples)"
echo "----------------------------------------------------------------"

if [[ ! -f "${FULL_MANIFEST}" ]]; then
    echo "Error: Source manifest not found: ${FULL_MANIFEST}"
    exit 1
fi

# Use head to extract the first N lines (which have mel files ready)
head -n "${SUBSET_SIZE}" "${FULL_MANIFEST}" > "${SUBSET_MANIFEST}"
echo "Created: ${SUBSET_MANIFEST}"
echo "Line count: $(wc -l < "${SUBSET_MANIFEST}")"
echo ""

# 2. Configure Training
# Full dataset: ~5,483,856 samples
# Batch size: 16, Grad Acc: 4 -> Effective Batch: 64
# Steps per Epoch: 5,483,856 / 64 = 85,685
# Total Steps (3 Epochs): 85,685 * 3 = 257,055
#
# We use MAX_STEPS=257055 so the scheduler behaves as if we are training on the full dataset.

export DATASET_DIR="${HDD_DIR}"
export TRAIN_MANIFEST="${SUBSET_MANIFEST}"
export EPOCHS=3
export MAX_STEPS=257055
export NUM_WORKERS=16  # Reduced from 32 to prevent OOM
export RESUME="auto"   # Resume from latest checkpoint if available

echo "----------------------------------------------------------------"
echo "[2/2] Starting Training"
echo "----------------------------------------------------------------"
echo "Dataset: ${TRAIN_MANIFEST}"
echo "Max Steps: ${MAX_STEPS} (Fixed for WSD Scheduler)"
echo "Workers: ${NUM_WORKERS}"
echo "Log File: ${LOG_FILE}"
echo "----------------------------------------------------------------"

# Run training in background with nohup
nohup "${PROJECT_ROOT}/tools/run_stage2_training.sh" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo ""
echo "To monitor progress, run:"
echo "  tail -f ${LOG_FILE}"
