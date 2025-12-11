#!/usr/bin/env bash
# Regenerate mel manifests for 500K subset (mel files already exist)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

SRC_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
DST_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"
NUM_WORKERS="${NUM_WORKERS:-32}"

echo "================================================================"
echo "Regenerate mel manifests (500K subset)"
echo "================================================================"
echo ""

# Train
echo "[1/2] Processing train (500K)..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${SRC_DIR}/gpt_pairs_train_500000.jsonl" \
    --output-manifest "${DST_DIR}/gpt_pairs_train_500000_mel.jsonl" \
    --data-dir "${SRC_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "[2/2] Processing val (10K)..."
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${SRC_DIR}/gpt_pairs_val_10k.jsonl" \
    --output-manifest "${DST_DIR}/gpt_pairs_val_10k_mel.jsonl" \
    --data-dir "${SRC_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "================================================================"
echo "Done!"
echo "================================================================"
echo ""
echo "Train: $(wc -l < "${DST_DIR}/gpt_pairs_train_500000_mel.jsonl") samples"
echo "Val: $(wc -l < "${DST_DIR}/gpt_pairs_val_10k_mel.jsonl") samples"
