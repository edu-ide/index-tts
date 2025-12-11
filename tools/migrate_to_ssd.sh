#!/usr/bin/env bash
# Migrate 500K subset data from HDD to SSD (remove symlink, actual copy)
set -euo pipefail

HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
SSD_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"

echo "================================================================"
echo "Migrate data to SSD"
echo "================================================================"
echo ""

# Check if symlink exists
if [[ -L "${SSD_DIR}" ]]; then
    echo "[1/4] Removing symlink: ${SSD_DIR}"
    rm "${SSD_DIR}"
else
    echo "[1/4] No symlink found (already real directory or doesn't exist)"
fi

# Create real directory
echo "[2/4] Creating SSD directory..."
mkdir -p "${SSD_DIR}/mel"

# Copy mel files (56GB)
echo "[3/4] Copying mel files to SSD (56GB, ~5-10 min)..."
rsync -av --progress "${HDD_DIR}/mel/" "${SSD_DIR}/mel/"

# Copy manifests and regenerate
echo "[4/4] Copying manifests..."
cp "${HDD_DIR}/gpt_pairs_train_500000.jsonl" "${SSD_DIR}/"
cp "${HDD_DIR}/gpt_pairs_val_10k.jsonl" "${SSD_DIR}/"

# Regenerate mel manifests with SSD paths
echo ""
echo "Regenerating mel manifests..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

python3 "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${SSD_DIR}/gpt_pairs_train_500000.jsonl" \
    --output-manifest "${SSD_DIR}/gpt_pairs_train_500000_mel.jsonl" \
    --data-dir "${SSD_DIR}" \
    --num-workers 32

python3 "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${SSD_DIR}/gpt_pairs_val_10k.jsonl" \
    --output-manifest "${SSD_DIR}/gpt_pairs_val_10k_mel.jsonl" \
    --data-dir "${SSD_DIR}" \
    --num-workers 32

echo ""
echo "================================================================"
echo "Migration complete!"
echo "================================================================"
echo ""
echo "Train manifest: $(wc -l < "${SSD_DIR}/gpt_pairs_train_500000_mel.jsonl") samples"
echo "Val manifest: $(wc -l < "${SSD_DIR}/gpt_pairs_val_10k_mel.jsonl") samples"
echo ""
echo "Now run: ./tools/run_stage2_training.sh"
