#!/usr/bin/env bash
# Copy mel data from HDD to SSD for faster training
set -euo pipefail

SRC_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
DST_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"

echo "================================================================"
echo "Copy mel data to SSD"
echo "================================================================"
echo ""
echo "Source: ${SRC_DIR}/mel (56GB)"
echo "Destination: ${DST_DIR}/mel"
echo ""

# Check source
if [[ ! -d "${SRC_DIR}/mel" ]]; then
    echo "ERROR: Source mel directory not found!"
    exit 1
fi

# Check SSD space
FREE_GB=$(df -BG /mnt/sda1 | tail -1 | awk '{print $4}' | sed 's/G//')
echo "SSD free space: ${FREE_GB}GB"
if [[ ${FREE_GB} -lt 60 ]]; then
    echo "WARNING: Less than 60GB free space!"
fi
echo ""

# Create destination
mkdir -p "${DST_DIR}/mel"

# Copy mel files
echo "[1/3] Copying mel files..."
rsync -av --progress "${SRC_DIR}/mel/" "${DST_DIR}/mel/"
echo ""

# Copy and update manifests
echo "[2/3] Copying and updating manifests..."
for manifest in gpt_pairs_train_500000_mel.jsonl gpt_pairs_val_10k_mel.jsonl; do
    if [[ -f "${SRC_DIR}/${manifest}" ]]; then
        echo "  Processing ${manifest}..."
        sed 's|/mnt/sdb1/|/mnt/sda1/|g' "${SRC_DIR}/${manifest}" > "${DST_DIR}/${manifest}"
    fi
done
echo ""

# Verify
echo "[3/3] Verifying..."
SRC_COUNT=$(find "${SRC_DIR}/mel" -name "*.npy" | wc -l)
DST_COUNT=$(find "${DST_DIR}/mel" -name "*.npy" | wc -l)
echo "Source mel files: ${SRC_COUNT}"
echo "Destination mel files: ${DST_COUNT}"

if [[ ${SRC_COUNT} -eq ${DST_COUNT} ]]; then
    echo ""
    echo "================================================================"
    echo "Copy complete!"
    echo "================================================================"
    echo ""
    echo "Now update run_stage2_training.sh:"
    echo "  DATASET_DIR=/mnt/sda1/emilia-yodas/KO_preprocessed"
    echo ""
else
    echo ""
    echo "WARNING: File count mismatch!"
fi
