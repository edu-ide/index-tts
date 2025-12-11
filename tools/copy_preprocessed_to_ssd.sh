#!/usr/bin/env bash
# Copy all preprocessed directories from HDD to SSD
set -euo pipefail

HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
SSD_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"

# Directories to copy (exclude mel and KO-B* which are already there)
DIRS_TO_COPY=(
    "text_ids"
    "codes"
    "condition"
    "emo_vec"
)

echo "================================================================"
echo "Copy Preprocessed Data to SSD"
echo "================================================================"
echo ""
echo "Source: ${HDD_DIR}"
echo "Target: ${SSD_DIR}"
echo ""

# Check SSD free space
echo "=== SSD Free Space ==="
df -h /mnt/sda1
echo ""

# Copy each directory
TOTAL=${#DIRS_TO_COPY[@]}
COUNT=0

for dir in "${DIRS_TO_COPY[@]}"; do
    COUNT=$((COUNT + 1))
    SRC="${HDD_DIR}/${dir}"
    DST="${SSD_DIR}/${dir}"

    if [[ ! -d "${SRC}" ]]; then
        echo "[${COUNT}/${TOTAL}] ${dir}: NOT FOUND on HDD, skipping"
        continue
    fi

    echo "[${COUNT}/${TOTAL}] Copying ${dir}..."
    mkdir -p "${DST}"
    rsync -av --progress "${SRC}/" "${DST}/"
    echo ""
done

echo "================================================================"
echo "Done!"
echo "================================================================"
echo ""

# Summary
echo "=== SSD Summary ==="
for dir in mel text_ids codes condition emo_vec; do
    if [[ -d "${SSD_DIR}/${dir}" ]]; then
        count=$(ls "${SSD_DIR}/${dir}" 2>/dev/null | wc -l)
        echo "${dir}: ${count} files"
    else
        echo "${dir}: NOT FOUND"
    fi
done
echo ""
echo "Next: ./tools/run_stage2_training.sh"
