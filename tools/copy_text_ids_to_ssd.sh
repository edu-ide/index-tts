#!/usr/bin/env bash
# Copy text_ids from HDD to SSD
set -euo pipefail

HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
SSD_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"

echo "================================================================"
echo "Copy text_ids to SSD"
echo "================================================================"
echo ""
echo "Source: ${HDD_DIR}/text_ids/"
echo "Target: ${SSD_DIR}/text_ids/"
echo ""

# Check SSD free space
echo "=== SSD Free Space ==="
df -h /mnt/sda1
echo ""

# Create target directory
mkdir -p "${SSD_DIR}/text_ids"

# Copy with rsync
echo "=== Copying text_ids ==="
rsync -av --progress "${HDD_DIR}/text_ids/" "${SSD_DIR}/text_ids/"

echo ""
echo "================================================================"
echo "Done!"
echo "================================================================"
echo ""
echo "SSD text_ids: $(ls ${SSD_DIR}/text_ids/ | wc -l) files"
