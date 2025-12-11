#!/usr/bin/env bash
# Copy 500K subset audio files from HDD to SSD
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
SSD_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"
MANIFEST="${SSD_DIR}/gpt_pairs_train_500000_mel.jsonl"

echo "================================================================"
echo "Copy 500K Audio Files to SSD"
echo "================================================================"
echo ""
echo "HDD source: ${HDD_DIR}"
echo "SSD target: ${SSD_DIR}"
echo "Manifest: ${MANIFEST}"
echo ""

# Check SSD free space
echo "=== SSD Free Space ==="
df -h /mnt/sda1
echo ""

# Remove existing symlinks to KO-B* directories
echo "=== Removing existing symlinks ==="
for link in ${SSD_DIR}/KO-B*; do
    if [[ -L "$link" ]]; then
        echo "Removing symlink: $link"
        rm "$link"
    fi
done
echo ""

# Run copy script
echo "=== Copying audio files ==="
python3 "${PROJECT_ROOT}/tools/copy_audio_to_ssd.py" \
    --manifest "${MANIFEST}" \
    --hdd-dir "${HDD_DIR}" \
    --ssd-dir "${SSD_DIR}" \
    --num-workers 64

echo ""
echo "================================================================"
echo "Done!"
echo "================================================================"
echo ""
echo "SSD audio files: $(find ${SSD_DIR}/KO-B* -name '*.mp3' 2>/dev/null | wc -l)"
echo "SSD used space: $(du -sh ${SSD_DIR} 2>/dev/null | cut -f1)"
echo ""
echo "Next: ./tools/run_stage2_training.sh"
