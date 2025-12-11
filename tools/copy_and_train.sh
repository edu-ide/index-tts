#!/usr/bin/env bash
# Copy all preprocessed data to SSD and start Stage 2 training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

HDD_DIR="/mnt/sdb1/emilia-yodas/KO_preprocessed"
SSD_DIR="/mnt/sda1/emilia-yodas/KO_preprocessed"

echo "================================================================"
echo "Stage 2 Training Pipeline (Copy + Train)"
echo "================================================================"
echo ""
echo "Started at: $(date)"
echo ""

# ============================================================
# Step 1: Copy preprocessed data to SSD
# ============================================================
echo "================================================================"
echo "[Step 1/2] Copying Preprocessed Data to SSD"
echo "================================================================"
echo ""

DIRS_TO_COPY=(
    "text_ids"
    "codes"
    "condition"
    "emo_vec"
)

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

echo "=== Copy Complete ==="
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

# ============================================================
# Step 2: Start Stage 2 Training
# ============================================================
echo "================================================================"
echo "[Step 2/2] Starting Stage 2 GRL Training"
echo "================================================================"
echo ""

cd "${PROJECT_ROOT}"

# Training config
TRAIN_MANIFEST="${SSD_DIR}/gpt_pairs_train_500000_mel.jsonl"
VAL_MANIFEST="${SSD_DIR}/gpt_pairs_val_10k_mel.jsonl"
TOKENIZER="/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model"
CONFIG="/mnt/sda1/models/IndexTTS-2/config.yaml"
BASE_CHECKPOINT="/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
OUTPUT_DIR="/mnt/sda1/models/index-tts-ko/stage2"
SPEAKER_MAPPING="/mnt/sda1/models/index-tts-ko/speaker_mapping_500k.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Train manifest: ${TRAIN_MANIFEST}"
echo "Val manifest: ${VAL_MANIFEST}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check manifests exist
if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
    echo "[Error] Train manifest not found: ${TRAIN_MANIFEST}"
    exit 1
fi

if [[ ! -f "${VAL_MANIFEST}" ]]; then
    echo "[Error] Val manifest not found: ${VAL_MANIFEST}"
    exit 1
fi

echo "Starting training at: $(date)"
echo ""

# Run training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python trainers/train_gpt_v2.py \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --tokenizer "${TOKENIZER}" \
    --config "${CONFIG}" \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 8 \
    --grad-accumulation 8 \
    --learning-rate 2e-4 \
    --warmup-steps 1000 \
    --epochs 3 \
    --grad-clip 0.5 \
    --enable-grl \
    --speaker-mapping "${SPEAKER_MAPPING}" \
    --grl-lambda 1.0 \
    --speaker-loss-weight 0.1 \
    --log-interval 100 \
    --val-interval 500 \
    --num-workers 16 \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================"
echo "Training Complete!"
echo "================================================================"
echo "Finished at: $(date)"
echo "Output: ${OUTPUT_DIR}"
