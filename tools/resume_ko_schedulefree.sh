#!/usr/bin/env bash
# Schedule-Free AdamW 학습 재개
set -euo pipefail

# Activate virtual environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "🔄 Schedule-Free AdamW 학습 재개"
echo "================================================================"
echo ""
echo "📊 실험 설정:"
echo "  - Optimizer: Schedule-Free AdamW"
echo "  - Resume from: latest.pth"
echo "  - Optimizer state 유지하여 재개"
echo ""

SCRIPT_DIR="/mnt/sdc1/ws/workspace/monorepo/external/index-tts"

cd "${SCRIPT_DIR}"

# Find latest checkpoint
CHECKPOINT_DIR="/mnt/sda1/models/index-tts-ko/checkpoints"
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pth"

if [ ! -f "${LATEST_CHECKPOINT}" ]; then
    echo "❌ Error: latest.pth not found at ${LATEST_CHECKPOINT}"
    exit 1
fi

echo "[KO-STEP4] Resuming from ${LATEST_CHECKPOINT}..."

SKIP_DATA_CHECK=1 \
OPTIMIZER=schedulefree \
LR=5e-4 \
BATCH_SIZE=8 \
GRAD_ACC=1 \
LOG_INTERVAL=100 \
VAL_INTERVAL=10000 \
MAX_STEPS=240000 \
EPOCHS=999 \
NUM_WORKERS=32 \
RESUME="${LATEST_CHECKPOINT}" \
"${SCRIPT_DIR}/tools/ko_step4_train_gpt.sh"
