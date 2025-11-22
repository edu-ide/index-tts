#!/usr/bin/env bash
# Prodigy Optimizer 재개 - 기존 Prodigy checkpoint에서 이어서 학습
set -euo pipefail

# Activate virtual environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

echo "================================================================"
echo "🔄 Prodigy Optimizer 학습 재개"
echo "================================================================"
echo ""
echo "📊 실험 설정:"
echo "  - Optimizer: Prodigy (parameter-free, auto LR)"
echo "  - Resume from: latest.pth (Prodigy checkpoint)"
echo "  - Optimizer state 유지하여 재개"
echo ""

SCRIPT_DIR="/mnt/sdc1/ws/workspace/monorepo/external/index-tts"

cd "${SCRIPT_DIR}"

# Check if latest.pth exists
CKPT_PATH="${CKPT_PATH:-/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth}"
if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "❌ Error: ${CKPT_PATH} not found!"
    echo "   Use train_ko_prodigy.sh to start fresh training first."
    exit 1
fi

SKIP_DATA_CHECK=1 \
OPTIMIZER=prodigy \
BATCH_SIZE=8 \
GRAD_ACC=1 \
AMP=1 \
LOG_INTERVAL=200 \
VAL_INTERVAL=10000 \
MAX_STEPS=240000 \
EPOCHS=999 \
NUM_WORKERS=12 \
RESUME="${CKPT_PATH}" \
"${SCRIPT_DIR}/tools/ko_step4_train_gpt.sh" --no-aim --scheduler none
