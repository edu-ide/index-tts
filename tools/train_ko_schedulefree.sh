#!/usr/bin/env bash
# Schedule-Free AdamW - Step 0부터 완전 새로 시작
set -euo pipefail

# Activate virtual environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "🚀 Schedule-Free AdamW 완전 새로 시작"
echo "================================================================"
echo ""
echo "📊 실험 설정:"
echo "  - Optimizer: Schedule-Free AdamW (3-4x faster than Prodigy)"
echo "  - Learning Rate: 5e-4 (권장값)"
echo "  - No LR Scheduler needed (built-in warmup)"
echo "  - Starting from: Step 0 (fresh start)"
echo "  - 목표: Step 240k까지 빠르고 안정적인 학습"
echo ""

SCRIPT_DIR="/mnt/sdc1/ws/workspace/monorepo/external/index-tts"

cd "${SCRIPT_DIR}"

SKIP_DATA_CHECK=1 \
OPTIMIZER=schedulefree \
LR=5e-4 \
BATCH_SIZE=8 \
GRAD_ACC=1 \
LOG_INTERVAL=100 \
VAL_INTERVAL=10000 \
MAX_STEPS=240000 \
EPOCHS=999 \
NUM_WORKERS=16 \
"${SCRIPT_DIR}/tools/ko_step4_train_gpt.sh"
