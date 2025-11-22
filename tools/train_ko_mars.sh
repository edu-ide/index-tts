#!/usr/bin/env bash
# MARS Optimizer + WSD Scheduler 실험 - Step 0부터 완전 새로 시작
set -euo pipefail

# Activate virtual environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "🚀 MARS Optimizer + WSD Scheduler 완전 새로 시작"
echo "================================================================"
echo ""
echo "📊 실험 설정:"
echo "  - Optimizer: MARS (Variance Reduction)"
echo "  - Scheduler: WSD (Warmup-Stable-Decay)"
echo "  - Learning Rate: 6e-3 (권장값)"
echo "  - Starting from: Step 0 (fresh start)"
echo "  - 목표: Step 240k까지 안정적이고 빠른 학습"
echo ""

SCRIPT_DIR="/mnt/sdc1/ws/workspace/monorepo/external/index-tts"

cd "${SCRIPT_DIR}"

SKIP_DATA_CHECK=1 \
OPTIMIZER=mars \
SCHEDULER=wsd \
LR=6e-3 \
WSD_STABLE_RATIO=0.9 \
BATCH_SIZE=8 \
GRAD_ACC=1 \
LOG_INTERVAL=100 \
VAL_INTERVAL=10000 \
MAX_STEPS=240000 \
EPOCHS=999 \
NUM_WORKERS=32 \
"${SCRIPT_DIR}/tools/ko_step4_train_gpt.sh"
