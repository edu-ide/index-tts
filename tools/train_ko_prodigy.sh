#!/usr/bin/env bash
# Prodigy Optimizer 실험 - Step 0부터 완전 새로 시작
set -euo pipefail

# Activate virtual environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

echo "================================================================"
echo "🚀 Prodigy Optimizer 완전 새로 시작"
echo "================================================================"
echo ""
echo "📊 실험 설정:"
echo "  - Optimizer: Prodigy (parameter-free, auto LR)"
echo "  - Starting from: Step 0 (fresh start)"
echo "  - No resume - clean training with Prodigy"
echo "  - 목표: Step 240k까지 자동 LR 조정"
echo ""

SCRIPT_DIR="/mnt/sdc1/ws/workspace/monorepo/external/index-tts"

cd "${SCRIPT_DIR}"

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
"${SCRIPT_DIR}/tools/ko_step4_train_gpt.sh" --no-aim --scheduler none
