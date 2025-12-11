#!/usr/bin/env bash
# Stage 2 GRL Training (mel preprocessing already done)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate venv
source "${PROJECT_ROOT}/.venv/bin/activate"

# Dataset (SSD for faster loading)
DATASET_DIR="${DATASET_DIR:-/mnt/sdb1/emilia-yodas/KO_preprocessed}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_DIR}/gpt_pairs_train_mel.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_val_10k_mel.jsonl}"

# Model paths
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth}"
SPEAKER_MAPPING="${SPEAKER_MAPPING:-/mnt/sda1/models/index-tts-ko/speaker_mapping_from_manifest.json}"
CONFIG="${CONFIG:-/mnt/sda1/models/index-tts-ko/checkpoints/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

# Hyperparameters
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
EPOCHS="${EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
GRL_LAMBDA="${GRL_LAMBDA:-1.0}"
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAX_STEPS="${MAX_STEPS:-0}"
RESUME="${RESUME:-auto}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "Stage 2 GRL Training"
echo "================================================================"
echo ""
echo "Train: ${TRAIN_MANIFEST}"
echo "Val: ${VAL_MANIFEST}"
echo "Speaker Mapping: ${SPEAKER_MAPPING}"
echo "Output: ${STAGE2_DIR}"
echo ""
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACC} = $((BATCH_SIZE * GRAD_ACC))"
echo "LR: ${LR}, GRL Lambda: ${GRL_LAMBDA}"
echo "Max Steps: ${MAX_STEPS}"
echo "Resume: ${RESUME}"
echo "================================================================"
echo ""

# Show checkpoint status before training
echo "=== Checkpoint Status ==="
if [[ -f "${STAGE2_DIR}/latest_full.pth" ]]; then
    echo "Found: latest_full.pth ($(stat -c %y "${STAGE2_DIR}/latest_full.pth" | cut -d. -f1))"
    python -c "
import torch
ckpt = torch.load('${STAGE2_DIR}/latest_full.pth', map_location='cpu', weights_only=False)
epoch = ckpt.get('epoch', 0)
step = ckpt.get('global_step', 0)
best_loss = ckpt.get('best_val_loss', 'N/A')
print(f'  → Epoch: {epoch}, Step: {step}, Best Val Loss: {best_loss}')
"
elif [[ -f "${STAGE2_DIR}/latest.pth" ]]; then
    echo "Found: latest.pth ($(stat -c %y "${STAGE2_DIR}/latest.pth" | cut -d. -f1))"
    python -c "
import torch
ckpt = torch.load('${STAGE2_DIR}/latest.pth', map_location='cpu', weights_only=False)
epoch = ckpt.get('epoch', 0)
step = ckpt.get('global_step', 0)
best_loss = ckpt.get('best_val_loss', 'N/A')
print(f'  → Epoch: {epoch}, Step: {step}, Best Val Loss: {best_loss}')
"
else
    echo "No checkpoint found. Starting fresh from Stage 1."
fi
echo ""

PYTHONUNBUFFERED=1 python "${PROJECT_ROOT}/trainers/train_gpt_v2.py" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --val-manifest "${VAL_MANIFEST}" \
    --tokenizer "${TOKENIZER}" \
    --config "${CONFIG}" \
    --base-checkpoint "${STAGE1_CHECKPOINT}" \
    --output-dir "${STAGE2_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accumulation "${GRAD_ACC}" \
    --learning-rate "${LR}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --epochs "${EPOCHS}" \
    --max-steps "${MAX_STEPS}" \
    --grad-clip "${GRAD_CLIP}" \
    --enable-grl \
    --speaker-mapping "${SPEAKER_MAPPING}" \
    --grl-lambda "${GRL_LAMBDA}" \
    --speaker-loss-weight "${SPEAKER_LOSS_WEIGHT}" \
    --log-interval "${LOG_INTERVAL}" \
    --val-interval "${VAL_INTERVAL}" \
    --num-workers "${NUM_WORKERS}" \
    --resume "${RESUME}"

echo ""
echo "================================================================"
echo "Stage 2 Training Complete!"
echo "================================================================"
