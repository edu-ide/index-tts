#!/usr/bin/env bash
# Stage 2 GRL Training - Low LR Version (안정적인 학습)
# 기존 LR=2e-4가 너무 높아서 모델이 손상됨
# 권장: LR=1e-5 ~ 5e-5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/.venv/bin/activate"

# Dataset (SSD for faster loading)
DATASET_DIR="${DATASET_DIR:-/mnt/sdb1/emilia-yodas/KO_preprocessed}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_DIR}/gpt_pairs_train_mel.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_val_10k_mel.jsonl}"

# Model paths
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2_lowlr}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth}"
SPEAKER_MAPPING="${SPEAKER_MAPPING:-/mnt/sda1/models/index-tts-ko/speaker_mapping_full.json}"
CONFIG="${CONFIG:-/mnt/sda1/models/index-tts-ko/checkpoints/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

# Hyperparameters - 안정적인 설정
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-5}"              # 2e-4 → 2e-5 (10배 낮춤)
WARMUP_STEPS="${WARMUP_STEPS:-2000}"  # 1000 → 2000 (더 긴 warmup)
EPOCHS="${EPOCHS:-1}"
GRAD_CLIP="${GRAD_CLIP:-0.5}"
GRL_LAMBDA="${GRL_LAMBDA:-0.5}"       # 1.0 → 0.5 (GRL 강도 낮춤)
SPEAKER_LOSS_WEIGHT="${SPEAKER_LOSS_WEIGHT:-0.1}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAX_STEPS="${MAX_STEPS:-0}"
RESUME="${RESUME:-none}"      # 새로 시작

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "Stage 2 GRL Training (Low LR - 안정 버전)"
echo "================================================================"
echo ""
echo "⚠️  이전 LR=2e-4 학습에서 모델이 손상되어 새로 시작합니다"
echo ""
echo "변경된 설정:"
echo "  - LR: 2e-4 → ${LR} (10배 낮춤)"
echo "  - GRL_LAMBDA: 1.0 → ${GRL_LAMBDA}"
echo "  - WARMUP_STEPS: 1000 → ${WARMUP_STEPS}"
echo "  - 출력 디렉토리: ${STAGE2_DIR} (별도)"
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

# 출력 디렉토리 생성
mkdir -p "${STAGE2_DIR}"

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
echo "Stage 2 Training (Low LR) Complete!"
echo "================================================================"
