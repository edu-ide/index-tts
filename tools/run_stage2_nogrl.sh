#!/usr/bin/env bash
# Stage 2 Sanity Check - No GRL (GRL 없이 순수 Fine-tuning)
# GRL이 멜 손실(발음/음질) 저하의 원인인지 확인하기 위한 테스트
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/.venv/bin/activate"

# Dataset (SSD for faster loading)
DATASET_DIR="${DATASET_DIR:-/mnt/sdb1/emilia-yodas/KO_preprocessed}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATASET_DIR}/gpt_pairs_train_mel.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_val_10k_mel.jsonl}"

# Model paths - Stage 1 Best 사용
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2_nogrl}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth}"
CONFIG="${CONFIG:-/mnt/sda1/models/index-tts-ko/checkpoints/config.yaml}"
TOKENIZER="${TOKENIZER:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"

# Hyperparameters - 비교를 위해 Low LR 설정 유지
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EPOCHS="${EPOCHS:-1}" # 테스트용이므로 짧게 설정하거나 모니터링 후 중단
GRAD_CLIP="${GRAD_CLIP:-0.5}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-32}"
MAX_STEPS="${MAX_STEPS:-0}"
RESUME="${RESUME:-none}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================"
echo "Stage 2 Sanity Check (NO GRL)"
echo "================================================================"
echo "목표: GRL을 끄고 학습했을 때 Mel Loss가 5.0 미만으로 떨어지는지 확인"
echo ""
echo "  - Enable GRL: FALSE (OFF)"
echo "  - Output: ${STAGE2_DIR}"
echo "  - LR: ${LR}"
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
    --log-interval "${LOG_INTERVAL}" \
    --val-interval "${VAL_INTERVAL}" \
    --num-workers "${NUM_WORKERS}" \
    --resume "${RESUME}"
    # --enable-grl 제거됨
    # --speaker-mapping 제거됨

echo ""
echo "================================================================"
echo "No-GRL Training Check Complete!"
echo "================================================================"
