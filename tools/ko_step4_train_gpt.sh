#!/usr/bin/env bash
# Step 4: Fine-tune the GPT component on the Korean GPT pair manifests.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경을 활성화하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl::ko}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_val.jsonl::ko}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"
BASE_TOKENIZER_MODEL="${BASE_TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/bpe.model}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/mnt/sda1/models/IndexTTS-2/gpt.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-0.2}"
MEL_LOSS_WEIGHT="${MEL_LOSS_WEIGHT:-0.8}"
AMP="${AMP:-0}"
SEED="${SEED:-1234}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FLAG="${RESUME:-}"

IFS=',' read -ra TRAIN_ARRAY <<< "${TRAIN_MANIFEST}"
TRAIN_FLAGS=()
for entry in "${TRAIN_ARRAY[@]}"; do
  trimmed="$(echo "${entry}" | xargs)"
  [[ -z "${trimmed}" ]] && continue
  TRAIN_FLAGS+=(--train-manifest "${trimmed}")
done

IFS=',' read -ra VAL_ARRAY <<< "${VAL_MANIFEST}"
VAL_FLAGS=()
for entry in "${VAL_ARRAY[@]}"; do
  trimmed="$(echo "${entry}" | xargs)"
  [[ -z "${trimmed}" ]] && continue
  VAL_FLAGS+=(--val-manifest "${trimmed}")
done

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/../trainers/train_gpt_v2.py")
CMD+=("${TRAIN_FLAGS[@]}")
CMD+=("${VAL_FLAGS[@]}")
CMD+=(
  --tokenizer "${TOKENIZER_MODEL}"
  --base-tokenizer "${BASE_TOKENIZER_MODEL}"
  --config "${CONFIG_PATH}"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation "${GRAD_ACC}"
  --epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-steps "${WARMUP_STEPS}"
  --max-steps "${MAX_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --val-interval "${VAL_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --grad-clip "${GRAD_CLIP}"
  --text-loss-weight "${TEXT_LOSS_WEIGHT}"
  --mel-loss-weight "${MEL_LOSS_WEIGHT}"
  --seed "${SEED}"
)

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
fi

if [[ -n "${RESUME_FLAG}" ]]; then
  CMD+=(--resume "${RESUME_FLAG}")
fi

echo "[KO-STEP4] train manifests=${TRAIN_FLAGS[*]}"
echo "[KO-STEP4] val manifests=${VAL_FLAGS[*]}"
echo "[KO-STEP4] output-dir=${OUTPUT_DIR}"

if [[ "${TIMEOUT_SECS}" -gt 0 ]]; then
  timeout "${TIMEOUT_SECS}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "[KO-STEP4] GPT fine-tuning command finished."
