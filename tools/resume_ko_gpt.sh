#!/usr/bin/env bash
# Resume Korean GPT fine-tuning from the latest checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
RESUME_PATH="${RESUME_PATH:-${CHECKPOINT_DIR}/latest.pth}"
DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
RAW_MANIFEST="${RAW_MANIFEST:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"
# Worker profile: default=auto, low=NUM_WORKERS=0,etc.
WORKER_PROFILE="${1:-auto}"
if [[ $# -gt 0 ]]; then
  shift
fi

SKIP_DATA_CHECK="${SKIP_DATA_CHECK:-1}"

if [[ ! -f "${RESUME_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${RESUME_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_DATA_CHECK}" != "1" ]]; then
  echo "[Resume] Running manifest/data consistency check before training..." >&2
  DATASET_DIR="${DATASET_DIR}" RAW_MANIFEST="${RAW_MANIFEST}" \
    "${SCRIPT_DIR}/ko_step2_fix_broken.sh" --scan-empty --scan-missing
else
  echo "[Resume] Skipping data consistency check (SKIP_DATA_CHECK=1)." >&2
fi

case "${WORKER_PROFILE}" in
  low)
    export NUM_WORKERS=0
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
    export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-2}"
    echo "[Resume] Worker profile=low (NUM_WORKERS=0, OMP/MKL/TORCH threads<=2)" >&2
    ;;
  med)
    export NUM_WORKERS=2
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
    echo "[Resume] Worker profile=med (NUM_WORKERS=2, threads<=4)" >&2
    ;;
  auto|*)
    echo "[Resume] Worker profile=auto (respect existing env)" >&2
    ;;
esac

echo "[Resume] Using checkpoint ${RESUME_PATH}" >&2
RESUME="${RESUME_PATH}" "${SCRIPT_DIR}/ko_step4_train_gpt.sh" "$@"
