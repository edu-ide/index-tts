#!/usr/bin/env bash
# Execute the Korean tokenizer → preprocessing → pair generation pipeline.
# All heavy steps use `timeout` to honour the non-blocking command policy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEXTTS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${INDEXTTS_ROOT}/.." && pwd)"

# Default paths (override via environment variables before invoking the script).
MANIFEST_PATH="${MANIFEST_PATH:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"
MODEL_DIR="${MODEL_DIR:-/mnt/sda1/models/IndexTTS-2}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MODEL_DIR}/tokenizer_ko}"
PREPROCESS_DIR="${PREPROCESS_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
PAIRS_OUTPUT="${PAIRS_OUTPUT:-/mnt/sda1/emilia-yodas/KO_gpt_pairs.jsonl}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.01}"
LANG_HINT="${LANG_HINT:-ko}"

# Timeouts in seconds.
TOKENIZER_TIMEOUT="${TOKENIZER_TIMEOUT:-3600}"
PREPROCESS_TIMEOUT="${PREPROCESS_TIMEOUT:-0}"  # 0 = no timeout
PAIRS_TIMEOUT="${PAIRS_TIMEOUT:-1800}"

# Batch / worker knobs for preprocessing.
PREPROCESS_BATCH="${PREPROCESS_BATCH:-1}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-0}"

# Toggle steps (1 = run, 0 = skip).
RUN_EXTEND="${RUN_EXTEND:-1}"
RUN_PREPROCESS="${RUN_PREPROCESS:-1}"
RUN_GENERATE_PAIRS="${RUN_GENERATE_PAIRS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_dir() {
    mkdir -p "$1"
}

compute_target_vocab() {
    local base_model="$1"
    local increment="${2:-4000}"
    "${PYTHON_BIN}" - <<'PY' "$base_model" "$increment"
import sys
from pathlib import Path
from sentencepiece import sentencepiece_model_pb2 as sp_model

base_model = Path(sys.argv[1])
increment = int(sys.argv[2])
proto = sp_model.ModelProto()
proto.ParseFromString(base_model.read_bytes())
base_vocab = len(proto.pieces)
print(base_vocab + increment)
PY
}

extend_tokenizer() {
    local base_model="${MODEL_DIR}/bpe.model"
    local output_model="${TOKENIZER_DIR}/bpe.model"
    local output_vocab="${TOKENIZER_DIR}/bpe.vocab"

    ensure_dir "${TOKENIZER_DIR}"

    local target_vocab="${TARGET_VOCAB:-}"
    if [[ -z "${target_vocab}" ]]; then
        log "TARGET_VOCAB not provided; estimating base vocab + 4000."
        target_vocab="$(compute_target_vocab "${base_model}" "${TARGET_INCREMENT:-4000}")"
        log "Computed target vocab size: ${target_vocab}"
    fi

    local cmd=(
        "${PYTHON_BIN}" "${INDEXTTS_ROOT}/tools/tokenizer/extend_bpe.py"
        --base-model "${base_model}"
        --manifests "${MANIFEST_PATH}"
        --output-model "${output_model}"
        --output-vocab "${output_vocab}"
        --target-size "${target_vocab}"
    )

    log "Extending tokenizer → ${output_model} (target vocab ${target_vocab})"
    if [[ "${TOKENIZER_TIMEOUT}" -gt 0 ]]; then
        "${TIMEOUT_BIN}" "${TOKENIZER_TIMEOUT}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
    log "Tokenizer extension complete."
}

preprocess_dataset() {
    ensure_dir "${PREPROCESS_DIR}"

    local cmd=(
        "${PYTHON_BIN}" "${INDEXTTS_ROOT}/tools/preprocess_data.py"
        --manifest "${MANIFEST_PATH}"
        --output-dir "${PREPROCESS_DIR}"
        --tokenizer "${TOKENIZER_DIR}/bpe.model"
        --config "${MODEL_DIR}/config.yaml"
        --gpt-checkpoint "${MODEL_DIR}/gpt.pth"
        --language "${LANG_HINT}"
        --batch-size "${PREPROCESS_BATCH}"
        --workers "${PREPROCESS_WORKERS}"
        --val-ratio "${VALIDATION_SPLIT}"
    )

    log "Starting preprocessing → ${PREPROCESS_DIR}"
    if [[ "${PREPROCESS_TIMEOUT}" -gt 0 ]]; then
        "${TIMEOUT_BIN}" "${PREPROCESS_TIMEOUT}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
    log "Preprocessing complete."
}

generate_pairs() {
    local cmd=(
        "${PYTHON_BIN}" "${INDEXTTS_ROOT}/tools/generate_gpt_pairs.py"
        --input-dir "${PREPROCESS_DIR}"
        --output "${PAIRS_OUTPUT}"
    )

    log "Generating GPT prompt/target pairs → ${PAIRS_OUTPUT}"
    if [[ "${PAIRS_TIMEOUT}" -gt 0 ]]; then
        "${TIMEOUT_BIN}" "${PAIRS_TIMEOUT}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
    log "GPT pairs generation complete."
}

# Ensure Python sees the indextts package.
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${INDEXTTS_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${INDEXTTS_ROOT}"
fi

if [[ "${RUN_EXTEND}" -eq 1 ]]; then
    extend_tokenizer
else
    log "Skipping tokenizer extension (RUN_EXTEND=${RUN_EXTEND})."
fi

if [[ "${RUN_PREPROCESS}" -eq 1 ]]; then
    preprocess_dataset
else
    log "Skipping preprocessing (RUN_PREPROCESS=${RUN_PREPROCESS})."
fi

if [[ "${RUN_GENERATE_PAIRS}" -eq 1 ]]; then
    generate_pairs
else
    log "Skipping GPT pair generation (RUN_GENERATE_PAIRS=${RUN_GENERATE_PAIRS})."
fi

log "Pipeline finished."
