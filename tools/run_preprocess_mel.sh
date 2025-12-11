#!/usr/bin/env bash
# Preprocess mel-spectrograms for Stage 2 GRL training (논문 방식)
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DATA_DIR="${DATA_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
NUM_WORKERS="${NUM_WORKERS:-32}"

echo "================================================================"
echo "🎵 Mel-Spectrogram 전처리 (Stage 2 논문 방식)"
echo "================================================================"
echo ""
echo "📂 Data directory: ${DATA_DIR}"
echo "👷 Workers: ${NUM_WORKERS}"
echo ""
echo "📊 Mel config:"
echo "  - Sample rate: 22050"
echo "  - n_mels: 80"
echo "  - n_fft: 1024"
echo "  - hop_length: 256"
echo ""

# Train manifest
echo "================================================================"
echo "📚 [1/2] Train manifest 전처리"
echo "================================================================"
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DATA_DIR}/gpt_pairs_train.jsonl" \
    --output-manifest "${DATA_DIR}/gpt_pairs_train_mel.jsonl" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""

# Val manifest
echo "================================================================"
echo "📚 [2/2] Val manifest 전처리"
echo "================================================================"
python "${PROJECT_ROOT}/tools/preprocess_mel_for_stage2.py" \
    --input-manifest "${DATA_DIR}/gpt_pairs_val.jsonl" \
    --output-manifest "${DATA_DIR}/gpt_pairs_val_mel.jsonl" \
    --data-dir "${DATA_DIR}" \
    --num-workers "${NUM_WORKERS}"

echo ""
echo "================================================================"
echo "✅ Mel 전처리 완료!"
echo "================================================================"
echo ""
echo "📁 생성된 파일:"
echo "  - ${DATA_DIR}/mel/ (mel-spectrogram .npy 파일)"
echo "  - ${DATA_DIR}/gpt_pairs_train_mel.jsonl"
echo "  - ${DATA_DIR}/gpt_pairs_val_mel.jsonl"
echo ""
echo "🚀 다음 단계: Stage 2 학습"
echo "   ./tools/train_ko_stage2.sh"
echo ""
