#!/usr/bin/env bash
# Stage 2 GRL 모델로 추론하는 스크립트
# Usage: bash tools/infer_stage2.sh [text] [ref_audio] [output]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate venv
source "${PROJECT_ROOT}/.venv/bin/activate"

# Stage 2 모델 경로
STAGE2_DIR="${STAGE2_DIR:-/mnt/sda1/models/index-tts-ko/stage2}"
CHECKPOINT="${CHECKPOINT:-${STAGE2_DIR}/latest.pth}"
CONFIG="${CONFIG:-/mnt/sda1/models/index-tts-ko/checkpoints/config.yaml}"
MODEL_DIR="${MODEL_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"

# 기본값
TEXT="${1:-안녕하세요. 학습중 생성된 샘플입니다.}"
REF_AUDIO="${2:-${PROJECT_ROOT}/examples/voice_01.wav}"
OUTPUT="${3:-/tmp/stage2_output.wav}"

echo "================================================================"
echo "Stage 2 GRL 모델 추론"
echo "================================================================"
echo ""
echo "Checkpoint: ${CHECKPOINT}"
echo "Config: ${CONFIG}"
echo "Model Dir: ${MODEL_DIR}"
echo ""
echo "Text: ${TEXT}"
echo "Reference Audio: ${REF_AUDIO}"
echo "Output: ${OUTPUT}"
echo ""

# 체크포인트 확인
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "❌ 체크포인트 파일을 찾을 수 없습니다: ${CHECKPOINT}"
    echo ""
    echo "사용 가능한 체크포인트:"
    ls -la "${STAGE2_DIR}"/*.pth 2>/dev/null || echo "  (없음)"
    exit 1
fi

# 체크포인트 정보 출력
echo "=== 체크포인트 정보 ==="
python -c "
import torch
ckpt = torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False)
epoch = ckpt.get('epoch', 'N/A')
step = ckpt.get('global_step', 'N/A')
best_loss = ckpt.get('best_val_loss', 'N/A')
print(f'  Epoch: {epoch}, Step: {step}, Best Val Loss: {best_loss}')
"
echo ""

# 추론 실행 (CPU)
echo "=== 추론 시작 ==="
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/infer_cpu_sample.py" \
    --ckpt "${CHECKPOINT}" \
    --text "${TEXT}" \
    --ref-audio "${REF_AUDIO}" \
    --output "${OUTPUT}" \
    --model-dir "${MODEL_DIR}" \
    --config "${CONFIG}"

echo ""
echo "================================================================"
echo "✅ 추론 완료!"
echo "Output: ${OUTPUT}"
echo "================================================================"
