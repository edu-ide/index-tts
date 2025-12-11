#!/usr/bin/env bash
# Stage 1 vs Stage 2 추론 비교 스크립트
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/.venv/bin/activate"

TEXT="${1:-안녕하세요. 학습중 생성된 샘플입니다.}"
REF_AUDIO="${2:-${PROJECT_ROOT}/examples/voice_01.wav}"

echo "=== Stage 1 (best_model) 추론 ==="
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/infer_cpu_sample.py" \
    --ckpt /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth \
    --text "${TEXT}" \
    --ref-audio "${REF_AUDIO}" \
    --output /tmp/stage1_test.wav \
    --model-dir /mnt/sda1/models/index-tts-ko/checkpoints \
    --config /mnt/sda1/models/index-tts-ko/checkpoints/config.yaml

echo ""
echo "=== Stage 2 (latest) 추론 ==="
CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/infer_cpu_sample.py" \
    --ckpt /mnt/sda1/models/index-tts-ko/stage2/latest.pth \
    --text "${TEXT}" \
    --ref-audio "${REF_AUDIO}" \
    --output /tmp/stage2_test.wav \
    --model-dir /mnt/sda1/models/index-tts-ko/checkpoints \
    --config /mnt/sda1/models/index-tts-ko/checkpoints/config.yaml

echo ""
echo "=== 결과 ==="
echo "Stage 1: /tmp/stage1_test.wav"
echo "Stage 2: /tmp/stage2_test.wav"
