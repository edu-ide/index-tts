#!/bin/bash

# IndexTTS WebUI 서버 시작 스크립트
# 사용법: ./start_webui.sh

cd "$(dirname "$0")"

# 가상환경 활성화
source .venv/bin/activate

echo "=========================================="
echo "IndexTTS WebUI 서버 시작 중..."
echo "=========================================="
echo ""
echo "GPU: RTX 3060 (12GB) 사용"
echo "모델 디렉토리: ~/models/index-tts-ko/checkpoints"
echo "모델 크기: 3.3GB (추론용)"
echo "서버 주소: http://0.0.0.0:7860"
echo ""
echo "모델 로딩 중... (약 1-2분 소요)"
echo ""

CUDA_VISIBLE_DEVICES=1 python webui.py \
  --host 0.0.0.0 \
  --port 7860 \
  --model_dir ~/models/index-tts-ko/checkpoints
