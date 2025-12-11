#!/bin/bash

# IndexTTS API Server 시작 스크립트
# 사용법: ./start_api.sh [포트번호]
#
# 예시:
#   ./start_api.sh           # 기본값: 포트 8765
#   ./start_api.sh 8080      # 포트 8080

cd "$(dirname "$0")"

PORT=${1:-8765}
MODEL_DIR="${MODEL_DIR:-$HOME/models/index-tts-ko/checkpoints}"

# 기존 서버 프로세스 종료
EXISTING_PID=$(lsof -ti:$PORT)
if [ ! -z "$EXISTING_PID" ]; then
    echo "포트 $PORT에서 실행 중인 프로세스($EXISTING_PID)를 종료합니다..."
    kill -9 $EXISTING_PID
    sleep 1
fi

echo "=========================================="
echo "IndexTTS API 서버 v2.0"
echo "=========================================="
echo ""
echo "설정:"
echo "  - GPU: CUDA_VISIBLE_DEVICES=0 (RTX 3060 12GB, PCI_BUS_ID order)"
echo "  - 모델 디렉토리: $MODEL_DIR"
echo "  - 서버 주소: http://0.0.0.0:$PORT"
echo "  - API 문서: http://0.0.0.0:$PORT/docs"
echo ""
echo "사용 가능한 엔드포인트:"
echo "  - POST /tts          : 단일 TTS 생성"
echo "  - POST /tts/stream   : 스트리밍 TTS"
echo "  - POST /tts/batch    : 배치 TTS 처리"
echo "  - GET  /model/status : 모델 상태 조회"
echo "  - POST /model/load   : 체크포인트 변경"
echo "  - GET  /vram/status  : VRAM 상태 조회"
echo "  - POST /vram/unload  : VRAM 수동 해제"
echo ""
echo "VRAM 관리: 추론 완료 후 즉시 언로드 (재요청시 자동 로드)"
echo ""
echo "모델 로딩 중... (약 1-2분 소요)"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python api_server.py \
  --model_dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port $PORT
