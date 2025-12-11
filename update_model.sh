#!/bin/bash

# IndexTTS 모델 업데이트 스크립트
# NFS에서 최신 best_model.pth를 복사하고 추론용으로 추출합니다.
# 사용법: ./update_model.sh

set -e  # 에러 발생시 중단

cd "$(dirname "$0")"

SOURCE_MODEL="/mnt/models/index-tts-ko/checkpoints/best_model.pth"
TARGET_DIR="$HOME/models/index-tts-ko/checkpoints"
TARGET_MODEL="$TARGET_DIR/best_model.pth"
INFERENCE_MODEL="$TARGET_DIR/best_model_inference.pth"
GPT_LINK="$TARGET_DIR/gpt.pth"

echo "=========================================="
echo "IndexTTS 모델 업데이트"
echo "=========================================="
echo ""

# 1. NFS 모델 존재 확인
if [ ! -f "$SOURCE_MODEL" ]; then
    echo "❌ 소스 모델을 찾을 수 없습니다: $SOURCE_MODEL"
    exit 1
fi

# 2. 타겟 디렉토리 생성
mkdir -p "$TARGET_DIR"

# 3. 모델 복사
echo "📥 1/3: NFS에서 모델 복사 중..."
echo "    소스: $SOURCE_MODEL"
echo "    목적지: $TARGET_MODEL"
echo ""

# 복사 전 타임스탬프 확인
if [ -f "$TARGET_MODEL" ]; then
    SOURCE_TIME=$(stat -c %Y "$SOURCE_MODEL")
    TARGET_TIME=$(stat -c %Y "$TARGET_MODEL")

    if [ "$SOURCE_TIME" -le "$TARGET_TIME" ]; then
        echo "ℹ️  로컬 모델이 이미 최신입니다. 복사를 건너뜁니다."
        SKIP_COPY=1
    fi
fi

if [ -z "$SKIP_COPY" ]; then
    cp "$SOURCE_MODEL" "$TARGET_MODEL"
    echo "✅ 모델 복사 완료"
else
    echo "✅ 복사 단계 스킵됨"
fi

echo ""

# 4. 추론용 모델 추출
echo "🔧 2/3: 추론용 모델 추출 중..."
echo "    입력: $TARGET_MODEL"
echo "    출력: $INFERENCE_MODEL"
echo ""

uv run python extract_inference_model.py \
    "$TARGET_MODEL" \
    "$INFERENCE_MODEL"

echo ""
echo "✅ 추론용 모델 추출 완료"
echo ""

# 5. 심볼릭 링크 업데이트
echo "🔗 3/3: gpt.pth 심볼릭 링크 업데이트 중..."

rm -f "$GPT_LINK"
ln -sf best_model_inference.pth "$GPT_LINK"

echo "✅ 심볼릭 링크 업데이트 완료"
echo ""

# 6. 최종 상태 확인
echo "=========================================="
echo "📊 최종 상태"
echo "=========================================="
ls -lh "$TARGET_DIR"/{best_model.pth,best_model_inference.pth,gpt.pth} 2>/dev/null || true
echo ""
echo "✅ 모델 업데이트 완료!"
echo ""
echo "💡 서버를 재시작하려면:"
echo "   - WebUI: ./start_webui.sh"
echo "   - API: ./start_api.sh"
echo ""
