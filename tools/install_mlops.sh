#!/usr/bin/env bash
# MLOps 도구 설치 스크립트
# WandB, Whisper, Slack 모니터링에 필요한 패키지 설치

set -euo pipefail

echo "================================================================"
echo "🚀 MLOps 도구 설치"
echo "================================================================"
echo ""
echo "다음 패키지를 설치합니다:"
echo "  1. aim - Experiment tracking"
echo "  2. openai-whisper - TTS evaluation (ASR)"
echo "  3. jiwer - WER/CER calculation"
echo "  4. librosa - Audio processing"
echo "  5. requests - Slack integration"
echo ""
echo "================================================================"
echo ""

# 가상환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

echo "✅ 가상환경: ${VIRTUAL_ENV}"
echo ""

# 사용자 확인
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

echo ""
echo "📦 패키지 설치 중..."
echo ""

# 1. Aim (Experiment Tracking)
echo "1/5: Installing aim..."
uv pip install aim --quiet

# 2. Whisper (ASR for evaluation)
echo "2/5: Installing openai-whisper..."
uv pip install openai-whisper --quiet

# 3. jiwer (WER/CER calculation)
echo "3/5: Installing jiwer..."
uv pip install jiwer --quiet

# 4. librosa (Audio processing)
echo "4/5: Installing librosa..."
uv pip install librosa --quiet

# 5. requests (Slack integration)
echo "5/5: Installing requests..."
uv pip install requests --quiet

echo ""
echo "================================================================"
echo "✅ 설치 완료!"
echo "================================================================"
echo ""
echo "다음 단계:"
echo ""
echo "1. Aim UI 실행 (실험 결과 확인):"
echo "   aim up"
echo ""
echo "2. Slack Webhook 설정 (선택사항):"
echo "   export SLACK_WEBHOOK_URL=\"https://hooks.slack.com/services/...\""
echo ""
echo "3. 학습 시작 (Aim 자동 활성화):"
echo "   ./tools/train_ko_optimized_a6000.sh"
echo ""
echo "4. 모니터링 시작 (선택사항):"
echo "   nohup python tools/monitor_training.py > /tmp/monitor.log 2>&1 &"
echo ""
echo "5. 체크포인트 평가:"
echo "   python tools/evaluate_tts.py --checkpoint <path> --test-manifest <path>"
echo ""
echo "📖 자세한 사용법: MLOPS_GUIDE.md 참고"
echo ""
