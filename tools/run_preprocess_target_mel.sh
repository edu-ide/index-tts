#!/usr/bin/env bash
# 추가 mel 전처리 실행 스크립트
# mel이 없는 샘플에 대해 target_audio에서 mel을 추출합니다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/.venv/bin/activate"

# 설정
DATA_DIR="${DATA_DIR:-/mnt/sdb1/emilia-yodas/KO_preprocessed}"
MANIFEST="${MANIFEST:-${DATA_DIR}/gpt_pairs_train.jsonl}"
NUM_WORKERS="${NUM_WORKERS:-32}"
BATCH_SIZE="${BATCH_SIZE:-100000}"
START_FROM="${START_FROM:-0}"
LIMIT="${LIMIT:-}"  # 비어있으면 전체 처리

echo "================================================================"
echo "추가 mel 전처리 (target_audio → mel)"
echo "================================================================"
echo ""
echo "데이터 디렉토리: ${DATA_DIR}"
echo "Manifest: ${MANIFEST}"
echo "Workers: ${NUM_WORKERS}"
echo "Batch size: ${BATCH_SIZE}"
echo "시작 위치: ${START_FROM}"
echo "제한: ${LIMIT:-전체}"
echo ""

# 현재 mel 파일 수
echo "=== 현재 mel 상태 ==="
MEL_COUNT=$(ls "${DATA_DIR}/mel/" 2>/dev/null | wc -l)
TOTAL_SAMPLES=$(wc -l < "${MANIFEST}")
echo "mel 파일: ${MEL_COUNT}개"
echo "전체 샘플: ${TOTAL_SAMPLES}개"
echo "필요한 추가 전처리: $((TOTAL_SAMPLES - MEL_COUNT))개"
echo ""
echo "================================================================"
echo ""

# 실행
ARGS=(
    --manifest "${MANIFEST}"
    --data-dir "${DATA_DIR}"
    --num-workers "${NUM_WORKERS}"
    --batch-size "${BATCH_SIZE}"
    --start-from "${START_FROM}"
)

if [[ -n "${LIMIT}" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi

PYTHONUNBUFFERED=1 python "${SCRIPT_DIR}/preprocess_target_mel.py" "${ARGS[@]}"

echo ""
echo "================================================================"
echo "mel 전처리 완료!"
echo "================================================================"
echo ""
echo "다음 단계:"
echo "  1. manifest 재생성: bash tools/generate_mel_manifest_ssd.sh"
echo "  2. 학습 재시작: bash tools/run_stage2_training.sh"
echo ""
