#!/usr/bin/env bash
# Phase 1: 초저 LR로 5000 step 검증
# 목적: step 351,000 체크포인트가 초저 learning rate로 회복 가능한지 확인

set -euo pipefail

echo "================================================================"
echo "Phase 1: 초저 LR 검증 (5000 step)"
echo "================================================================"
echo ""
echo "현재 상황:"
echo "  - 기존 학습: step 438,000까지 진행했으나 loss 폭발"
echo "  - 최선 지점: step 298,800 (text_loss: 0.94)"
echo "  - 사용 체크포인트: step 351,000 (유일한 백업)"
echo ""
echo "검증 전략:"
echo "  - LR: 1e-6 (기존 2e-5의 1/20)"
echo "  - Warmup: 3000 step"
echo "  - Max steps: 5000 step (약 1-2시간)"
echo "  - Gradient clip: 0.5 (기존 1.0의 절반)"
echo ""
echo "판단 기준:"
echo "  ✅ 5000 step 후 loss 감소 → Phase 2 계속 학습"
echo "  ❌ 5000 step 후 loss 증가 → Base 모델부터 재학습"
echo ""
echo "================================================================"

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="/mnt/sda1/models/index-tts-ko/checkpoints/model_step351000 (사본).pth"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] 체크포인트를 찾을 수 없습니다: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

echo "체크포인트: ${CHECKPOINT_PATH}"
echo "TensorBoard: http://localhost:6006"
echo ""
echo "학습을 시작합니다..."
echo ""

# Best checkpoint 모니터링 시작 (백그라운드)
nohup python3 << 'PYEOF' > /tmp/phase1_best_monitor.log 2>&1 &
import time, shutil
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

log_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints/logs")
ckpt_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints")
best_ckpt = ckpt_dir / "phase1_best.pth"

best_loss = float('inf')
last_step = -1

while True:
    try:
        runs = sorted(log_dir.glob("run_*"))
        if not runs: time.sleep(30); continue

        ea = event_accumulator.EventAccumulator(str(runs[-1]))
        ea.Reload()

        tag = 'train/mel_loss'
        if tag not in ea.Tags()['scalars']: time.sleep(30); continue

        events = ea.Scalars(tag)
        if not events: time.sleep(30); continue

        latest = events[-1]
        if latest.step == last_step: time.sleep(30); continue

        last_step = latest.step
        if latest.value < best_loss:
            best_loss = latest.value
            step_ckpt = ckpt_dir / f"model_step{(latest.step // 1000) * 1000}.pth"
            if step_ckpt.exists():
                shutil.copy2(step_ckpt, best_ckpt)
                print(f"New best: {best_loss:.4f}")

        time.sleep(30)
    except: time.sleep(30)
PYEOF

MONITOR_PID=$!
echo "Best checkpoint monitor started (PID: ${MONITOR_PID})"

# Phase 1 학습 시작
# VAL_INTERVAL=500 추가 - validation loss 확인
# SKIP_DATA_CHECK=1 - 데이터 검증 스킵 (이미 여러 번 확인함)
SKIP_DATA_CHECK=1 \
LR=1e-6 \
WARMUP_STEPS=3000 \
MAX_STEPS=5000 \
GRAD_CLIP=0.5 \
BATCH_SIZE=4 \
LOG_INTERVAL=50 \
VAL_INTERVAL=500 \
RESUME="${CHECKPOINT_PATH}" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

# 학습 완료 후 모니터 종료
kill ${MONITOR_PID} 2>/dev/null || true

echo ""
echo "================================================================"
echo "Phase 1 완료!"
echo "================================================================"
echo ""
echo "다음 단계:"
echo "  1. TensorBoard 확인: http://localhost:6006"
echo "  2. 최근 5000 step의 loss 추이 분석"
echo "  3. 결정:"
echo "     - loss 감소 → ./tools/phase2_continue.sh 실행"
echo "     - loss 증가 → ./tools/restart_from_base.sh 실행"
echo ""
