#!/usr/bin/env bash
# Base 모델부터 재학습 (Best Checkpoint 저장 기능 추가)

set -euo pipefail

echo "================================================================"
echo "Base 모델부터 재학습 (Best Checkpoint 보존)"
echo "================================================================"
echo ""
echo "개선 사항:"
echo "  ✅ Best checkpoint 자동 저장"
echo "  ✅ 최근 5개 체크포인트 유지 (기존 3개 → 5개)"
echo "  ✅ Validation loss 기록 (VAL_INTERVAL=1000)"
echo ""
echo "================================================================"

# 환경 확인
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth"
BACKUP_DIR="/mnt/sda1/models/index-tts-ko/checkpoints_backup_$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "[ERROR] Base 체크포인트를 찾을 수 없습니다: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

# 기존 체크포인트 백업 확인
echo "⚠️  경고: 기존 학습 데이터를 백업합니다."
echo "백업 위치: ${BACKUP_DIR}"
echo ""
read -p "계속하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "취소되었습니다."
  exit 0
fi

# 백업
echo "기존 체크포인트 백업 중..."
mkdir -p "${BACKUP_DIR}"
cp -r /mnt/sda1/models/index-tts-ko/checkpoints/* "${BACKUP_DIR}/" || true
echo "백업 완료: ${BACKUP_DIR}"
echo ""

# Best checkpoint 모니터링 스크립트 생성
cat > /tmp/monitor_best_checkpoint.py << 'EOF'
#!/usr/bin/env python3
"""
Best checkpoint 모니터링 및 자동 저장
TensorBoard 로그를 실시간으로 체크하여 최고 성능 체크포인트를 별도 저장
"""
import time
import shutil
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

log_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints/logs")
ckpt_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints")
best_ckpt_path = ckpt_dir / "best_model.pth"
best_loss_file = ckpt_dir / "best_loss.txt"

# 초기 best loss
if best_loss_file.exists():
    with open(best_loss_file, 'r') as f:
        best_loss = float(f.read().strip())
else:
    best_loss = float('inf')

print(f"Best checkpoint monitor started (current best: {best_loss:.4f})")

def get_latest_run():
    runs = sorted(log_dir.glob("run_*"))
    return runs[-1] if runs else None

last_checked_step = -1

while True:
    try:
        latest_run = get_latest_run()
        if not latest_run:
            time.sleep(30)
            continue

        ea = event_accumulator.EventAccumulator(str(latest_run))
        ea.Reload()

        # Validation loss 확인 (없으면 train loss 사용)
        loss_tag = 'val/mel_loss' if 'val/mel_loss' in ea.Tags()['scalars'] else 'train/mel_loss'

        if loss_tag not in ea.Tags()['scalars']:
            time.sleep(30)
            continue

        events = ea.Scalars(loss_tag)
        if not events:
            time.sleep(30)
            continue

        latest_event = events[-1]

        if latest_event.step == last_checked_step:
            time.sleep(30)
            continue

        last_checked_step = latest_event.step
        current_loss = latest_event.value

        # Best 업데이트 확인
        if current_loss < best_loss:
            best_loss = current_loss

            # 해당 step의 체크포인트 찾기
            step_ckpt = ckpt_dir / f"model_step{latest_event.step}.pth"

            # 1000 step 단위로 저장되므로, 가장 가까운 step 찾기
            rounded_step = (latest_event.step // 1000) * 1000
            step_ckpt = ckpt_dir / f"model_step{rounded_step}.pth"

            if step_ckpt.exists():
                # Best checkpoint 복사
                shutil.copy2(step_ckpt, best_ckpt_path)

                # Best loss 저장
                with open(best_loss_file, 'w') as f:
                    f.write(f"{best_loss:.6f}")

                print(f"\n🎉 New best! Step {latest_event.step}: {current_loss:.4f} (saved to best_model.pth)\n")
            else:
                print(f"⚠️  New best found but checkpoint not yet saved: step {latest_event.step}")

        time.sleep(30)

    except KeyboardInterrupt:
        print("\nMonitor stopped")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(30)
EOF

# 백그라운드로 best checkpoint 모니터 실행
nohup python3 /tmp/monitor_best_checkpoint.py > /tmp/best_ckpt_monitor.log 2>&1 &
MONITOR_PID=$!
echo "Best checkpoint monitor started (PID: ${MONITOR_PID})"
echo "로그: /tmp/best_ckpt_monitor.log"
echo ""

echo "학습을 시작합니다..."
echo "TensorBoard: http://localhost:6006"
echo ""
echo "참고: 데이터 검증 스킵 (SKIP_DATA_CHECK=1)"
echo "      검증하려면: SKIP_DATA_CHECK=0 ./tools/restart_from_base_with_best.sh"
echo ""

# 재학습 시작
SKIP_DATA_CHECK=1 \
LR=5e-6 \
WARMUP_STEPS=10000 \
MAX_STEPS=0 \
GRAD_CLIP=0.5 \
BATCH_SIZE=8 \
LOG_INTERVAL=100 \
VAL_INTERVAL=1000 \
EPOCHS=2 \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
"${SCRIPT_DIR}/ko_step4_train_gpt.sh"

# 학습 완료 후 모니터 종료
kill ${MONITOR_PID} 2>/dev/null || true

echo ""
echo "================================================================"
echo "재학습 완료!"
echo "================================================================"
echo ""
echo "저장된 체크포인트:"
echo "  - 최고 성능: /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
echo "  - 최신: /mnt/sda1/models/index-tts-ko/checkpoints/latest.pth"
echo "  - 최근 5개: model_step*.pth"
echo ""
cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt 2>/dev/null && \
  echo "Best mel_loss: $(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt)"
