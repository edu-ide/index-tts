#!/usr/bin/env bash
# 학습 모니터링 스크립트
# 실시간으로 loss와 learning rate 추이를 확인

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/mnt/sda1/models/index-tts-ko/checkpoints/logs"

echo "================================================================"
echo "학습 모니터링"
echo "================================================================"
echo ""
echo "TensorBoard: http://localhost:6006"
echo "로그 디렉토리: ${LOG_DIR}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Python 모니터링 스크립트
cat > /tmp/monitor_training.py << 'EOF'
#!/usr/bin/env python3
import time
import sys
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

log_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints/logs")

def get_latest_run():
    runs = sorted(log_dir.glob("run_*"))
    return runs[-1] if runs else None

def monitor():
    latest_run = get_latest_run()
    if not latest_run:
        print("로그를 찾을 수 없습니다.")
        return

    print(f"모니터링: {latest_run.name}\n")

    ea = event_accumulator.EventAccumulator(str(latest_run))
    prev_step = -1

    while True:
        ea.Reload()

        if 'train/text_loss' not in ea.Tags()['scalars']:
            time.sleep(30)
            continue

        events = ea.Scalars('train/text_loss')
        if not events:
            time.sleep(30)
            continue

        latest = events[-1]
        if latest.step == prev_step:
            time.sleep(30)
            continue

        prev_step = latest.step

        # 모든 메트릭 가져오기
        text_loss = latest.value
        mel_loss = ea.Scalars('train/mel_loss')[-1].value if 'train/mel_loss' in ea.Tags()['scalars'] else -1
        lr = ea.Scalars('train/lr')[-1].value if 'train/lr' in ea.Tags()['scalars'] else -1

        # 출력
        print(f"\r[Step {latest.step:6d}] text_loss: {text_loss:.4f} | mel_loss: {mel_loss:.4f} | lr: {lr:.8f}", end='', flush=True)

        # 30초마다 새 줄
        if latest.step % 10 == 0:
            print()

        time.sleep(30)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n모니터링 종료")
        sys.exit(0)
EOF

# 가상환경에서 실행
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${SCRIPT_DIR}/../.venv/bin/activate" 2>/dev/null || true
fi

python3 /tmp/monitor_training.py
