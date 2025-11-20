#!/usr/bin/env bash
# Step 4: Fine-tune the GPT component on the Korean GPT pair manifests.

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경을 활성화하세요." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl::ko}"
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_val.jsonl::ko}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model}"
BASE_TOKENIZER_MODEL="${BASE_TOKENIZER_MODEL:-/mnt/sda1/models/IndexTTS-2/bpe.model}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/sda1/models/IndexTTS-2/config.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/mnt/sda1/models/IndexTTS-2/gpt.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-0.2}"
MEL_LOSS_WEIGHT="${MEL_LOSS_WEIGHT:-0.8}"
AMP="${AMP:-0}"
SEED="${SEED:-1234}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FLAG="${RESUME:-}"

IFS=',' read -ra TRAIN_ARRAY <<< "${TRAIN_MANIFEST}"
TRAIN_FLAGS=()
for entry in "${TRAIN_ARRAY[@]}"; do
  trimmed="$(echo "${entry}" | xargs)"
  [[ -z "${trimmed}" ]] && continue
  TRAIN_FLAGS+=(--train-manifest "${trimmed}")
done

IFS=',' read -ra VAL_ARRAY <<< "${VAL_MANIFEST}"
VAL_FLAGS=()
for entry in "${VAL_ARRAY[@]}"; do
  trimmed="$(echo "${entry}" | xargs)"
  [[ -z "${trimmed}" ]] && continue
  VAL_FLAGS+=(--val-manifest "${trimmed}")
done

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/../trainers/train_gpt_v2.py")
CMD+=("${TRAIN_FLAGS[@]}")
CMD+=("${VAL_FLAGS[@]}")
CMD+=(
  --tokenizer "${TOKENIZER_MODEL}"
  --base-tokenizer "${BASE_TOKENIZER_MODEL}"
  --config "${CONFIG_PATH}"
  --base-checkpoint "${BASE_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --grad-accumulation "${GRAD_ACC}"
  --epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-steps "${WARMUP_STEPS}"
  --max-steps "${MAX_STEPS}"
  --log-interval "${LOG_INTERVAL}"
  --val-interval "${VAL_INTERVAL}"
  --num-workers "${NUM_WORKERS}"
  --grad-clip "${GRAD_CLIP}"
  --text-loss-weight "${TEXT_LOSS_WEIGHT}"
  --mel-loss-weight "${MEL_LOSS_WEIGHT}"
  --seed "${SEED}"
)

if [[ "${AMP}" == "1" ]]; then
  CMD+=(--amp)
fi

if [[ -n "${RESUME_FLAG}" ]]; then
  CMD+=(--resume "${RESUME_FLAG}")
fi

echo "[KO-STEP4] train manifests=${TRAIN_FLAGS[*]}"
echo "[KO-STEP4] val manifests=${VAL_FLAGS[*]}"
echo "[KO-STEP4] output-dir=${OUTPUT_DIR}"

# Best checkpoint 모니터링 시작
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

nohup python3 /tmp/monitor_best_checkpoint.py > /tmp/best_ckpt_monitor.log 2>&1 &
MONITOR_PID=$!
echo "✅ Best checkpoint monitor started (PID: ${MONITOR_PID})"
echo "   Log: /tmp/best_ckpt_monitor.log"
echo ""

if [[ "${TIMEOUT_SECS}" -gt 0 ]]; then
  timeout "${TIMEOUT_SECS}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

# 학습 완료 후 모니터 종료
kill ${MONITOR_PID} 2>/dev/null || true
echo ""
echo "[KO-STEP4] GPT fine-tuning command finished."
echo ""
if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt" ]]; then
  echo "🏆 Best mel_loss: $(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt)"
  echo "📁 Best checkpoint: /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth"
fi
