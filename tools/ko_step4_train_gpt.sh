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
VAL_MANIFEST="${VAL_MANIFEST:-/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_val_subset.jsonl::ko}"
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
NUM_WORKERS="${NUM_WORKERS:-$(nproc)}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TEXT_LOSS_WEIGHT="${TEXT_LOSS_WEIGHT:-0.2}"
MEL_LOSS_WEIGHT="${MEL_LOSS_WEIGHT:-0.8}"
AMP="${AMP:-0}"
SEED="${SEED:-1234}"
TIMEOUT_SECS="${TIMEOUT_SECS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_FLAG="${RESUME:-}"
OPTIMIZER_FLAG="${OPTIMIZER:-adamw}"
SCHEDULER_FLAG="${SCHEDULER:-cosine}"
WSD_STABLE_RATIO="${WSD_STABLE_RATIO:-0.9}"
WSD_MIN_LR_RATIO="${WSD_MIN_LR_RATIO:-0.0}"
BEST_STEP_ROUND="${BEST_STEP_ROUND:-0}"   # if >0 use model_stepXXXX.pth; if 0 use latest.pth directly

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
  --optimizer "${OPTIMIZER_FLAG}"
  --scheduler "${SCHEDULER_FLAG}"
  --wsd-stable-ratio "${WSD_STABLE_RATIO}"
  --wsd-min-lr-ratio "${WSD_MIN_LR_RATIO}"
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
# Kill existing monitors to avoid duplicates (Bash)
if command -v pkill >/dev/null 2>&1; then
  pkill -f monitor_best_checkpoint.py || true
fi

cat > /tmp/monitor_best_checkpoint.py << 'EOF'
#!/usr/bin/env python3
"""
Best checkpoint 모니터링 및 자동 저장 (체크포인트에 저장된 loss 기준)
- latest.pth를 주기적으로 읽어 loss를 비교 후 best_model_stepXXXX.pth를 갱신
- TensorBoard 로그 의존성을 제거해 누락/손상 시에도 동작
"""
import time
import shutil
from pathlib import Path

import torch

ckpt_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints")
latest_ckpt = ckpt_dir / "latest.pth"

# Best model (loss 기준): keep only one best_model_stepXXXX.pth
best_loss_file = ckpt_dir / "best_loss.txt"
best_step_file = ckpt_dir / "best_step.txt"

# 초기 best loss
if best_loss_file.exists():
    try:
        with open(best_loss_file, 'r') as f:
            best_loss = float(f.read().strip())
    except Exception:
        best_loss = float('inf')
else:
    best_loss = float('inf')

print("Best checkpoint monitor started (checkpoint loss criterion)")
print(f"  Current best loss: {best_loss if best_loss < float('inf') else 'inf'}")


def extract_loss(ckpt: dict):
    """Validation 우선: val_text_loss > val_mel_loss. 없으면 None 반환."""
    extra = ckpt.get("extra") or {}
    candidates = [
        ("val_text_loss", ckpt.get("val_text_loss")),
        ("val_mel_loss", ckpt.get("val_mel_loss")),
        ("val_text_loss", extra.get("val_text_loss")),
        ("val_mel_loss", extra.get("val_mel_loss")),
    ]
    for name, value in candidates:
        if value is not None:
            try:
                return name, float(value)
            except Exception:
                continue
    return None, None


last_mtime = 0.0

while True:
    try:
        if not latest_ckpt.exists():
            time.sleep(30)
            continue

        mtime = latest_ckpt.stat().st_mtime
        if mtime == last_mtime:
            time.sleep(30)
            continue

        last_mtime = mtime

        try:
            ckpt = torch.load(latest_ckpt, map_location="cpu")
        except Exception as load_err:
            print(f"Error loading {latest_ckpt.name}: {load_err}")
            time.sleep(30)
            continue

        metric_name, current_loss = extract_loss(ckpt)
        if metric_name is None:
            print("Warning: no validation loss found in checkpoint; waiting for next validation")
            time.sleep(30)
            continue

        step = ckpt.get("step") or ckpt.get("global_step") or ckpt.get("epoch")
        if step is None:
            step = 0

        if current_loss < best_loss:
            best_loss = current_loss

            # 기본 복사 대상은 latest.pth
            target_ckpt = latest_ckpt
            target_step = step

            # BEST_STEP_ROUND > 0이면 rounding된 model_stepXXXX.pth가 있으면 우선 사용
            # latest.pth만 유지하므로 rounding은 무시하고 latest 사용

            # 이전 best 삭제 (legacy 이름도 제거)
            for old_best in ckpt_dir.glob("best_model_step*.pth"):
                try:
                    old_best.unlink()
                except Exception:
                    pass
            legacy = ckpt_dir / "best_model.pth"
            if legacy.exists():
                try:
                    legacy.unlink()
                except Exception:
                    pass

            best_target = ckpt_dir / f"best_model_step{target_step}.pth"

            # 비동기 + 원자적 교체: tmp에 저장 후 rename
            def _copy_best():
                tmp = best_target.with_suffix(best_target.suffix + ".tmp")
                shutil.copy2(target_ckpt, tmp)
                tmp.replace(best_target)

            import threading
            t = threading.Thread(target=_copy_best, daemon=True)
            t.start()
            t.join()
            with open(best_loss_file, 'w') as f:
                f.write(f"{best_loss:.6f}")
            with open(best_step_file, 'w') as f:
                f.write(str(step))

            print(
                f"\n🎯 New best! step={step} metric={metric_name} loss={current_loss:.4f} "
                f"(copied from {target_ckpt.name} -> {best_target.name})\n"
            )

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
  echo "🎯 Best text_loss: $(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_loss.txt)"
  if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_step.txt" ]]; then
    best_step=$(cat /mnt/sda1/models/index-tts-ko/checkpoints/best_step.txt)
    echo "🪜 Best step: ${best_step}"
    if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_model_step${best_step}.pth" ]]; then
      echo "📁 Best checkpoint: /mnt/sda1/models/index-tts-ko/checkpoints/best_model_step${best_step}.pth"
    fi
  fi
fi
