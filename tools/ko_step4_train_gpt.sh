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
latest_light = ckpt_dir / "latest.pth"
latest_full = ckpt_dir / "latest_full.pth"

# Best model (loss 기준): keep only one best_model_stepXXXX.pth and best_model_stepXXXX_full.pth
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
    """Validation 우선, 없으면 train loss로 폴백."""
    extra = ckpt.get("extra") or {}
    candidates = [
        ("val_text_loss", ckpt.get("val_text_loss")),
        ("val_mel_loss", ckpt.get("val_mel_loss")),
        ("val_text_loss", extra.get("val_text_loss")),
        ("val_mel_loss", extra.get("val_mel_loss")),
        ("train_text_loss", ckpt.get("train_text_loss")),
        ("train_mel_loss", ckpt.get("train_mel_loss")),
    ]
    for name, value in candidates:
        if value is not None:
            try:
                return name, float(value)
            except Exception:
                continue
    return None, None


last_mtime_light = 0.0
last_mtime_full = 0.0

while True:
    try:
        # 최신 스냅샷 여부 확인
        light_exists = latest_light.exists()
        full_exists = latest_full.exists()

        if not light_exists and not full_exists:
            time.sleep(30)
            continue

        light_changed = False
        full_changed = False
        if light_exists:
            mtime_l = latest_light.stat().st_mtime
            if mtime_l != last_mtime_light:
                last_mtime_light = mtime_l
                light_changed = True
        if full_exists:
            mtime_f = latest_full.stat().st_mtime
            if mtime_f != last_mtime_full:
                last_mtime_full = mtime_f
                full_changed = True

        # 둘 다 안 바뀌었으면 skip
        if not light_changed and not full_changed:
            time.sleep(30)
            continue

        # 우선 loss 평가에는 light(모델만) 사용
        try:
            ckpt_light = torch.load(latest_light, map_location="cpu") if light_exists else None
        except Exception as load_err:
            print(f"Error loading {latest_light.name}: {load_err}")
            ckpt_light = None

        if ckpt_light is None:
            time.sleep(30)
            continue

        metric_name, current_loss = extract_loss(ckpt_light)
        if metric_name is None:
            print("Warning: no validation loss found in checkpoint; waiting for next validation")
            time.sleep(30)
            continue

        step = ckpt_light.get("step") or ckpt_light.get("global_step") or ckpt_light.get("epoch")
        if step is None:
            step = 0

        if current_loss < best_loss:
            best_loss = current_loss

            # 이전 best 삭제
            for old_best in ckpt_dir.glob("best_model_step*.pth"):
                try:
                    old_best.unlink()
                except Exception:
                    pass
            for old_best_full in ckpt_dir.glob("best_model_step*_full.pth"):
                try:
                    old_best_full.unlink()
                except Exception:
                    pass
            legacy = ckpt_dir / "best_model.pth"
            if legacy.exists():
                try:
                    legacy.unlink()
                except Exception:
                    pass

            best_target_light = ckpt_dir / f"best_model_step{step}.pth"
            best_target_full = ckpt_dir / f"best_model_step{step}_full.pth"

            def _copy(src: Path, dst: Path):
                tmp = dst.with_suffix(dst.suffix + ".tmp")
                shutil.copy2(src, tmp)
                tmp.replace(dst)

            import threading

            # light 저장
            if light_exists:
                t1 = threading.Thread(target=_copy, args=(latest_light, best_target_light), daemon=True)
                t1.start()
                t1.join()

            # full 저장 (full이 존재하고 step이 일치할 때만)
            if full_exists:
                try:
                    ckpt_full = torch.load(latest_full, map_location="cpu")
                    full_step = ckpt_full.get("step") or ckpt_full.get("global_step") or ckpt_full.get("epoch") or 0
                    if full_step == step:
                        t2 = threading.Thread(target=_copy, args=(latest_full, best_target_full), daemon=True)
                        t2.start()
                        t2.join()
                    else:
                        # 스텝이 안 맞아도 최소한 최신 full 스냅샷을 best_model_full.pth 로 보관
                        fallback_full = ckpt_dir / "best_model_full.pth"
                        t2 = threading.Thread(target=_copy, args=(latest_full, fallback_full), daemon=True)
                        t2.start()
                        t2.join()
                        print(f"[Best Monitor] latest_full step={full_step} != best step={step}; saved fallback {fallback_full.name}.")
                except Exception as e:
                    print(f"[Best Monitor] Failed to copy full checkpoint: {e}")

            with open(best_loss_file, 'w') as f:
                f.write(f"{best_loss:.6f}")
            with open(best_step_file, 'w') as f:
                f.write(str(step))

            copied = f" -> {best_target_light.name}"
            if best_target_full.exists():
                copied += f", {best_target_full.name}"

            print(
                f"\n🎯 New best! step={step} metric={metric_name} loss={current_loss:.4f} "
                f"(copied from latest.pth{copied})\n"
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
    if [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_model_step${best_step}_full.pth" ]]; then
      echo "📁 Best checkpoint (full): /mnt/sda1/models/index-tts-ko/checkpoints/best_model_step${best_step}_full.pth"
    elif [[ -f "/mnt/sda1/models/index-tts-ko/checkpoints/best_model_full.pth" ]]; then
      echo "📁 Best checkpoint (full fallback): /mnt/sda1/models/index-tts-ko/checkpoints/best_model_full.pth"
    fi
  fi
fi
