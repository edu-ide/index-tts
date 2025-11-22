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

# Best checkpoint 모니터링 스크립트 생성 (체크포인트의 loss를 직접 비교)
cat > /tmp/monitor_best_checkpoint.py << 'EOF'
#!/usr/bin/env python3
"""
Best checkpoint 모니터링 및 자동 저장 (체크포인트에 저장된 loss 기준)
- latest.pth를 읽어 loss 비교 후 best_model_stepXXXX.pth 갱신
"""
import time
import shutil
from pathlib import Path

import torch

ckpt_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints")
latest_light = ckpt_dir / "latest.pth"
latest_full = ckpt_dir / "latest_full.pth"
best_loss_file = ckpt_dir / "best_loss.txt"
best_step_file = ckpt_dir / "best_step.txt"

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
    # validation 우선, 없으면 train loss로 폴백
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

        if not light_changed and not full_changed:
            time.sleep(30)
            continue

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

            if light_exists:
                t1 = threading.Thread(target=_copy, args=(latest_light, best_target_light), daemon=True)
                t1.start()
                t1.join()

            if full_exists:
                try:
                    ckpt_full = torch.load(latest_full, map_location="cpu")
                    full_step = ckpt_full.get("step") or ckpt_full.get("global_step") or ckpt_full.get("epoch") or 0
                    if full_step == step:
                        t2 = threading.Thread(target=_copy, args=(latest_full, best_target_full), daemon=True)
                        t2.start()
                        t2.join()
                    else:
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
