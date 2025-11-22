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
latest_ckpt = ckpt_dir / "latest.pth"
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
    # validation 우선, 없으면 None 반환
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

            # 복사 대상: latest.pth (rounding 필요 없으므로 그대로 사용)
            target_ckpt = latest_ckpt
            target_step = step

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
