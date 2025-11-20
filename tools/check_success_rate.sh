#!/usr/bin/env bash
# Phase 1 완료 후 성공 가능성 평가

set -euo pipefail

echo "================================================================"
echo "Phase 1 성공 가능성 평가"
echo "================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "${SCRIPT_DIR}/../.venv/bin/activate" 2>/dev/null || true
fi

cat > /tmp/evaluate_phase1.py << 'EOF'
#!/usr/bin/env python3
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

log_dir = Path("/mnt/sda1/models/index-tts-ko/checkpoints/logs")
latest_run = sorted(log_dir.glob("run_*"))[-1]

print(f"분석: {latest_run.name}\n")

ea = event_accumulator.EventAccumulator(str(latest_run))
ea.Reload()

text_events = ea.Scalars('train/text_loss')
mel_events = ea.Scalars('train/mel_loss')

text_values = [e.value for e in text_events]
text_steps = [e.step for e in text_events]

mel_values = [e.value for e in mel_events]

# Phase 1은 마지막 5000 step
if len(text_values) < 50:
    print("데이터가 부족합니다. 더 학습이 필요합니다.")
    exit(1)

# 최근 5000 step 분석
recent_text = text_values[-50:]
recent_steps = text_steps[-50:]

# 추세 계산 (linear regression)
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(recent_text)), recent_text)

print(f"=== Phase 1 결과 (최근 50개 step) ===")
print(f"시작 loss: {recent_text[0]:.4f}")
print(f"종료 loss: {recent_text[-1]:.4f}")
print(f"변화: {recent_text[-1] - recent_text[0]:+.4f}")
print(f"추세 기울기: {slope:.6f}")
print(f"R² (선형성): {r_value**2:.4f}")
print()

# Validation loss 확인
has_val = 'val/text_loss' in ea.Tags()['scalars']
if has_val:
    val_events = ea.Scalars('val/text_loss')
    val_values = [e.value for e in val_events]
    if val_values:
        print(f"Validation loss: {val_values[-1]:.4f}")
        print()

# 성공 가능성 평가
success_prob = 0

if slope < -0.001:  # 감소 추세
    success_prob += 40
    print("✅ Loss 감소 추세 (+40%)")
elif slope < 0:
    success_prob += 20
    print("⚠️  약한 감소 추세 (+20%)")
else:
    print("❌ Loss 증가 추세 (0%)")

if recent_text[-1] < recent_text[0]:
    success_prob += 30
    print("✅ 최종 loss < 시작 loss (+30%)")

if not has_val or (has_val and val_values[-1] < val_values[0] * 1.1):
    success_prob += 30
    print("✅ Validation loss 정상 (+30%)")

print()
print(f"=== 총 성공 가능성: {success_prob}% ===")
print()

if success_prob >= 70:
    print("🎉 Phase 2 진행을 강력히 추천합니다!")
    print("   ./tools/phase2_continue.sh")
elif success_prob >= 40:
    print("⚠️  Phase 2 진행 가능하나 주의가 필요합니다.")
    print("   LR을 더 낮춰보는 것도 고려하세요.")
elif success_prob >= 20:
    print("🔴 Phase 2 성공 가능성 낮음")
    print("   처음부터 재학습을 권장합니다.")
    print("   ./tools/restart_from_base.sh")
else:
    print("❌ 재학습 필수")
    print("   ./tools/restart_from_base.sh")
EOF

python3 /tmp/evaluate_phase1.py
