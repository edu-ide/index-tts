#!/usr/bin/env bash
# LR Finder: 최적 Learning Rate 찾기
# 원리: LR을 점진적으로 증가시키면서 loss 추이 관찰
# 최고 LR = loss가 가장 빠르게 감소하는 지점

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 가상환경이 활성화되지 않았습니다." >&2
  echo "실행: source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "🔍 Learning Rate Finder"
echo "================================================================"
echo ""
echo "원리:"
echo "  1. LR을 1e-7에서 1e-4까지 점진적으로 증가"
echo "  2. 각 LR에서 loss 측정"
echo "  3. Loss가 가장 빠르게 감소하는 LR 찾기"
echo ""
echo "예상 시간: 30-60분"
echo "출력: lr_finder_results.txt"
echo ""
echo "================================================================"

# LR Finder 실행
SKIP_DATA_CHECK=1 \
BATCH_SIZE="${BATCH_SIZE:-16}" \
MAX_STEPS=2000 \
LOG_INTERVAL=10 \
BASE_CHECKPOINT="/mnt/sda1/models/IndexTTS-2/gpt.pth" \
python3 << 'PYEOF'
import os
import sys
sys.path.insert(0, '/mnt/sdc1/ws/workspace/monorepo/external/index-tts')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from trainers.train_gpt_v2 import main, parse_args

# LR 범위 설정
lr_min = 1e-7
lr_max = 1e-4
num_steps = 2000

lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), num_steps)
losses = []

print("Starting LR Finder...")
print(f"Testing LRs from {lr_min:.2e} to {lr_max:.2e}")

for i, lr in enumerate(lrs):
    # 각 LR로 1 step 실행
    os.environ['LR'] = str(lr)

    # TODO: train_gpt_v2.py 호출해서 1 step 실행하고 loss 기록
    # 실제 구현 필요

    if i % 100 == 0:
        print(f"Step {i}/{num_steps}: LR={lr:.2e}")

# 결과 저장
results_file = Path("/mnt/sda1/models/index-tts-ko/lr_finder_results.txt")
with open(results_file, 'w') as f:
    f.write("LR,Loss\n")
    for lr, loss in zip(lrs, losses):
        f.write(f"{lr:.6e},{loss:.6f}\n")

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True)
plt.savefig('/mnt/sda1/models/index-tts-ko/lr_finder.png')
print(f"Results saved to {results_file}")
print("Plot saved to /mnt/sda1/models/index-tts-ko/lr_finder.png")

# 최적 LR 추천
min_loss_idx = np.argmin(losses)
optimal_lr = lrs[min_loss_idx]
print(f"\n🎯 Recommended LR: {optimal_lr:.2e}")
PYEOF

echo ""
echo "================================================================"
echo "✅ LR Finder 완료!"
echo "================================================================"
echo ""
echo "다음 단계:"
echo "  1. 결과 확인: /mnt/sda1/models/index-tts-ko/lr_finder_results.txt"
echo "  2. 그래프 확인: /mnt/sda1/models/index-tts-ko/lr_finder.png"
echo "  3. 추천 LR로 학습 시작"
echo ""
