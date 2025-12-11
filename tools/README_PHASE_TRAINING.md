# 📚 Phase 별 학습 가이드

## 🎯 현재 상황

- **데이터**: 548만개 샘플 (충분!)
- **진행**: step 438,000 (0.32 epoch)
- **문제**: step 298,800 이후 loss 폭발
- **원인**: Learning rate가 너무 높음 (2e-5)
- **백업**: step 351,000 체크포인트만 있음

## 📋 학습 전략

### Phase 1: 빠른 검증 (1-2시간)
초저 LR로 기존 체크포인트가 회복 가능한지 확인

```bash
cd /mnt/sdc1/ws/workspace/monorepo/external/index-tts
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate
./tools/phase1_validate_lr.sh
```

**설정:**
- LR: 1e-6 (기존의 1/20)
- Steps: 5000
- 시간: 1-2시간

**판단 기준:**
- ✅ loss 감소 → Phase 2로 진행
- ❌ loss 증가 → Base 모델부터 재학습

### Phase 2: 계속 학습 (Phase 1 성공 시)
LR을 점진적으로 증가시키며 1 epoch 완료

```bash
./tools/phase2_continue.sh
```

**설정:**
- LR: 2e-6 (Phase 1의 2배)
- Steps: 제한 없음
- 목표: 1 epoch 완료

### 대안: Base 모델부터 재학습 (Phase 1 실패 시)
처음부터 보수적인 LR로 학습

```bash
./tools/restart_from_base.sh
```

**설정:**
- LR: 5e-6
- Batch size: 8
- 예상 시간: 50-70시간

## 📊 모니터링

### 실시간 모니터링
```bash
# 터미널 1: 학습 실행
./tools/phase1_validate_lr.sh

# 터미널 2: 모니터링
./tools/monitor_training.sh
```

### TensorBoard
```bash
# 브라우저에서 열기
http://localhost:6006
```

## 🔍 체크포인트 위치

- **현재 체크포인트**: `/mnt/sda1/models/index-tts-ko/checkpoints/`
- **로그**: `/mnt/sda1/models/index-tts-ko/checkpoints/logs/`
- **백업** (재학습 시): `/mnt/sda1/models/index-tts-ko/checkpoints_backup_YYYYMMDD_HHMMSS/`

## 📈 성공 지표

### Phase 1 성공:
- text_loss: 감소 추세
- mel_loss: 유지 또는 감소
- 5000 step 후 text_loss < 2.0

### 최종 목표:
- text_loss < 0.9
- mel_loss < 3.5
- mel_top1 accuracy > 15%
- 최소 1 epoch 완료

## ⚠️ 주의사항

1. **학습 중단 시**: Ctrl+C로 안전하게 종료 (자동 저장됨)
2. **디스크 공간**: 체크포인트는 7.3GB씩 차지 (여유 공간 확인)
3. **GPU 모니터링**: `nvidia-smi` 또는 `watch -n 1 nvidia-smi`로 확인
4. **메모리**: OOM 발생 시 BATCH_SIZE 줄이기

## 🚀 빠른 시작

```bash
# 1. 가상환경 활성화
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

# 2. Phase 1 시작
cd /mnt/sdc1/ws/workspace/monorepo/external/index-tts
./tools/phase1_validate_lr.sh

# 3. TensorBoard로 확인
# 브라우저: http://localhost:6006
```

## 💡 문제 해결

### Loss가 계속 증가:
```bash
# Learning rate를 더 낮춰서 재시도
LR=5e-7 ./tools/phase1_validate_lr.sh
```

### OOM 에러:
```bash
# Batch size 줄이기
BATCH_SIZE=2 ./tools/phase1_validate_lr.sh
```

### 학습 속도가 너무 느림:
```bash
# Worker 수 늘리기 (주의: 메모리 사용량 증가)
NUM_WORKERS=2 ./tools/phase1_validate_lr.sh
```

## 📞 도움말

문제가 발생하면:
1. TensorBoard에서 loss 확인
2. 로그 확인: `tail -f /mnt/sda1/models/index-tts-ko/checkpoints/logs/run_*/events.out.tfevents.*`
3. GPU 상태: `nvidia-smi`
4. 디스크 공간: `df -h /mnt/sda1`
