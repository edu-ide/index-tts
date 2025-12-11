# Stage 2 GRL Training Guide

## 개요

Stage 2는 Gradient Reversal Layer (GRL)를 사용하여 화자 독립적인 표현을 학습합니다.
- **목표**: 화자 정보를 제거하면서 음성 특성 유지
- **데이터**: sdb1의 전체 5.4M 샘플 (mel 있는 샘플 우선)

## 데이터 구조

```
/mnt/sdb1/emilia-yodas/KO_preprocessed/
├── gpt_pairs_train.jsonl      # 전체 5,483,856 샘플
├── gpt_pairs_train_mel.jsonl  # mel 정렬된 manifest (학습용)
├── mel/                        # mel-spectrogram (현재 226,890개)
├── codes/                      # semantic codes
├── condition/                  # condition vectors
├── emo_vec/                    # emotion vectors
├── text_ids/                   # text token ids
└── KO-B*/                      # 원본 오디오 파일
```

## 워크플로우

### 1. manifest 생성 (mel 우선 정렬)

```bash
bash tools/generate_mel_manifest_ssd.sh
```

**결과**: `gpt_pairs_train_mel.jsonl`
- mel 있는 샘플이 앞에 위치 (226,890개)
- mel 없는 샘플이 뒤에 위치 (~5.2M개)
- WSD 스케줄은 전체 5.4M 기준으로 계산

### 2. Stage 2 학습 시작

```bash
bash tools/run_stage2_training.sh
```

**기본 설정**:
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| BATCH_SIZE | 16 | 배치 크기 |
| GRAD_ACC | 4 | Gradient accumulation |
| LR | 2e-4 | Learning rate |
| WARMUP_STEPS | 1000 | Warmup 단계 |
| GRL_LAMBDA | 1.0 | GRL 강도 |
| SPEAKER_LOSS_WEIGHT | 0.1 | 화자 분류 손실 가중치 |

**커스텀 설정**:
```bash
BATCH_SIZE=24 GRAD_ACC=2 LR=1e-4 bash tools/run_stage2_training.sh
```

### 3. 추가 mel 전처리 (선택)

학습 중 226,890개 mel을 모두 사용했다면 추가 전처리:

```bash
# 전체 또는 일부 mel 추가 생성
bash tools/run_preprocess_target_mel.sh

# 옵션 지정
NUM_WORKERS=48 LIMIT=500000 bash tools/run_preprocess_target_mel.sh
```

### 4. manifest 재생성 및 학습 재개

```bash
# manifest 재생성 (새 mel 반영)
bash tools/generate_mel_manifest_ssd.sh

# 학습 재개 (자동으로 checkpoint에서 resume)
bash tools/run_stage2_training.sh
```

## 스크립트 목록

| 스크립트 | 설명 |
|----------|------|
| `tools/run_stage2_training.sh` | Stage 2 학습 실행 |
| `tools/generate_mel_manifest_ssd.sh` | mel 우선 정렬 manifest 생성 |
| `tools/run_preprocess_target_mel.sh` | 추가 mel 전처리 |
| `tools/preprocess_target_mel.py` | mel 전처리 Python 스크립트 |

## 체크포인트 구조

```
/mnt/sda1/models/index-tts-ko/stage2/
├── latest_full.pth    # 최신 전체 체크포인트 (resume용)
├── latest.pth         # 최신 모델 가중치만
├── best_model.pth     # 최고 성능 모델
├── config.yaml        # 학습 설정
└── step_XXXXX.pth     # 단계별 체크포인트
```

## WSD 스케줄러 동작

**Warmup-Stable-Decay (WSD)** 스케줄러:
- **Warmup**: 0 → max LR (warmup_steps 동안)
- **Stable**: max LR 유지
- **Decay**: max LR → 0 (마지막 10%)

**핵심**: manifest 크기가 동일해야 스케줄 일관성 유지
- 전체 5.4M manifest 사용
- mel 있는 샘플만 앞에 정렬
- 추가 전처리 후에도 manifest 크기 동일

## 문제 해결

### speaker_acc가 0인 경우
- `speaker_mapping_from_manifest.json` 확인
- manifest의 speaker 필드와 매핑 일치 여부 확인

### mel 로딩 실패
- `target_mel_path` 필드 존재 여부 확인
- mel 파일 경로 유효성 확인

### OOM (Out of Memory)
```bash
BATCH_SIZE=8 GRAD_ACC=8 bash tools/run_stage2_training.sh
```

## 모니터링

### TensorBoard
```bash
bash tools/run_tensorboard_stage2.sh
# http://localhost:6006
```

### 주요 메트릭
- `train/gpt_loss`: GPT 언어 모델 손실
- `train/speaker_loss`: 화자 분류 손실 (낮을수록 좋음)
- `train/speaker_acc`: 화자 분류 정확도 (GRL로 낮아져야 함)
- `val/loss`: 검증 손실

## 예상 소요 시간

| 데이터 크기 | 예상 시간 (1 epoch) |
|-------------|---------------------|
| 226,890 (현재 mel) | ~2-3시간 |
| 500,000 | ~5-6시간 |
| 1,000,000 | ~10-12시간 |
| 5,400,000 (전체) | ~50-60시간 |

*RTX 4090 기준, batch=16, grad_acc=4*
