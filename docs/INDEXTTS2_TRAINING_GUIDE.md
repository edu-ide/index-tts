# IndexTTS2 3-Stage Training 완벽 가이드

> **IndexTTS2**: Zero-shot Voice Cloning with Emotion Disentanglement
>
> arXiv:2506.21619v2 (2025)

## 목차
1. [IndexTTS2란?](#indexts2란)
2. [왜 3-Stage Training인가?](#왜-3-stage-training인가)
3. [Stage 1: Basic TTS Training](#stage-1-basic-tts-training)
4. [Stage 2: Emotion Disentanglement with GRL](#stage-2-emotion-disentanglement-with-grl)
5. [Stage 3: Fine-tuning](#stage-3-fine-tuning)
6. [실습 가이드](#실습-가이드)
7. [이론 심화](#이론-심화)
8. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## IndexTTS2란?

### 개요
IndexTTS2는 **zero-shot voice cloning** 모델로, 단 몇 초의 음성 샘플만으로 새로운 화자의 목소리를 복제할 수 있습니다.

### 핵심 특징
1. **Zero-shot**: 학습 때 본 적 없는 화자의 목소리도 복제 가능
2. **Emotion Control**: 화자와 무관하게 감정 표현 제어
3. **3-Stage Training**: 체계적인 단계별 학습 전략

### 아키텍처
```
Input Text
    ↓
[Text Encoder]
    ↓
[GPT Backbone] ← [Speaker Condition] + [Emotion Condition]
    ↓
Semantic Codes
    ↓
[Vocoder]
    ↓
Audio
```

**Key Components**:
- **Speaker Perceiver**: 화자의 음색/스타일 추출
- **Emotion Perceiver**: 감정 표현 추출
- **GPT Backbone**: Text → Semantic codes 생성
- **Vocoder**: Semantic codes → Audio

---

## 왜 3-Stage Training인가?

### 문제 상황
**초기 학습 시 문제점**:
```python
# ❌ 한 번에 모든 것을 학습하면?
speaker_feature + emotion_feature → GPT → output

문제 1: Speaker와 Emotion이 엉킴 (entanglement)
  - 감정 A를 화자 X에게서만 학습
  - 화자 Y에게 감정 A를 적용하면 화자 X처럼 들림

문제 2: Feature extraction과 generation이 동시 학습
  - 두 과제가 서로 방해
  - 최적화 어려움
```

### 해결책: 3-Stage Training
```
Stage 1: 기본 TTS 능력 학습
  → 모든 컴포넌트가 협력하여 음성 생성 학습

Stage 2: Speaker-Emotion Disentanglement
  → Emotion에서 Speaker 정보 제거
  → 어떤 화자에게든 감정 적용 가능

Stage 3: Fine-tuning
  → Feature는 고정, 생성 품질만 개선
  → Overfitting 방지
```

---

## Stage 1: Basic TTS Training

### 목적
**"기본적인 Text-to-Speech 능력 학습"**

### 학습 설정
```yaml
Dataset: 전체 데이터 (2.7M samples)
Trainable: 모든 컴포넌트
  - Speaker Perceiver ✅
  - Emotion Perceiver ✅
  - GPT Backbone ✅
Learning Rate: 2e-4
Epochs: 여러 epoch (수렴할 때까지)
```

### 학습 과정
```python
# Pseudo-code
for batch in dataloader:
    # 1. Feature extraction
    speaker_feat = speaker_perceiver(speaker_mel)
    emotion_feat = emotion_perceiver(emotion_mel)

    # 2. Combine features
    condition = speaker_feat + emotion_feat

    # 3. Generate semantic codes
    codes = gpt(text, condition)

    # 4. Compute loss
    loss = cross_entropy(codes, target_codes)
    loss.backward()
    optimizer.step()
```

### 학습 결과
✅ Text → Semantic codes 매핑 학습
✅ Speaker feature extraction 학습
✅ Emotion feature extraction 학습
❌ **하지만**: Speaker와 Emotion이 entangled (섞여있음)

### 실행 방법
```bash
cd /mnt/sdc1/ws/workspace/monorepo/external/index-tts
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

./tools/train_ko_optimized_4090.sh
```

### 모니터링
```bash
tensorboard --logdir=/mnt/sda1/models/index-tts-ko/logs

# 확인할 지표:
# - train/mel_loss: 감소해야 함
# - train/mel_top1: 증가해야 함 (accuracy)
# - val/mel_loss: train과 비슷하게 감소 (overfitting 체크)
```

---

## Stage 2: Emotion Disentanglement with GRL

### 목적
**"Emotion feature에서 Speaker 정보 제거"**

### 문제 상황
Stage 1 후 상태:
```python
emotion_vec = emotion_perceiver(audio)
# ❌ 문제: emotion_vec에 speaker 정보가 섞여있음

# 예시:
speaker_A + emotion_happy → "행복한 A의 목소리"
speaker_B + emotion_happy → "여전히 A처럼 들림" (❌)
```

### 해결책: GRL (Gradient Reversal Layer)

#### GRL 원리
```python
class GradientReversalLayer:
    def forward(self, x):
        return x  # Forward: 그대로 통과

    def backward(self, grad):
        return -lambda * grad  # Backward: gradient 반전!
```

#### Adversarial Training
```
Emotion Encoder의 목표:
  - 감정 정보는 잘 추출하고
  - Speaker 정보는 제거하고 싶음

Speaker Classifier의 목표:
  - Emotion vector로부터 speaker 분류

GRL의 역할:
  - Speaker Classifier는 정상적으로 학습 (speaker 분류 잘하려고 함)
  - Emotion Encoder는 reversed gradient를 받음
    → Speaker Classifier를 "속이려고" 학습
    → 결과적으로 speaker 정보 제거!
```

### 상세 구현

#### 1. Forward Pass (Real-time Emo Vec Computation)
```python
# ✅ 이상적인 방식 (논문과 동일)
condition = load_mel_spectrogram()  # [batch, cond_len, 1024]

# Real-time으로 emo_vec 계산
emo_features = emo_conditioning_encoder(condition)
emo_vec_raw = emo_perceiver_encoder(emo_features)
emo_vec = emo_layer(emovec_layer(emo_vec_raw))

# Gradient가 여기로 흐름! ↓
```

#### 2. GRL + Speaker Classification
```python
# Apply GRL
emo_vec_reversed = GRL(emo_vec)

# Speaker classification
speaker_logits = speaker_classifier(emo_vec_reversed)
speaker_loss = cross_entropy(speaker_logits, speaker_labels)
```

#### 3. Backward Pass
```python
# Total loss
total_loss = tts_loss + alpha * speaker_loss

# Backward
total_loss.backward()

# Gradient flow:
# tts_loss → GPT, emotion encoder (정상)
# speaker_loss → speaker_classifier (정상)
#             → GRL (reversed!) → emotion encoder
#
# Emotion encoder는:
# - TTS loss로부터: 감정 정보 추출하라
# - Speaker loss로부터: Speaker 정보 제거하라 (reversed gradient)
```

### 학습 설정
```yaml
Dataset: 전체 데이터 (또는 감정 데이터 135시간)
Trainable:
  - Speaker Perceiver: ❌ FROZEN
  - Emotion Perceiver: ✅ TRAINABLE
  - GPT Backbone: ✅ TRAINABLE
  - GRL + Speaker Classifier: ✅ TRAINABLE
Learning Rate: 2e-4
GRL Lambda: 1.0
Speaker Loss Weight: 0.1
Epochs: 2 (논문 권장)
```

### 핵심 코드 분석
```python
# trainers/train_gpt_v2.py의 compute_losses()

# Real-time emo_vec computation
if enable_stage2_realtime_emo and model.enable_grl:
    # condition → emo encoder → emo_vec (실시간 계산)
    condition_transposed = condition.transpose(1, 2)
    emo_features = model.emo_conditioning_encoder(
        condition_transposed, condition_lengths
    )
    emo_vec_raw = model.emo_perceiver_encoder(emo_features)
    emo_vec = model.emo_layer(model.emovec_layer(emo_vec_raw.squeeze(1)))

    # GRL 적용
    emo_vec_reversed = model.grl(emo_vec)
    speaker_logits = model.speaker_classifier(emo_vec_reversed)

    # Speaker loss 계산
    speaker_loss = F.cross_entropy(speaker_logits, speaker_labels)
```

### 실행 방법

#### Step 1: Speaker Mapping 생성
```bash
python tools/build_speaker_mapping.py \
    --manifest /mnt/sda1/emilia-yodas/KO_preprocessed/train_manifest.jsonl \
    --output /mnt/sda1/models/index-tts-ko/speaker_mapping.json \
    --top-k 500 \
    --min-samples 50

# 출력:
# Total speakers: 132,389
# Selected top 500 speakers: 167,734 samples (6.03%)
```

#### Step 2: Stage 2 학습
```bash
# Stage 1 checkpoint 확인
export STAGE1_CHECKPOINT=/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth

# Stage 2 실행
./tools/train_ko_stage2.sh
```

### 모니터링
```bash
tensorboard --logdir=/mnt/sda1/models/index-tts-ko/stage2/logs

# 중요 지표:
# - train/speaker_loss: 감소해야 함
# - train/speaker_acc: 30-60% 유지 (중요!)
#   → 너무 높으면 (>80%): GRL이 효과 없음, emotion에 여전히 speaker 정보 많음
#   → 너무 낮으면 (<20%): speaker classifier가 학습 안됨
# - train/mel_loss: Stage 1과 비슷하게 유지
```

### 성공 기준
✅ Speaker accuracy: 30-60% (random보다는 높지만 너무 높지 않음)
✅ Mel loss: Stage 1과 비슷 (TTS 품질 유지)
✅ Emotion transfer: 다른 화자에게 감정 적용 시 원래 감정 유지

---

## Stage 3: Fine-tuning

### 목적
**"Feature는 보존하면서 생성 품질만 개선"**

### 왜 필요한가?

#### 문제 상황
Stage 2 후:
```python
# ✅ 좋은 점:
# - Speaker feature 잘 추출됨 (Stage 1)
# - Emotion feature 잘 분리됨 (Stage 2)

# ❌ 문제:
# - 계속 학습하면 feature drift 발생 가능
# - Speaker/Emotion perceiver가 변하면 Stage 2의 disentanglement 망가짐
```

#### 해결책: Freeze Conditioners
```python
# Feature extractors 고정
speaker_perceiver.requires_grad = False  # 🔒
emotion_perceiver.requires_grad = False  # 🔒

# GPT만 학습
gpt.requires_grad = True  # ✅

# 결과:
# - Stage 1, 2의 feature 보존
# - GPT의 생성 품질만 개선
# - Overfitting 방지
```

### 학습 설정
```yaml
Dataset: 전체 데이터
Frozen (🔒):
  - Speaker conditioning encoder
  - Speaker perceiver encoder
  - Emotion conditioning encoder
  - Emotion perceiver encoder
  - Emovec layer
  - Emo layer
Trainable (✅):
  - GPT Backbone
  - Text/Mel embeddings
  - Text/Mel heads
Learning Rate: 1e-4 (Stage 1/2의 절반!)
Epochs: 1
GRL: Disabled
```

### 핵심 코드
```python
# trainers/train_gpt_v2.py의 main()

if args.freeze_conditioners:
    # Freeze all feature extractors
    for module in [
        model.conditioning_encoder,
        model.perceiver_encoder,
        model.emo_conditioning_encoder,
        model.emo_perceiver_encoder,
        model.emovec_layer,
        model.emo_layer
    ]:
        for param in module.parameters():
            param.requires_grad = False

    # 결과 확인
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    # 출력 예시: Trainable: 45,234,567 / 98,765,432 (45.8%)
```

### 실행 방법
```bash
# Stage 2 checkpoint 확인
export STAGE2_CHECKPOINT=/mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth

# Stage 3 실행
./tools/train_ko_stage3.sh
```

### 모니터링
```bash
tensorboard --logdir=/mnt/sda1/models/index-tts-ko/stage3/logs

# 확인 사항:
# - train/mel_loss: 미세하게 감소 (큰 개선 기대하지 말것)
# - Trainable params: ~40-50%로 감소 확인
# - 학습 속도: Stage 1/2보다 ~2배 빠름
```

### 성공 기준
✅ Mel loss: Stage 2와 비슷하거나 약간 개선
✅ Speaker similarity: 유지
✅ Emotion transfer: 유지 (Stage 2의 결과 보존)
✅ 학습 안정성: Loss가 튀지 않음 (frozen features 덕분)

---

## 실습 가이드

### 전체 워크플로우

#### 0. 환경 준비
```bash
# 1. 가상환경 활성화
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate

# 2. 프로젝트 디렉토리 이동
cd /mnt/sdc1/ws/workspace/monorepo/external/index-tts

# 3. GPU 확인
nvidia-smi

# 4. TensorBoard 실행 (별도 터미널)
tensorboard --logdir=/mnt/sda1/models/index-tts-ko
# 브라우저: http://localhost:6006
```

#### 1. Stage 1 학습
```bash
# 실행
./tools/train_ko_optimized_4090.sh

# 소요 시간: 수일 (데이터셋 크기에 따라)

# 모니터링 (TensorBoard)
# - train/mel_loss: 지속적 감소
# - train/mel_top1: 0.6-0.8 정도 도달
# - val/mel_loss: train과 비슷하게 감소

# 완료 조건:
# - mel_loss가 수렴 (더 이상 감소 안함)
# - 생성된 오디오 품질 acceptable

# Checkpoint 위치:
# /mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth
```

#### 2. Speaker Mapping 생성 (한 번만)
```bash
python tools/build_speaker_mapping.py \
    --manifest /mnt/sda1/emilia-yodas/KO_preprocessed/train_manifest.jsonl \
    --output /mnt/sda1/models/index-tts-ko/speaker_mapping.json \
    --top-k 500 \
    --min-samples 50

# 출력 예시:
Building speaker mapping from /mnt/sda1/emilia-yodas/KO_preprocessed/train_manifest.jsonl
Total samples: 2,783,826
Total unique speakers: 132,389
Speakers with >= 50 samples: 14,244
Selected top 500 speakers: 167,734 samples (6.03% of total)

Speaker mapping saved to /mnt/sda1/models/index-tts-ko/speaker_mapping.json

# 검증:
cat /mnt/sda1/models/index-tts-ko/speaker_mapping.json | jq 'length'
# 출력: 500
```

#### 3. Stage 2 학습
```bash
# Stage 1 checkpoint 경로 설정
export STAGE1_CHECKPOINT=/mnt/sda1/models/index-tts-ko/checkpoints/best_model.pth

# 실행
./tools/train_ko_stage2.sh

# 소요 시간: 1-2 epochs

# 모니터링 (TensorBoard)
# 🎯 핵심 지표: train/speaker_acc
# - 이상적: 30-60%
# - 너무 높음 (>80%): GRL lambda 증가 (1.0 → 2.0)
# - 너무 낮음 (<20%): Speaker loss weight 증가 (0.1 → 0.2)

# 완료 조건:
# - Speaker accuracy 30-60% 안정화
# - Mel loss Stage 1과 비슷
# - 2 epochs 완료

# Checkpoint 위치:
# /mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth
```

#### 4. Stage 3 학습
```bash
# Stage 2 checkpoint 경로 설정
export STAGE2_CHECKPOINT=/mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth

# 실행
./tools/train_ko_stage3.sh

# 소요 시간: 1 epoch

# 시작 시 출력 확인:
[Stage 3] Freezing feature conditioners...
  ✅ Speaker conditioning encoder frozen
  ✅ Speaker perceiver encoder frozen
  ✅ Emotion conditioning encoder frozen
  ✅ Emotion perceiver encoder frozen
  ✅ Emovec layer frozen
  ✅ Emo layer frozen
[Stage 3] Trainable parameters: 45,234,567 / 98,765,432 (45.8%)

# 모니터링:
# - 학습 속도: Stage 1/2보다 빠름 (50% params)
# - Mel loss: 미세 개선 또는 유지
# - 안정성: Loss가 튀지 않음

# 최종 Checkpoint:
# /mnt/sda1/models/index-tts-ko/stage3/checkpoints/best_model.pth
```

### 학습 중단 및 재개
```bash
# 각 stage는 자동으로 checkpoint 저장
# 재개 시 --base-checkpoint에 마지막 checkpoint 지정

# 예: Stage 2 재개
python trainers/train_gpt_v2.py \
    ... (다른 arguments) ... \
    --base-checkpoint /mnt/sda1/models/index-tts-ko/stage2/checkpoints/model_step_5000.pth
```

---

## 이론 심화

### 1. GRL (Gradient Reversal Layer) 수학적 이해

#### Forward Pass
```python
y = GRL(x) = x  # Identity function
```

#### Backward Pass
```python
∂L/∂x = -λ * ∂L/∂y

여기서:
- L: Total loss
- λ: Reversal strength (typically 1.0)
- ∂L/∂y: Gradient from next layer
```

#### 왜 작동하는가?

**일반적인 학습**:
```python
# Encoder의 목표: Loss 최소화
loss = f(encoder(x))
∂loss/∂encoder_params = ∂loss/∂f * ∂f/∂encoder  # 정상 gradient
encoder_params -= lr * ∂loss/∂encoder_params  # Loss 감소 방향으로 업데이트
```

**GRL 적용**:
```python
# Emotion Encoder의 목표: Speaker classifier를 속이기
speaker_pred = classifier(GRL(emotion_encoder(x)))
loss = cross_entropy(speaker_pred, speaker_label)

# Backward:
∂loss/∂encoder_params = ∂loss/∂classifier * ∂classifier/∂GRL * (-λ) * ∂GRL/∂encoder
                      = ∂loss/∂classifier * ∂classifier/∂GRL * (-λ) * ∂encoder/∂encoder
                                                                   ↑
                                                            Gradient reversed!

# 결과:
# - Classifier: speaker 잘 분류하려고 학습 (정상 gradient)
# - Encoder: speaker 못 분류하게 만들려고 학습 (reversed gradient)
```

### 2. Adversarial Training의 균형

#### Min-Max Game
```
min_θ_emo max_θ_classifier L_speaker

여기서:
- θ_emo: Emotion encoder parameters
- θ_classifier: Speaker classifier parameters
- L_speaker: Speaker classification loss

Emotion encoder는 L을 최소화하려 하고 (speaker 제거)
Classifier는 L을 최대화하려 함 (speaker 분류)
```

#### Nash Equilibrium
이상적인 균형점:
```python
Speaker Accuracy = 1/N  # Random guess level

여기서 N = number of speakers

예: 500 speakers → ideal accuracy = 0.2% (random)
실제: 30-60% 정도면 충분히 좋음 (perfect random은 어려움)
```

### 3. 왜 Real-time Emo Vec이 중요한가?

#### Pre-computed 방식
```python
# Preprocessing
emo_vec = emotion_perceiver(mel).detach()  # Gradient 끊김!
save(emo_vec, "emo_vec.npy")

# Training (Stage 2)
emo_vec = load("emo_vec.npy")  # No gradient!
emo_vec_reversed = GRL(emo_vec)
loss.backward()
# ❌ Gradient가 emotion_perceiver로 못 흐름!
```

#### Real-time 방식
```python
# Training (Stage 2)
mel = load_mel_spectrogram()
emo_vec = emotion_perceiver(mel)  # ✅ Gradient 살아있음
emo_vec_reversed = GRL(emo_vec)
loss.backward()
# ✅ Gradient가 emotion_perceiver로 흐름!
```

**결과 비교**:
| 방식 | Gradient Flow | 효과 | 속도 |
|------|--------------|------|------|
| Pre-computed | ❌ | GRL이 제대로 작동 안함 | 빠름 |
| Real-time | ✅ | GRL이 제대로 작동 | ~10-15% 느림 |

### 4. Feature Freezing의 수학

#### Without Freezing
```python
# All parameters trainable
θ = [θ_speaker, θ_emotion, θ_gpt]
loss = L_tts(y, y_target)
∂loss/∂θ = [∂loss/∂θ_speaker, ∂loss/∂θ_emotion, ∂loss/∂θ_gpt]

# 문제:
# - Stage 2에서 학습한 θ_emotion이 변함
# - Speaker-emotion disentanglement 망가짐
```

#### With Freezing (Stage 3)
```python
# Only GPT trainable
θ_frozen = [θ_speaker, θ_emotion]  # Fixed
θ_trainable = [θ_gpt]  # Updated

loss = L_tts(y, y_target)
∂loss/∂θ_trainable = [∂loss/∂θ_gpt]  # Only this updates

# 장점:
# 1. θ_speaker, θ_emotion 보존 (Stage 1, 2 결과 유지)
# 2. 학습 안정성 향상 (fewer parameters to optimize)
# 3. Overfitting 방지
# 4. 학습 속도 향상 (~50% parameters)
```

---

## FAQ & Troubleshooting

### General

#### Q: 3 stage 모두 필수인가요?
**A**: 논문에 따르면 최상의 결과를 위해 3 stage 모두 권장합니다.
- Stage 1 only: 기본 TTS 가능, 감정 제어 제한적
- Stage 1+2: 감정 제어 가능, overfitting 가능성
- Stage 1+2+3: 최적 (감정 제어 + 안정성)

#### Q: 각 stage 학습 시간은?
**A**: 데이터셋 크기에 따라 다름 (2.7M samples 기준)
- Stage 1: ~3-5일 (여러 epoch)
- Stage 2: ~1-2일 (2 epochs)
- Stage 3: ~12-24시간 (1 epoch)

#### Q: Stage 2에서 speaker_acc가 80% 이상입니다
**A**: GRL이 효과가 없음. 다음 시도:
1. GRL lambda 증가: `export GRL_LAMBDA=2.0` 또는 `3.0`
2. Speaker loss weight 증가: `export SPEAKER_LOSS_WEIGHT=0.2`
3. Learning rate 확인: 너무 높으면 GRL 학습 불안정

#### Q: Stage 2에서 speaker_acc가 20% 이하입니다
**A**: Speaker classifier가 학습 안됨. 다음 시도:
1. Speaker mapping 확인: 500개 speaker가 충분한가?
2. Speaker loss weight 증가: `export SPEAKER_LOSS_WEIGHT=0.2`
3. Speaker mapping 재생성: min_samples 낮추기

#### Q: Stage 3에서 mel_loss가 증가합니다
**A**: Overfitting 또는 learning rate 과다. 다음 시도:
1. Learning rate 낮추기: `export LR=5e-5`
2. Epochs 줄이기: `export EPOCHS=0.5` (half epoch)
3. Early stopping: mel_loss 증가 시 중단

### Stage 1 Issues

#### Q: Out of memory error
**A**: Batch size 또는 gradient accumulation 조정
```bash
export BATCH_SIZE=4  # Default: 8
export GRAD_ACC=16   # Default: 8
# 실효 batch size = 4 * 16 = 64 (동일)
```

#### Q: mel_loss가 수렴하지 않습니다
**A**:
1. Learning rate 확인: 너무 낮으면 학습 느림
2. Warmup steps 증가: `export WARMUP_STEPS=10000`
3. Gradient clipping: `export GRAD_CLIP=1.0` (더 크게)

### Stage 2 Issues

#### Q: "Speaker mapping not found"
**A**: Speaker mapping 먼저 생성:
```bash
python tools/build_speaker_mapping.py \
    --manifest /mnt/sda1/emilia-yodas/KO_preprocessed/train_manifest.jsonl \
    --output /mnt/sda1/models/index-tts-ko/speaker_mapping.json \
    --top-k 500
```

#### Q: Real-time emo vec이 너무 느립니다
**A**: Fallback mode 사용 (pre-computed):
```bash
# train_ko_stage2.sh에서 --enable-stage2-realtime-emo 제거
# 단, GRL 효과는 감소
```

#### Q: Speaker classification loss가 발산합니다
**A**:
1. Speaker loss weight 낮추기: `export SPEAKER_LOSS_WEIGHT=0.05`
2. GRL lambda 낮추기: `export GRL_LAMBDA=0.5`
3. Mixed precision 비활성화: `--no-amp`

### Stage 3 Issues

#### Q: "Stage 2 checkpoint not found"
**A**: Stage 2를 먼저 완료해야 합니다:
```bash
ls -lh /mnt/sda1/models/index-tts-ko/stage2/checkpoints/best_model.pth
# 없으면 Stage 2 먼저 실행
```

#### Q: Frozen parameters가 제대로 적용되었는지 확인
**A**: 학습 시작 시 로그 확인:
```
[Stage 3] Freezing feature conditioners...
  ✅ Speaker conditioning encoder frozen
  ...
[Stage 3] Trainable parameters: 45,234,567 / 98,765,432 (45.8%)
```

---

## 참고 자료

### 논문
1. **IndexTTS2**: [arXiv:2506.21619v2](https://arxiv.org/abs/2506.21619)
2. **GRL (Domain-Adversarial Training)**: Ganin et al., 2016, JMLR

### 코드
- **Implementation**: `/mnt/sdc1/ws/workspace/monorepo/external/index-tts/`
- **Training Scripts**: `tools/train_ko_stage*.sh`
- **Main Trainer**: `trainers/train_gpt_v2.py`

### 문서
- **STAGE2_IMPLEMENTATION.md**: 상세 구현 내용
- **README.md**: 프로젝트 개요

---

## 요약

### 3-Stage Training Flow
```
📊 Prepare Data
    ↓
🎯 Stage 1: Basic TTS (수일)
    ├─ All components trainable
    ├─ Learn text → audio mapping
    └─ Checkpoint: best_model.pth
    ↓
🔧 Build Speaker Mapping (1회만)
    ├─ Select top 500 speakers
    └─ Output: speaker_mapping.json
    ↓
🎭 Stage 2: Emotion Disentanglement (1-2일)
    ├─ Speaker perceiver frozen
    ├─ GRL + Real-time emo vec
    ├─ Speaker-emotion separation
    └─ Checkpoint: stage2/best_model.pth
    ↓
🔒 Stage 3: Fine-tuning (12-24시간)
    ├─ All conditioners frozen
    ├─ GPT only trainable
    ├─ Quality refinement
    └─ Checkpoint: stage3/best_model.pth
    ↓
✅ Final Model
```

### Key Commands
```bash
# Environment
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate
cd /mnt/sdc1/ws/workspace/monorepo/external/index-tts

# Stage 1
./tools/train_ko_optimized_4090.sh

# Speaker Mapping (once)
python tools/build_speaker_mapping.py --manifest ... --output ...

# Stage 2
./tools/train_ko_stage2.sh

# Stage 3
./tools/train_ko_stage3.sh

# Monitor
tensorboard --logdir=/mnt/sda1/models/index-tts-ko
```

### Success Criteria
| Stage | Key Metric | Target |
|-------|-----------|--------|
| Stage 1 | mel_loss | < 2.0, converged |
| Stage 1 | mel_top1 | > 0.6 |
| Stage 2 | speaker_acc | 30-60% |
| Stage 2 | mel_loss | Similar to Stage 1 |
| Stage 3 | mel_loss | Similar or better than Stage 2 |
| Stage 3 | trainable% | ~40-50% |

---

**Happy Training! 🚀**

*Last updated: 2025-11-19*
