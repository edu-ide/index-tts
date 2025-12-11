# IndexTTS2 구현 검증 보고서

**검증 대상**: `indextts/gpt/model_v2.py`, `trainers/train_gpt_v2.py` 및 관련 모듈
**참조 논문**: IndexTTS2 (arXiv:2506.21619v2)

## ✅ 요약
현재 코드베이스는 **IndexTTS2 논문의 핵심 기능, 특히 Stage 2 (Emotion Disentanglement)를 충실히 구현**하고 있는 것으로 확인됩니다.

---

## 🔍 상세 검증 결과

### 1. 모델 아키텍처 (Model Architecture)
*   **Backbone**: GPT-2 기반의 Autoregressive Transformer (`GPT2InferenceModel`)를 사용하고 있으며, 이는 IndexTTS 시리즈의 표준 구조와 일치합니다.
*   **Inputs**: Text Token, Mel Code (Semantic Token), Conditioning Latent, Emotion Vector를 모두 처리하도록 설계되었습니다.
*   **Conditioning**:
    *   `ConditioningEncoder`: 화자/스타일 정보를 위한 Perceiver/Conformer 구조가 구현되어 있습니다.
    *   `EmoConditioningEncoder`: 감정 정보를 위한 별도의 인코더가 존재합니다.

### 2. Stage 2: 감정-화자 분리 (Emotion-Speaker Disentanglement)
논문의 핵심 기여 중 하나인 "Adversarial Training을 통한 화자와 감정의 분리"가 정확히 구현되어 있습니다.

*   **Gradient Reversal Layer (GRL)**:
    *   `indextts/gpt/gradient_reversal.py`에 Ganin et al. (2016) 방식이 구현됨.
    *   `UnifiedVoice` 모델에서 감정 벡터(`emo_vec`)에 GRL을 적용하여, 감정 인코더가 화자 정보를 "잊게" 만듭니다.
*   **Speaker Classifier**:
    *   `indextts/gpt/speaker_classifier.py`에 구현됨.
    *   GRL을 통과한 감정 벡터로부터 화자를 예측하도록 학습됩니다.
    *   **목적**: 감정 벡터에 화자 정보가 남아있다면 Classifier가 맞출 것이고, GRL은 이를 방해하여(Gradient Reversal) 감정 벡터가 순수한 "감정"만 담도록 유도합니다.
*   **Loss Function**:
    *   `train_gpt_v2.py`에서 `speaker_loss`가 계산되고 총 Loss에 합산됩니다.

### 3. 학습 파이프라인 (Training Pipeline)
*   **Optimizers**: AdamW와 Prodigy를 지원하며, 최근 연구 트렌드를 반영하고 있습니다.
*   **Efficiency**: Flash Attention 2, `torch.compile`, Mixed Precision (AMP) 등 최신 최적화 기법이 적용되어 있습니다.

---

## ⚠️ 주의/제안 사항

1.  **Stage 2 활성화**: 학습 시 `--enable-grl` 플래그를 반드시 켜야 Stage 2 (Disentanglement) 학습이 진행됩니다.
2.  **Speaker Mapping**: `--speaker-mapping` 인자를 통해 화자 ID 매핑 파일을 제공해야 Speaker Classifier가 정상 작동합니다.
3.  **Optimizer**: 앞서 논의한 대로, 현재 배치 사이즈(8)에서는 **MARS** 또는 **AdEMAMix** 도입을 통해 학습 안정성을 더 높일 수 있습니다.

## 🏁 결론
**구현 상태: 우수 (Pass)**
논문에서 제시하는 아키텍처와 학습 방법론(특히 Stage 2 Adversarial Training)이 코드에 정확히 반영되어 있습니다.
