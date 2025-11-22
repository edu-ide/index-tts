# 2025 Best LR Scheduler: WSD (Warmup-Stable-Decay)

사용자님의 질문("최고의 스케줄러 2025")에 대한 리서치 결과입니다.

## 👑 2025년 대세: WSD (Warmup-Stable-Decay) Scheduler
**"LLM 학습의 새로운 표준"**

최근 Llama 3, GPT-4 등 최신 거대언어모델(LLM) 학습에서 **Cosine Annealing을 대체**하고 있는 가장 핫한 스케줄러입니다.

### 1. WSD가 왜 최고인가?
기존 Cosine Annealing의 단점("학습 단계 수를 미리 정해야 함", "중간에 멈추거나 더 학습하기 애매함")을 완벽하게 해결했습니다.

*   **Warmup**: 초반에 LR을 천천히 올림 (안정성 확보)
*   **Stable (Constant)**: **학습의 80~90% 기간 동안 높은 LR을 유지**합니다.
    *   이때 모델이 가장 많이 똑똑해집니다.
    *   언제든지 학습을 더 길게 늘리거나 줄일 수 있습니다 (유연성).
*   **Decay**: 학습 종료 직전에 LR을 급격히 떨어뜨려 **Loss를 "뚝" 떨어뜨립니다.** (수렴)

### 2. Cosine Annealing vs WSD 비교

| 특징 | 📉 Cosine Annealing (기존 강자) | 🚀 WSD (2025 신흥 강자) |
| :--- | :--- | :--- |
| **학습 곡선** | 처음부터 끝까지 천천히 내려감 | **끝까지 높게 유지하다가 막판에 급락** |
| **성능 (Loss)** | 좋음 | **더 좋음 (최신 연구 결과)** |
| **유연성** | ❌ 학습 Step 미리 고정 필수 | ⭐ **언제든 종료/연장 가능** |
| **추천 대상** | 일반적인 딥러닝 | **LLM, TTS, 생성형 AI (IndexTTS2)** |

---

## 💡 제안: MARS + WSD 조합

사용자님이 선택하신 **MARS Optimizer**와 **WSD Scheduler**를 조합하면 **"2025년 최신 트렌드 조합"**이 됩니다.

1.  **Optimizer**: MARS (작은 배치 노이즈 잡음)
2.  **Scheduler**: WSD (학습 효율 극대화 + 유연성)

**구현 방법**:
현재 코드(`get_cosine_schedule_with_warmup`)를 **WSD 스타일**로 변경하면 됩니다.
(PyTorch의 `SequentialLR` 등을 사용하여 구현 가능)

**WSD 스케줄러를 적용해 드릴까요?**
(기존 Cosine보다 학습 후반부 성능이 더 좋을 확률이 높습니다.)
