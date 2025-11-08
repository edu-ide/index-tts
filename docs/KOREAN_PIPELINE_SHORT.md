# IndexTTS-2 한국어 확장 실행 순서 (리맵 포함)

## 0. 준비

```bash
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate
export PYTHONPATH=/mnt/sdc1/ws/workspace/monorepo/external/index-tts:$PYTHONPATH
```

## 1. 기존 전처리 산출물 초기화

```bash
rm -rf /mnt/sda1/emilia-yodas/KO_preprocessed
```

필요하면 백업한 뒤 제거하세요.

## 2. 토크나이저 재학습 (완전 재학습 방식)

```bash
VOCAB_SIZE=16000 RETAIN_COUNT=0 \
external/index-tts/tools/ko_step1_train_tokenizer.sh
```

필요 시 `VOCAB_SIZE`, `CHAR_COVERAGE`, `MODEL_TYPE` 등을 환경변수로 조정하세요.

## 3. 전처리 재실행

```bash
external/index-tts/tools/ko_step2_preprocess.sh
```

`--skip-existing`이 기본 활성화되어 있어 중단 후 재시작 가능합니다.
테스트용으로 일부만 처리하려면 `MAX_SAMPLES=<숫자>`를 지정하거나 manifest를 나눠 실행하세요.

## 4. GPT 페어 생성

```bash
external/index-tts/tools/ko_step3_generate_pairs.sh
```

필요하면 `FORCE=1`로 덮어쓰기를 허용하세요.

## 5. GPT 미세조정 (토큰 리맵 포함)

```bash
BATCH_SIZE=4 GRAD_ACC=2 AMP=1 \
BASE_TOKENIZER_MODEL=/mnt/sda1/models/IndexTTS-2/bpe.model \
RESUME=auto \
external/index-tts/tools/ko_step4_train_gpt.sh
```

`BASE_TOKENIZER_MODEL`을 지정하면 `train_gpt_v2.py`에서 동일 문자열 토큰의 임베딩을 기존 모델에서 새로운 위치로 복사합니다. 배치 크기, 학습률 등은 환경 변수로 필요에 맞게 조정하세요.
`RESUME=auto` 또는 `RESUME=/경로/model_stepXXXX.pth`로 연속 학습을 이어갈 수 있습니다.

---

모든 단계가 완료되면 새 토크나이저와 매핑된 GPT 체크포인트로 한국어 모델을 fine-tuning할 준비가 끝납니다.*** End Patch
