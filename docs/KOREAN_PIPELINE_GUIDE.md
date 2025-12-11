# IndexTTS-2 한국어 확장 파이프라인 가이드

> 기반 환경: `/mnt/sdc1/ws/workspace/.venv_indextts` (Python 3.10), 프로젝트 루트 `/mnt/sdc1/ws/workspace/monorepo`

## 0. 공통 준비

```bash
source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate
export PYTHONPATH=/mnt/sdc1/ws/workspace/monorepo/external/index-tts:$PYTHONPATH
```

## 1단계: 한국어 전용 SentencePiece 토크나이저 학습

- 입력: `/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl`
- 출력: `/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model`, `ko_bpe.vocab`

```bash
VOCAB_SIZE=16000 \
RETAIN_COUNT=0 \
external/index-tts/tools/ko_step1_train_tokenizer.sh

# 공용 토큰을 일정량 보존하고 싶다면 retain_count를 양수로 설정:
# RETAIN_COUNT=2000 external/index-tts/tools/ko_step1_train_tokenizer.sh
```

동작 흐름:
1. `train_bpe.py`가 한국어 manifest로 SentencePiece를 재학습해 `*_raw.model`을 생성합니다.
2. `RETAIN_COUNT > 0`이면 `rebuild_bpe_with_base.py`가 기존 IndexTTS-2 BPE의 선두 토큰을 지정 개수만큼 보존한 뒤 새 토큰으로 채워 최종 모델을 만듭니다.
3. `RETAIN_COUNT = 0`이면 새로 학습한 토크나이저를 바로 사용합니다.

필요 시 `RETAIN_COUNT`, `TARGET_SIZE`, `VOCAB_SIZE`, `CHAR_COVERAGE`, `MODEL_TYPE`, `TIMEOUT_SECS` 등을 환경변수로 조정하세요. 학습 후에는 `sentencepiece`로 vocab 구성을 확인합니다.

```bash
python - <<'PY'
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model')
print('vocab_size', sp.vocab_size())
print([sp.id_to_piece(i) for i in range(20)])
PY
```

공통 토큰(기호, 숫자 등)만 남고 나머지가 한국어 중심 서브워드인지 확인합니다. `retain_count`를 조절해 기본 토큰을 더 많이/적게 유지할 수 있습니다.

## 2단계: 데이터 전처리

- 입력: `ko_manifest_raw.jsonl`, `ko_bpe.model`, 기본 체크포인트(`config.yaml`, `gpt.pth`, `s2mel.pth`, `feat*.pt`, `wav2vec2bert_stats.pt`)
- 출력: `/mnt/sda1/emilia-yodas/KO_preprocessed/` 아래 `codes/`, `condition/`, `emo_vec/`, `text_ids/`, `train_manifest.jsonl`, `val_manifest.jsonl`, `stats.json`

```bash
external/index-tts/tools/ko_step2_preprocess.sh
```

`--skip-existing`이 기본 적용되어 있어 재실행 시 이미 처리된 샘플은 건너뛴다. GPU/CPU 자원에 맞춰 `BATCH_SIZE`, `NUM_WORKERS`, `DEVICE` 환경변수를 조정한다.
부분 전처리를 테스트하려면 `MAX_SAMPLES=<숫자>`를 지정해 처음 N개만 처리하거나, manifest를 분할해 여러 번 실행하면 된다.

## 3단계: GPT 프롬프트/타깃 페어 생성

- 입력: 2단계 산출물(`train_manifest.jsonl`, `val_manifest.jsonl`)
- 출력: `/mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl`, `gpt_pairs_val.jsonl`

```bash
external/index-tts/tools/ko_step3_generate_pairs.sh
```

기본적으로 각 타깃에 2개 프롬프트를 매칭하며(`PAIRS_PER_TARGET=2`), 덮어쓰기가 필요하면 `FORCE=1`을 설정한다.

## 4단계: GPT 미세조정

- 입력: `gpt_pairs_train.jsonl`, `gpt_pairs_val.jsonl`, 한국어 토크나이저, 기본 GPT 체크포인트
- 출력: `/mnt/sda1/models/index-tts-ko/checkpoints/` (기본값) 이하 학습된 `.pth`, 로그, TensorBoard 이벤트

```bash
BATCH_SIZE=4 \
GRAD_ACC=2 \
OUTPUT_DIR=/mnt/sda1/models/index-tts-ko/checkpoints \
AMP=1 \
BASE_TOKENIZER_MODEL=/mnt/sda1/models/IndexTTS-2/bpe.model \
RESUME=auto \
external/index-tts/tools/ko_step4_train_gpt.sh
```

학습 설정은 `EPOCHS`, `LEARNING_RATE`, `WARMUP_STEPS`, `TEXT_LOSS_WEIGHT`, `MEL_LOSS_WEIGHT` 등의 환경변수로 조절한다. 학습 로그는 `OUTPUT_DIR`에 저장되며 필요 시 TensorBoard로 확인한다.
`RESUME=auto` 또는 `RESUME=/path/to/checkpoint.pth`로 이어서 학습할 수도 있다.

## 추가 점검

- 토크나이저 교체 후에는 `trainers/train_gpt_v2.py`가 `resize_token_embeddings`를 호출해 새 vocab 크기를 반영하는지 확인한다.
- 모델 학습이 끝나면 기존 언어(영어/중국어) 발화 품질과 한국어 발화 품질을 모두 테스트해 회귀 여부를 판단한다.
- 필요 시 `TARGET_VOCAB`를 변경해 토큰 수(예: 14k, 16k)를 달리한 토크나이저를 반복 학습하며 비교 실험을 진행한다.
