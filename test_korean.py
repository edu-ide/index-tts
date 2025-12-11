#!/usr/bin/env python3
"""한국어 학습 모델 테스트"""

from indextts.infer_v2 import IndexTTS2

# 한국어 모델 초기화
print("한국어 모델 로딩 중...")
tts = IndexTTS2(
    cfg_path="checkpoints/config_korean.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    device="cuda:0"
)

# 한국어 추론 실행
print("한국어 음성 생성 중...")
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="안녕하세요, 이것은 한국어 테스트입니다. 인덱스 티티에스 한국어 모델을 테스트하고 있습니다.",
    output_path="korean_output.wav",
    verbose=True
)

print("완료! korean_output.wav 파일을 확인하세요.")
