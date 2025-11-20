#!/usr/bin/env python3
"""간단한 IndexTTS-2 추론 테스트"""

from indextts.infer_v2 import IndexTTS2

# 모델 초기화
print("모델 로딩 중...")
tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    device="cuda:0"
)

# 추론 실행
print("음성 생성 중...")
tts.infer(
    spk_audio_prompt='examples/voice_01.wav',
    text="Hello, this is a test. 안녕하세요, 테스트입니다.",
    output_path="test_output.wav",
    verbose=True
)

print("완료! test_output.wav 파일을 확인하세요.")
