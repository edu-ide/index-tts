#!/usr/bin/env python3
"""
IndexTTS WebUI with Model Comparison
원본 모델 vs 한국어 모델 비교 기능 추가
"""

import html
import json
import os
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI with Model Comparison",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'

# 모델 설정
MODEL_CONFIGS = {
    "원본 (Original - 12K tokens)": {
        "config": "checkpoints/config_original.yaml.backup",
        "gpt_checkpoint": "/mnt/sda1/models/IndexTTS-2/gpt.pth",
        "tokenizer": "/mnt/sda1/models/IndexTTS-2/bpe.model",
        "description": "중국어+영어 원본 모델 (12000 tokens)"
    },
    "한국어 (Korean - 16K tokens)": {
        "config": "checkpoints/config_korean.yaml",
        "gpt_checkpoint": "/mnt/sda1/models/index-tts-ko/checkpoints/latest.pth",
        "tokenizer": "/mnt/sda1/models/IndexTTS-2/tokenizer_ko/ko_bpe.model",
        "description": "한국어 학습 모델 (16000 tokens, step 297000)"
    }
}

# 현재 로드된 모델
current_model_name = None
tts = None

def load_model(model_name):
    global current_model_name, tts

    if current_model_name == model_name and tts is not None:
        return f"✓ 이미 '{model_name}' 모델이 로드되어 있습니다."

    model_config = MODEL_CONFIGS[model_name]

    print(f"\n{'='*60}")
    print(f"모델 전환 중: {model_name}")
    print(f"  Config: {model_config['config']}")
    print(f"  GPT: {model_config['gpt_checkpoint']}")
    print(f"  Tokenizer: {model_config['tokenizer']}")
    print(f"{'='*60}\n")

    # 기존 모델 해제
    if tts is not None:
        del tts
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 새 모델 로드
    tts = IndexTTS2(
        model_dir=cmd_args.model_dir,
        cfg_path=model_config['config'],
        use_fp16=cmd_args.fp16,
        use_deepspeed=cmd_args.deepspeed,
        use_cuda_kernel=cmd_args.cuda_kernel,
    )

    current_model_name = model_name

    return f"✓ '{model_name}' 모델 로드 완료!\n{model_config['description']}"

# 초기 모델 로드
load_model("원본 (Original - 12K tokens)")

EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def gen_single(prompt, text, emo_control_method, emo_ref, emo_weight, emo_text,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_random, max_text_tokens_per_segment, model_name, **kwargs):

    # 선택된 모델로 전환
    load_model(model_name)

    if prompt is None:
        return gr.update(value=None, visible=True)

    output_path = f"outputs/spk_{int(time.time())}.wav"

    emo_ref_path = emo_ref
    if emo_control_method == 0:
        emo_ref_path = None
    if emo_control_method == 1:
        pass
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"[{model_name}] Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(
        spk_audio_prompt=prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method==3),
        emo_text=emo_text,
        use_random=emo_random,
        verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **kwargs
    )
    return gr.update(value=output, visible=True)

with gr.Blocks(title="IndexTTS Model Comparison") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2 Model Comparison: 원본 vs 한국어 모델</center></h2>
    <p align="center">원본 모델과 한국어 학습 모델을 비교할 수 있습니다</p>
    ''')

    with gr.Tab("음성 생성 및 모델 비교"):
        # 모델 선택
        with gr.Row():
            model_selector = gr.Radio(
                choices=list(MODEL_CONFIGS.keys()),
                value="원본 (Original - 12K tokens)",
                label="🔄 모델 선택",
                info="모델을 선택하면 자동으로 전환됩니다"
            )
            model_status = gr.Textbox(
                label="모델 상태",
                value="✓ 원본 모델 로드됨",
                interactive=False,
                lines=2
            )

        with gr.Row():
            prompt_audio = gr.Audio(
                label="음색 참조 오디오",
                sources=["upload", "microphone"],
                type="filepath"
            )
            with gr.Column():
                input_text_single = gr.TextArea(
                    label="텍스트",
                    placeholder="생성할 텍스트를 입력하세요",
                    info="원본: 중국어/영어, 한국어: 한국어"
                )
                gen_button = gr.Button("🎙️ 생성", variant="primary")
            output_audio = gr.Audio(label="생성 결과", visible=True)

        with gr.Accordion("고급 설정", open=False):
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],
                    label="감정 제어 방식"
                )

            with gr.Group(visible=False) as emotion_reference_group:
                emo_upload = gr.Audio(label="감정 참조 오디오", type="filepath")

            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(label="감정 랜덤 샘플링", value=False)

            with gr.Group(visible=False) as emotion_vector_group:
                gr.Markdown("**감정 벡터 조정** (0.0 ~ 1.0)")
                with gr.Row():
                    vec1 = gr.Slider(0, 1, 0, step=0.05, label="Happy")
                    vec2 = gr.Slider(0, 1, 0, step=0.05, label="Angry")
                    vec3 = gr.Slider(0, 1, 0, step=0.05, label="Sad")
                    vec4 = gr.Slider(0, 1, 0, step=0.05, label="Afraid")
                with gr.Row():
                    vec5 = gr.Slider(0, 1, 0, step=0.05, label="Disgusted")
                    vec6 = gr.Slider(0, 1, 0, step=0.05, label="Melancholic")
                    vec7 = gr.Slider(0, 1, 0, step=0.05, label="Surprised")
                    vec8 = gr.Slider(0, 1, 0, step=0.05, label="Calm")

            emo_weight = gr.Slider(0, 1, 1.0, step=0.05, label="감정 강도")
            emo_text = gr.Textbox(label="감정 텍스트 설명", value="", visible=False)
            max_text_tokens_per_segment = gr.Number(value=120, label="세그먼트당 최대 토큰")

        # 모델 전환 이벤트
        model_selector.change(
            fn=load_model,
            inputs=[model_selector],
            outputs=[model_status]
        )

        # 감정 제어 UI 업데이트
        def update_emotion_ui(method):
            return {
                emotion_reference_group: gr.update(visible=(method == 1)),
                emotion_vector_group: gr.update(visible=(method == 2)),
                emotion_randomize_group: gr.update(visible=(method in [2, 3])),
                emo_text: gr.update(visible=(method == 3))
            }

        emo_control_method.change(
            fn=update_emotion_ui,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group, emotion_vector_group,
                    emotion_randomize_group, emo_text]
        )

        # 생성 버튼
        gen_button.click(
            fn=gen_single,
            inputs=[
                prompt_audio, input_text_single, emo_control_method,
                emo_upload, emo_weight, emo_text,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_random, max_text_tokens_per_segment, model_selector
            ],
            outputs=[output_audio]
        )

    with gr.Tab("📖 사용 가이드"):
        gr.Markdown("""
        ## 모델 비교 가이드

        ### 원본 모델 (Original - 12K tokens)
        - **언어**: 중국어, 영어
        - **토큰**: 12,000개
        - **감정 제어**: ✅ 매우 우수 (중국어/영어 감정 데이터 학습)
        - **추천 텍스트**: 영어 또는 중국어

        ### 한국어 모델 (Korean - 16K tokens)
        - **언어**: 한국어
        - **토큰**: 16,000개 (한국어 전용)
        - **학습**: Step 297,000까지 fine-tuning
        - **감정 제어**: ⚠️ 제한적 (원본 감정 매트릭스 사용)
        - **추천 텍스트**: 한국어만

        ### 비교 테스트 방법

        1. **같은 음색 참조 오디오 사용**
        2. **두 모델로 같은 의미의 텍스트 생성**
           - 원본: "Hello, this is a test."
           - 한국어: "안녕하세요, 이것은 테스트입니다."
        3. **같은 감정 벡터 설정** (예: Sad = 0.8)
        4. **결과 비교**

        ### 감정 벡터 추천 값
        - **Happy**: 0.6~0.8
        - **Sad**: 0.6~0.8
        - **Calm**: 0.5~0.7
        - **Surprised**: 0.7~0.9
        """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=False
    )
