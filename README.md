## Unofficial IndexTTS v2 Training Repo
> Loop and trainer implemented using Codex CLI and guided prompts
  - Train new languages by extending existing tokenizer
    - tools\tokenizer\train_bpe.py and tools\tokenizer\extend_bpe.py
  - Preprocess data to extract speaker embeddings for timbre, emotion, text, and mel tokens
    - tools\preprocess_data.py and tools\preprocess_multiproc.py (multiproc is an attempt to make it run faster, there are issues with it though crashing)
  - Create prompt/target pairs which is required for how IndexTTS2 trains in order to learn how to speak with speaker timbre while separating emotion (emotion has not yet been investigated)
    - tools\generate_gpt_pairs.py
  - Train/finetune the gpt model to learn to predict tokens for the language
    - trainers\train_gpt_v2.py and train.bat

The code here works and Japanese was *mostly* correct shown here: https://www.youtube.com/watch?v=47V7lS-HUpo (this model was trained on 1100 hours of audio for about 1.5 epochs)

### Training References (Korean)

- **전체 파이프라인 개요**: 토크나이저 확장 → 데이터 전처리 → 프롬프트/타깃 페어 생성 → GPT 미세조정 순으로 진행합니다. 관련 설명은 README 도입부와 `external/index-tts/README.md` 상단 요약을 참고하세요.
- **환경 구축**: `docs/README_zh.md`의 “환경配置” 절(예: 123, 158, 188행 근처)이 uv 기반 가상환경, git-lfs, Hugging Face 자산 다운로드 절차를 상세히 다룹니다. 네트워크가 느릴 경우 HF 미러 설정(`docs/README_zh.md` 205행)을 권장합니다.
- **토크나이저**: `tools/tokenizer/train_bpe.py`(12, 43행)으로 새 BPE를 학습하거나 `tools/tokenizer/extend_bpe.py`(1, 84행)로 기존 모델에 한국어 토큰을 추가할 수 있습니다.
- **데이터 전처리**: `tools/preprocess_data.py`(9, 90, 421행)는 텍스트 정규화 → SeamlessM4T/Wav2Vec2 피처 추출 → MaskGCT 양자화 → GPT 기반 컨디셔닝·감정 벡터 추출을 수행하며, MaskGCT 가중치를 Hugging Face에서 받아옵니다(639행). 대용량 처리는 `tools/preprocess_multiproc.py`(1, 32, 98행)로 병렬 실행을 지원합니다.
- **프롬프트/타깃 페어 생성**: `tools/build_gpt_prompt_pairs.py`(1, 68행)와 `tools/generate_gpt_pairs.py`(1, 74행)을 활용해 화자별 프롬프트·타깃 매니페스트(`gpt_pairs_*.jsonl`)를 생성합니다.
- **GPT 미세조정**: `trainers/train_gpt_v2.py`(3, 46행)이 학습 진입점이며, 페어 매니페스트의 필수 필드 검증(200행)과 TensorBoard, 체크포인트 기록(585행)을 자동으로 처리합니다.
- **필수 체크포인트**: `checkpoints/config.yaml`(14, 110행)이 참조하는 `gpt.pth`, `wav2vec2bert_stats.pt`, `s2mel.pth`, `feat1.pt`, `feat2.pt` 등을 Hugging Face `IndexTeam/IndexTTS-2`에서 내려받아 동일한 디렉터리 구조로 배치해야 합니다.

The latest updates are done with a focus on training a multilingual model which shows promise, while mostly retaining the base model abilities to speak English and Chinese. Emotion finetuning has not been investigated yet and it seems that full finetuning does not mess up the base emotion capabilities of the model.

<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>

<div align="center">
<a href="docs/README_zh.md" style="font-size: 24px">简体中文</a> | 
<a href="README.md" style="font-size: 24px">English</a>
</div>

## 👉🏻 IndexTTS2 👈🏻

<center><h3>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h3></center>

[![IndexTTS2](assets/IndexTTS2_banner.png)](assets/IndexTTS2_banner.png)


<div align="center">
  <a href='https://arxiv.org/abs/2506.21619'>
    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv'/>
  </a>
  <br/>
  <a href='https://github.com/index-tts/index-tts'>
    <img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'/>
  </a>
  <a href='https://index-tts.github.io/index-tts2.github.io/'>
    <img src='https://img.shields.io/badge/GitHub-Demo-orange?logo=github'/>
  </a>
  <br/>
  <a href='https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/HuggingFace-Demo-blue?logo=huggingface'/>
  </a>
  <a href='https://huggingface.co/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' />
  </a>
  <br/>
  <a href='https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/ModelScope-Demo-purple?logo=modelscope'/>
  </>
  <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope'/>
  </a>
</div>


### Abstract

Existing autoregressive large-scale text-to-speech (TTS) models have advantages in speech naturalness, but their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This becomes a significant limitation in applications requiring strict audio-visual synchronization, such as video dubbing.

This paper introduces IndexTTS2, which proposes a novel, general, and autoregressive model-friendly method for speech duration control.

The method supports two generation modes: one explicitly specifies the number of generated tokens to precisely control speech duration; the other freely generates speech in an autoregressive manner without specifying the number of tokens, while faithfully reproducing the prosodic features of the input prompt.

Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control over timbre and emotion. In the zero-shot setting, the model can accurately reconstruct the target timbre (from the timbre prompt) while perfectly reproducing the specified emotional tone (from the style prompt).

To enhance speech clarity in highly emotional expressions, we incorporate GPT latent representations and design a novel three-stage training paradigm to improve the stability of the generated speech. Additionally, to lower the barrier for emotional control, we designed a soft instruction mechanism based on text descriptions by fine-tuning Qwen3, effectively guiding the generation of speech with the desired emotional orientation.

Finally, experimental results on multiple datasets show that IndexTTS2 outperforms state-of-the-art zero-shot TTS models in terms of word error rate, speaker similarity, and emotional fidelity. Audio samples are available at: <a href="https://index-tts.github.io/index-tts2.github.io/">IndexTTS2 demo page</a>.

**Tips:** Please contact the authors for more detailed information. For commercial usage and cooperation, please contact <u>indexspeech@bilibili.com</u>.


### Feel IndexTTS2

<div align="center">

**IndexTTS2: The Future of Voice, Now Generating**

[![IndexTTS2 Demo](assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

*Click the image to watch the IndexTTS2 introduction video.*

</div>


### Contact

QQ Group：663272642(No.4) 1013410623(No.5)  \
Discord：https://discord.gg/uT32E7KDmy  \
Email：indexspeech@bilibili.com  \
You are welcome to join our community! 🌏  \
欢迎大家来交流讨论！

> [!CAUTION]
> Thank you for your support of the bilibili indextts project!
> Please note that the **only official channel** maintained by the core team is: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).
> ***Any other websites or services are not official***, and we cannot guarantee their security, accuracy, or timeliness.
> For the latest updates, please always refer to this official repository.


## 📣 Updates

- `2025/09/08` 🔥🔥🔥  We release **IndexTTS-2** to the world!
    - The first autoregressive TTS model with precise synthesis duration control, supporting both controllable and uncontrollable modes. <i>This functionality is not yet enabled in this release.</i>
    - The model achieves highly expressive emotional speech synthesis, with emotion-controllable capabilities enabled through multiple input modalities.
- `2025/05/14` 🔥🔥 We release **IndexTTS-1.5**, significantly improving the model's stability and its performance in the English language.
- `2025/03/25` 🔥 We release **IndexTTS-1.0** with model weights and inference code.
- `2025/02/12` 🔥 We submitted our paper to arXiv, and released our demos and test sets.


## 🖥️ Neural Network Architecture

Architectural overview of IndexTTS2, our state-of-the art speech model:

<picture>
  <img src="assets/IndexTTS2.png"  width="800"/>
</picture>


The key contributions of **IndexTTS2** are summarized as follows:

 - We propose a duration adaptation scheme for autoregressive TTS models. IndexTTS2 is the first autoregressive zero-shot TTS model to combine precise duration control with natural duration generation, and the method is scalable for any autoregressive large-scale TTS model.  
 - The emotional and speaker-related features are decoupled from the prompts, and a feature fusion strategy is designed to maintain semantic fluency and pronunciation clarity during emotionally rich expressions. Furthermore, a tool was developed for emotion control, utilizing natural language descriptions for the benefit of users.  
 - To address the lack of highly expressive speech data, we propose an effective training strategy, significantly enhancing the emotional expressiveness of zeroshot TTS to State-of-the-Art (SOTA) level.  
 - We will publicly release the code and pre-trained weights to facilitate future research and practical applications.  


## Model Download

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁 IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2) | [IndexTTS-2](https://modelscope.cn/models/IndexTeam/IndexTTS-2) |
| [IndexTTS-1.5](https://huggingface.co/IndexTeam/IndexTTS-1.5) | [IndexTTS-1.5](https://modelscope.cn/models/IndexTeam/IndexTTS-1.5) |
| [IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |


## Usage Instructions

### ⚙️ Environment Setup

1. Ensure that you have both [git](https://git-scm.com/downloads)
   and [git-lfs](https://git-lfs.com/) on your system.

The Git-LFS plugin must also be enabled on your current user account:

```bash
git lfs install
```

2. Download this repository:

```bash
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs pull  # download large repository files
```

3. Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).
   It is *required* for a reliable, modern installation environment.

> [!TIP]
> **Quick & Easy Installation Method:**
> 
> There are many convenient ways to install the `uv` command on your computer.
> Please check the link above to see all options. Alternatively, if you want
> a very quick and easy method, you can install it as follows:
> 
> ```bash
> pip install -U uv
> ```

> [!WARNING]
> We **only** support the `uv` installation method. Other tools, such as `conda`
> or `pip`, don't provide any guarantees that they will install the correct
> dependency versions. You will almost certainly have *random bugs, error messages,*
> ***missing GPU acceleration**, and various other problems* if you don't use `uv`.
> Please *do not report any issues* if you use non-standard installations, since
> almost all such issues are invalid.
> 
> Furthermore, `uv` is [up to 115x faster](https://github.com/astral-sh/uv/blob/main/BENCHMARKS.md)
> than `pip`, which is another *great* reason to embrace the new industry-standard
> for Python project management.

4. Install required dependencies:

We use `uv` to manage the project's dependency environment. The following command
will *automatically* create a `.venv` project-directory and then installs the correct
versions of Python and all required dependencies:

```bash
uv sync --all-extras
```

If the download is slow, please try a *local mirror*, for example any of these
local mirrors in China (choose one mirror from the list below):

```bash
uv sync --all-extras --default-index "https://mirrors.aliyun.com/pypi/simple"

uv sync --all-extras --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

> [!TIP]
> **Available Extra Features:**
> 
> - `--all-extras`: Automatically adds *every* extra feature listed below. You can
>   remove this flag if you want to customize your installation choices.
> - `--extra webui`: Adds WebUI support (recommended).
> - `--extra deepspeed`: Adds DeepSpeed support (may speed up inference on some
>   systems).

> [!IMPORTANT]
> **Important (Windows):** The DeepSpeed library may be difficult to install for
> some Windows users. You can skip it by removing the `--all-extras` flag. If you
> want any of the other extra features above, you can manually add their specific
> feature flags instead.
> 
> **Important (Linux/Windows):** If you see an error about CUDA during the installation,
> please ensure that you have installed NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
> version **12.8** (or newer) on your system.

5. Download the required models via [uv tool](https://docs.astral.sh/uv/guides/tools/#installing-tools):

Download via `huggingface-cli`:

```bash
uv tool install "huggingface-hub[cli,hf_xet]"

hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

Or download via `modelscope`:

```bash
uv tool install "modelscope"

modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

> [!IMPORTANT]
> If the commands above aren't available, please carefully read the `uv tool`
> output. It will tell you how to add the tools to your system's path.

> [!NOTE]
> In addition to the above models, some small models will also be automatically
> downloaded when the project is run for the first time. If your network environment
> has slow access to HuggingFace, it is recommended to execute the following
> command before running the code:
> 
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```


#### 🖥️ Checking PyTorch GPU Acceleration

If you need to diagnose your environment to see which GPUs are detected,
you can use our included utility to check your system:

```bash
uv run tools/gpu_check.py
```


### 🔥 IndexTTS2 Quickstart

#### 🌐 Web Demo

```bash
uv run webui.py
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

You can also adjust the settings to enable features such as FP16 inference (lower
VRAM usage), DeepSpeed acceleration, compiled CUDA kernels for speed, etc. All
available options can be seen via the following command:

```bash
uv run webui.py -h
```

Have fun!

> [!IMPORTANT]
> It can be very helpful to use **FP16** (half-precision) inference. It is faster
> and uses less VRAM, with a very small quality loss.
> 
> **DeepSpeed** *may* also speed up inference on some systems, but it could also
> make it slower. The performance impact is highly dependent on your specific
> hardware, drivers and operating system. Please try with and without it,
> to discover what works best on your personal system.
> 
> Lastly, be aware that *all* `uv` commands will **automatically activate** the correct
> per-project virtual environments. Do *not* manually activate any environments
> before running `uv` commands, since that could lead to dependency conflicts!


#### 📝 Using IndexTTS2 in Python

To run scripts, you *must* use the `uv run <file.py>` command to ensure that
the code runs inside your current "uv" environment. It *may* sometimes also be
necessary to add the current directory to your `PYTHONPATH`, to help it find
the IndexTTS modules.

Example of running a script via `uv`:

```bash
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py
```

Here are several examples of how to use IndexTTS2 in your own scripts:

1. Synthesize new speech with a single reference audio file (voice cloning):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```

2. Using a separate, emotional reference audio file to condition the speech synthesis:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

3. When an emotional reference audio file is specified, you can optionally set
   the `emo_alpha` to adjust how much it affects the output.
   Valid range is `0.0 - 1.0`, and the default value is `1.0` (100%):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "酒楼丧尽天良，开始借机竞拍房间，哎，一群蠢货。"
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

4. It's also possible to omit the emotional reference audio and instead provide
   an 8-float list specifying the intensity of each emotion, in the following order:
   `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
   You can additionally use the `use_random` parameter to introduce stochasticity
   during inference; the default is `False`, and setting it to `True` enables
   randomness:

> [!NOTE]
> Enabling random sampling will reduce the voice cloning fidelity of the speech
> synthesis.

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "哇塞！这个爆率也太高了！欧皇附体了！"
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)
```

5. Alternatively, you can enable `use_emo_text` to guide the emotions based on
   your provided `text` script. Your text script will then automatically
   be converted into emotion vectors.
   It's recommended to use `emo_alpha` around 0.6 (or lower) when using the text
   emotion modes, for more natural sounding speech.
   You can introduce randomness with `use_random` (default: `False`;
   `True` enables randomness):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

6. It's also possible to directly provide a specific text emotion description
   via the `emo_text` parameter. Your emotion text will then automatically be
   converted into emotion vectors. This gives you separate control of the text
   script and the text emotion description:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "快躲起来！是他要来了！他要来抓我们了！"
emo_text = "你吓死我了！你是鬼吗？"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```

> [!TIP]
> **Pinyin Usage Notes:**
> 
> IndexTTS2 still supports mixed modeling of Chinese characters and Pinyin.
> When you need precise pronunciation control, please provide text with specific Pinyin annotations to activate the Pinyin control feature.
> Note that Pinyin control does not work for every possible consonant–vowel combination; only valid Chinese Pinyin cases are supported.
> For the full list of valid entries, please refer to `checkpoints/pinyin.vocab`.
>
> Example:
> ```
> 之前你做DE5很好，所以这一次也DEI3做DE2很好才XING2，如果这次目标完成得不错的话，我们就直接打DI1去银行取钱。
> ```

### 🚀 한국어 모델 빠른 실행 (Quick Start for Korean Model)

한국어 모델 사용을 위한 편의 스크립트들입니다.

#### 1. WebUI 서버 시작

```bash
./start_webui.sh
```

- **기능**: Gradio 기반 웹 인터페이스 실행
- **서버 주소**: http://0.0.0.0:7860
- **GPU**: RTX 3060 (12GB) 사용 (CUDA_VISIBLE_DEVICES=1)
- **모델 위치**: ~/models/index-tts-ko/checkpoints
- **모델 크기**: 3.3GB (추론 전용)

#### 2. REST API 서버 시작

```bash
./start_api.sh
```

- **기능**: FastAPI 기반 REST API 서버 실행
- **API 주소**: http://0.0.0.0:8765
- **API 문서**: http://0.0.0.0:8765/docs
- **GPU**: RTX 3060 (12GB) 사용 (CUDA_VISIBLE_DEVICES=1)

**API 엔드포인트:**
- `GET /`: Health check
- `GET /health`: 모델 로딩 상태 확인
- `POST /tts`: JSON 요청으로 TTS 생성 (Base64 인코딩된 오디오 반환)
- `POST /tts_file`: Form 데이터로 TTS 생성 (WAV 파일 직접 반환)

**API 사용 예제 (curl):**

```bash
# 기본 TTS 요청
curl -X POST "http://localhost:8765/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 반갑습니다!",
    "prompt_audio_path": "examples/voice_busan.wav",
    "temperature": 0.7,
    "top_p": 0.9
  }'

# 파일로 직접 받기
curl -X POST "http://localhost:8765/tts_file" \
  -F "text=안녕하세요, 반갑습니다!" \
  -F "prompt_audio=@examples/voice_busan.wav" \
  -o output.wav
```

**Python 사용 예제:**

```python
import requests
import base64

# TTS 생성
response = requests.post(
    "http://localhost:8765/tts",
    json={
        "text": "안녕하세요, 반갑습니다!",
        "prompt_audio_path": "examples/voice_busan.wav",
        "temperature": 0.7,
        "top_p": 0.9,
        "emo_weight": 1.0
    }
)

result = response.json()

# Base64 디코딩 후 저장
audio_data = base64.b64decode(result["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_data)

print(f"Duration: {result['duration']:.2f}s")
print(f"Inference time: {result['inference_time']:.2f}s")
```

#### 3. 모델 업데이트

```bash
./update_model.sh
```

- **기능**: NFS에서 최신 `best_model.pth`를 가져와 추론용 모델로 자동 변환
- **동작 과정**:
  1. NFS (`/mnt/models/index-tts-ko/checkpoints/best_model.pth`)에서 최신 모델 확인
  2. 타임스탬프 비교 후 필요시에만 복사 (최신이면 스킵)
  3. 추론 전용 모델 추출 (7.3GB → 3.3GB, 54.9% 감소)
  4. `gpt.pth` 심볼릭 링크 자동 업데이트

- **사용 시점**: 학습 후 새 모델이 업데이트될 때마다
- **참고**: 서버는 자동으로 재시작되지 않으므로, 업데이트 후 `start_webui.sh` 또는 `start_api.sh`를 다시 실행하세요

#### 스크립트 권한 설정

처음 사용 시 실행 권한을 부여하세요:

```bash
chmod +x start_webui.sh start_api.sh update_model.sh
```

#### 참조 오디오 추가

WebUI 예제에 새로운 참조 오디오를 추가하려면:

1. 오디오 파일을 `examples/` 디렉토리에 복사 (WAV 형식, 22050Hz, mono 권장)
2. `examples/cases.jsonl` 파일에 항목 추가:

```json
{"prompt_audio":"your_audio.wav","text":"테스트 문장입니다.","emo_mode":0}
```

3. WebUI 재시작

---

### Legacy: IndexTTS1 User Guide

You can also use our previous IndexTTS1 model by importing a different module:

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！比如说，现在正在说话的其实是B站为我现场复刻的数字分身，简直就是平行宇宙的另一个我了。如果大家也想体验更多深入的AIGC功能，可以访问 bilibili studio，相信我，你们也会吃惊的。"
tts.infer(voice, text, 'gen.wav')
```

For more detailed information, see [README_INDEXTTS_1_5](archive/README_INDEXTTS_1_5.md),
or visit the IndexTTS1 repository at <a href="https://github.com/index-tts/index-tts/tree/v1.5.0">index-tts:v1.5.0</a>.


## Our Releases and Demos

### IndexTTS2: [[Paper]](https://arxiv.org/abs/2506.21619); [[Demo]](https://index-tts.github.io/index-tts2.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-2-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)

### IndexTTS1: [[Paper]](https://arxiv.org/abs/2502.05512); [[Demo]](https://index-tts.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS)


## Acknowledgements

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)

## Contributors in Bilibili
We sincerely thank colleagues from different roles at Bilibili, whose combined efforts made the IndexTTS series possible.

### Core Authors
 - **Wei Deng** - Core author; Initiated the IndexTTS project, led the development of the IndexTTS1 data pipeline, model architecture design and training, as well as iterative optimization of the IndexTTS series of models, focusing on fundamental capability building and performance optimization.
 - **Siyi Zhou** – Core author; in IndexTTS2, led model architecture design and training pipeline optimization, focusing on key features such as multilingual and emotional synthesis.
 - **Jingchen Shu** - Core author; worked on overall architecture design, cross-lingual modeling solutions, and training strategy optimization, driving model iteration.
 - **Xun Zhou** - Core author; worked on cross-lingual data processing and experiments, explored multilingual training strategies, and contributed to audio quality improvement and stability evaluation.
 - **Jinchao Wang** - Core author; worked on model development and deployment, building the inference framework and supporting system integration.
 - **Yiquan Zhou** - Core author; contributed to model experiments and validation, and proposed and implemented text-based emotion control.
 - **Yi He** - Core author; contributed to model experiments and validation.
 - **Lu Wang** – Core author; worked on data processing and model evaluation, supporting model training and performance verification.

### Technical Contributors
 - **Yining Wang** - Supporting contributor; contributed to open-source code implementation and maintenance, supporting feature adaptation and community release.
 - **Yong Wu** - Supporting contributor; worked on data processing and experimental support, ensuring data quality and efficiency for model training and iteration.
 - **Yaqin Huang** – Supporting contributor; contributed to systematic model evaluation and effect tracking, providing feedback to support iterative improvements.
 - **Yunhan Xu** – Supporting contributor; provided guidance in recording and data collection, while also offering feedback from a product and operations perspective to improve usability and practical application.
 - **Yuelang Sun** – Supporting contributor; provided professional support in audio recording and data collection, ensuring high-quality data for model training and evaluation.
 - **Yihuang Liang** - Supporting contributor; worked on systematic model evaluation and project promotion, helping IndexTTS expand its reach and engagement.

### Technical Guidance
 - **Huyang Sun** - Provided strong support for the IndexTTS project, ensuring strategic alignment and resource backing.
 - **Bin Xia** - Contributed to the review, optimization, and follow-up of technical solutions, focusing on ensuring model effectiveness.


## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.


IndexTTS2:

```
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```


IndexTTS:

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```
