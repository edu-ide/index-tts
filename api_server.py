#!/usr/bin/env python3
"""
IndexTTS API Server
Provides REST API for text-to-speech inference
- Single TTS generation
- Streaming TTS generation
- Batch TTS processing
- Model management (checkpoint/tokenizer switching)
"""

import os
import sys
import argparse
import base64
import tempfile
import uuid
import asyncio
import io
import wave
import struct
import glob as glob_module
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path to import indextts
sys.path.insert(0, str(Path(__file__).parent))

from indextts.infer_v2 import IndexTTS2

app = FastAPI(
    title="IndexTTS API",
    description="Text-to-Speech API using IndexTTS2 with streaming and batch support",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global TTS model instance
tts_model: Optional[IndexTTS2] = None
model_dir: str = None
cfg_path: str = None
current_gpt_checkpoint: str = None
current_bpe_tokenizer: str = None
model_lock = Lock()

# VRAM management
vram_auto_unload: bool = True  # Unload immediately after inference

# Batch job storage
batch_jobs: Dict[str, Dict[str, Any]] = {}
batch_executor = ThreadPoolExecutor(max_workers=4)


class TTSRequest(BaseModel):
    text: str
    prompt_audio_path: Optional[str] = None  # Path to reference audio
    prompt_audio_base64: Optional[str] = None  # Base64 encoded audio
    emo_weight: float = 1.0
    emo_mode: int = 0  # 0: from speaker, 1: from reference, 2: custom vectors
    emo_vector: Optional[List[float]] = None  # 8-element emotion vector
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    max_mel_tokens: int = 1300  # Fixed at 1300
    max_text_tokens_per_segment: int = 120


def estimate_max_mel_tokens(text: str, chars_per_second: float = 4.0, mel_tokens_per_second: float = 43.0, min_tokens: int = 100, max_tokens: int = 1300) -> int:
    """
    Estimate max_mel_tokens based on text length.

    Args:
        text: Input text
        chars_per_second: Estimated characters spoken per second (Korean ~4, English ~15)
        mel_tokens_per_second: Mel tokens generated per second (~43 at 22050Hz)
        min_tokens: Minimum tokens (for very short text)
        max_tokens: Maximum tokens cap

    Returns:
        Estimated max_mel_tokens with 50% buffer for pauses/prosody
    """
    # Count characters (Korean characters count more than English)
    char_count = len(text)

    # Estimate duration in seconds (with 50% buffer)
    estimated_duration = (char_count / chars_per_second) * 1.5

    # Convert to mel tokens
    estimated_tokens = int(estimated_duration * mel_tokens_per_second)

    # Clamp to range
    return max(min_tokens, min(estimated_tokens, max_tokens))


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int = 22050
    duration: float
    inference_time: float


# Batch processing models
class BatchTTSItem(BaseModel):
    text: str
    prompt_audio_path: Optional[str] = None
    prompt_audio_base64: Optional[str] = None
    emo_weight: float = 1.0
    emo_mode: int = 0
    emo_vector: Optional[List[float]] = None


class BatchTTSRequest(BaseModel):
    items: List[BatchTTSItem]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    max_mel_tokens: int = 1300


class BatchItemResult(BaseModel):
    index: int
    status: str  # pending, processing, completed, failed
    audio_base64: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None


class BatchTTSResponse(BaseModel):
    batch_id: str
    status: str  # queued, processing, completed, failed
    total_items: int
    completed_items: int = 0
    results: List[BatchItemResult] = []


# Model management models
class ModelStatus(BaseModel):
    gpt_checkpoint: str
    bpe_tokenizer: str
    device: str
    fp16: bool
    model_dir: str


class ModelLoadRequest(BaseModel):
    gpt_checkpoint: Optional[str] = None
    bpe_tokenizer: Optional[str] = None


class CheckpointList(BaseModel):
    checkpoints: List[str]


class TokenizerList(BaseModel):
    tokenizers: List[str]


def load_tts_model():
    """Load TTS model to GPU"""
    global tts_model, current_gpt_checkpoint, current_bpe_tokenizer

    if tts_model is not None:
        return  # Already loaded

    print(f"Loading IndexTTS2 model from: {model_dir}")
    tts_model = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device='cuda',  # Uses CUDA_VISIBLE_DEVICES from start_api.sh
        use_deepspeed=False,
        use_fp16=False
    )

    # Track current checkpoint/tokenizer
    current_gpt_checkpoint = os.path.basename(tts_model.gpt_path)
    current_bpe_tokenizer = os.path.basename(tts_model.bpe_path)
    print(f"Model loaded successfully!")
    print(f"GPT checkpoint: {current_gpt_checkpoint}")
    print(f"BPE tokenizer: {current_bpe_tokenizer}")


def unload_tts_model():
    """Unload TTS model from GPU to free VRAM"""
    global tts_model
    import torch
    import gc

    if tts_model is None:
        return

    print("Unloading model from GPU to free VRAM...")

    # Move all submodules to CPU first, then delete
    try:
        if hasattr(tts_model, 'gpt') and tts_model.gpt is not None:
            tts_model.gpt.cpu()
            del tts_model.gpt
        if hasattr(tts_model, 'bigvgan') and tts_model.bigvgan is not None:
            tts_model.bigvgan.cpu()
            del tts_model.bigvgan
        if hasattr(tts_model, 'dvae') and tts_model.dvae is not None:
            tts_model.dvae.cpu()
            del tts_model.dvae
    except Exception as e:
        print(f"Warning during submodule cleanup: {e}")

    del tts_model
    tts_model = None

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Log actual VRAM usage after cleanup
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"Model unloaded. VRAM: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")


def ensure_model_loaded():
    """Ensure model is loaded before inference"""
    if tts_model is None:
        load_tts_model()


def auto_unload_if_enabled():
    """Unload model after inference if auto_unload is enabled"""
    if vram_auto_unload:
        unload_tts_model()


@app.on_event("startup")
async def startup_event():
    """Initialize TTS model on startup"""
    global model_dir, cfg_path

    if model_dir is None:
        model_dir = os.path.expanduser("~/models/index-tts-ko/checkpoints")

    if cfg_path is None:
        cfg_path = os.path.join(model_dir, "config.yaml")

    print(f"Model directory: {model_dir}")
    print(f"Config path: {cfg_path}")
    print(f"VRAM auto-unload: {vram_auto_unload}")
    print("Model will be loaded on first request.")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "model": "IndexTTS2", "version": "2.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "gpt_checkpoint": current_gpt_checkpoint,
        "bpe_tokenizer": current_bpe_tokenizer,
        "vram_auto_unload": vram_auto_unload
    }


# =============================================================================
# VRAM Management Endpoints
# =============================================================================

@app.post("/vram/load")
async def vram_load():
    """Manually load model to GPU"""
    with model_lock:
        load_tts_model()
    return {"status": "loaded", "model_loaded": tts_model is not None}


@app.post("/vram/unload")
async def vram_unload():
    """Manually unload model from GPU"""
    with model_lock:
        unload_tts_model()
    return {"status": "unloaded", "model_loaded": tts_model is not None}


@app.post("/vram/auto_unload")
async def vram_auto_unload_config(enabled: bool = True):
    """Enable/disable auto-unload after inference"""
    global vram_auto_unload
    vram_auto_unload = enabled
    return {"vram_auto_unload": vram_auto_unload}


@app.get("/vram/status")
async def vram_status():
    """Get VRAM status"""
    import torch

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
        }

    return {
        "model_loaded": tts_model is not None,
        "vram_auto_unload": vram_auto_unload,
        "gpu": gpu_info
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status"""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelStatus(
        gpt_checkpoint=current_gpt_checkpoint,
        bpe_tokenizer=current_bpe_tokenizer,
        device=str(tts_model.device),
        fp16=tts_model.use_fp16,
        model_dir=model_dir
    )


@app.get("/model/checkpoints", response_model=CheckpointList)
async def list_checkpoints():
    """List available GPT checkpoints"""
    if model_dir is None:
        raise HTTPException(status_code=503, detail="Model directory not set")

    pattern = os.path.join(model_dir, "gpt*.pth")
    files = glob_module.glob(pattern)
    checkpoints = [os.path.basename(f) for f in sorted(files)]
    return CheckpointList(checkpoints=checkpoints)


@app.get("/model/tokenizers", response_model=TokenizerList)
async def list_tokenizers():
    """List available BPE tokenizers"""
    if model_dir is None:
        raise HTTPException(status_code=503, detail="Model directory not set")

    pattern = os.path.join(model_dir, "bpe*.model")
    files = glob_module.glob(pattern)
    tokenizers = [os.path.basename(f) for f in sorted(files)]
    return TokenizerList(tokenizers=tokenizers)


@app.post("/model/load", response_model=ModelStatus)
async def load_model(request: ModelLoadRequest):
    """Load a different GPT checkpoint or BPE tokenizer"""
    global tts_model, current_gpt_checkpoint, current_bpe_tokenizer

    if model_dir is None:
        raise HTTPException(status_code=503, detail="Model directory not set")

    gpt_ckpt = request.gpt_checkpoint
    bpe_tok = request.bpe_tokenizer

    # Validate files exist
    if gpt_ckpt:
        gpt_path = os.path.join(model_dir, gpt_ckpt)
        if not os.path.exists(gpt_path):
            raise HTTPException(status_code=400, detail=f"GPT checkpoint not found: {gpt_ckpt}")

    if bpe_tok:
        bpe_path = os.path.join(model_dir, bpe_tok)
        if not os.path.exists(bpe_path):
            raise HTTPException(status_code=400, detail=f"BPE tokenizer not found: {bpe_tok}")

    try:
        with model_lock:
            # Reload model with new checkpoint
            import torch
            if tts_model is not None:
                del tts_model
                torch.cuda.empty_cache()

            gpt_override = os.path.join(model_dir, gpt_ckpt) if gpt_ckpt else None

            tts_model = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                device='cuda',  # Uses CUDA_VISIBLE_DEVICES from start_api.sh
                use_deepspeed=False,
                use_fp16=False,
                gpt_ckpt_override=gpt_override
            )

            # Update BPE tokenizer if specified
            if bpe_tok:
                from indextts.utils.front import TextTokenizer
                tts_model.bpe_path = os.path.join(model_dir, bpe_tok)
                tts_model.tokenizer = TextTokenizer(tts_model.bpe_path, tts_model.normalizer)

            current_gpt_checkpoint = os.path.basename(tts_model.gpt_path)
            current_bpe_tokenizer = os.path.basename(tts_model.bpe_path)

        return ModelStatus(
            gpt_checkpoint=current_gpt_checkpoint,
            bpe_tokenizer=current_bpe_tokenizer,
            device=str(tts_model.device),
            fp16=tts_model.use_fp16,
            model_dir=model_dir
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text

    Parameters:
    - text: Text to synthesize
    - prompt_audio_path: Path to reference audio file (optional)
    - prompt_audio_base64: Base64 encoded reference audio (optional)
    - emo_weight: Emotion weight (0.0-1.0)
    - emo_mode: 0=from speaker, 1=from reference, 2=custom vectors
    - temperature: Sampling temperature
    - top_p: Top-p sampling
    - top_k: Top-k sampling
    - repetition_penalty: Repetition penalty
    - length_penalty: Length penalty
    - max_mel_tokens: Maximum mel tokens

    Returns:
    - audio_base64: Base64 encoded WAV audio
    - sample_rate: Sample rate (22050 Hz)
    - duration: Audio duration in seconds
    - inference_time: Inference time in seconds
    """
    # Ensure model is loaded (auto-load if offloaded)
    with model_lock:
        ensure_model_loaded()

    # Handle reference audio
    prompt_audio = None
    temp_audio_file = None

    try:
        if request.prompt_audio_base64:
            # Decode base64 audio
            print(f"Received prompt_audio_base64 (len: {len(request.prompt_audio_base64)})")
            audio_bytes = base64.b64decode(request.prompt_audio_base64)
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()
            prompt_audio = temp_audio_file.name
        elif request.prompt_audio_path:
            print(f"Received prompt_audio_path: {request.prompt_audio_path}")
            if not os.path.exists(request.prompt_audio_path):
                print(f"Error: Reference audio not found at {request.prompt_audio_path}")
                raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.prompt_audio_path}")
            prompt_audio = request.prompt_audio_path
        else:
            print("Warning: No reference audio provided. TTS will use default/random speaker embedding.")

        # Generate speech
        import time
        start_time = time.time()

        output_path = tempfile.mktemp(suffix=".wav")

        # Auto-estimate max_mel_tokens if not specified (0 = auto)
        effective_max_mel_tokens = request.max_mel_tokens
        if effective_max_mel_tokens <= 0:
            effective_max_mel_tokens = estimate_max_mel_tokens(request.text)
            print(f"Auto max_mel_tokens: {effective_max_mel_tokens} (text: '{request.text[:30]}...' len={len(request.text)})")

        # Call TTS inference
        tts_model.infer(
            spk_audio_prompt=prompt_audio,
            text=request.text,
            emo_alpha=request.emo_weight,
            emo_audio_prompt=None,  # Use prompt audio for emotion
            emo_vector=request.emo_vector,  # 8-element emotion vector
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else None,
            repetition_penalty=request.repetition_penalty,
            length_penalty=request.length_penalty,
            max_mel_tokens=effective_max_mel_tokens,
            output_path=output_path,
            verbose=False,
            num_beams=1  # Reduce VRAM usage (default: 3)
        )

        inference_time = time.time() - start_time

        # Read generated audio
        with open(output_path, 'rb') as f:
            audio_data = f.read()

        # Calculate duration (approximate)
        import wave
        with wave.open(output_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Cleanup
        os.remove(output_path)

        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=22050,
            duration=duration,
            inference_time=inference_time
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

    finally:
        # Cleanup temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
        # Auto-unload VRAM after inference
        with model_lock:
            auto_unload_if_enabled()


@app.post("/tts_file")
async def text_to_speech_file(
    text: str,
    prompt_audio: Optional[UploadFile] = File(None),
    emo_weight: float = 1.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20
):
    """
    Generate speech from text and return audio file

    Parameters (multipart/form-data):
    - text: Text to synthesize
    - prompt_audio: Reference audio file (optional)
    - emo_weight: Emotion weight
    - temperature: Sampling temperature
    - top_p: Top-p sampling
    - top_k: Top-k sampling

    Returns:
    - WAV audio file
    """
    # Ensure model is loaded (auto-load if offloaded)
    with model_lock:
        ensure_model_loaded()

    temp_audio_file = None
    prompt_audio_path = None

    try:
        # Save uploaded audio if provided
        if prompt_audio:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(await prompt_audio.read())
            temp_audio_file.close()
            prompt_audio_path = temp_audio_file.name

        # Generate speech
        output_path = tempfile.mktemp(suffix=".wav")

        tts_model.infer(
            spk_audio_prompt=prompt_audio_path,
            text=text,
            emo_weight=emo_weight,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            output_path=output_path,
            verbose=False,
            num_beams=1  # Reduce VRAM usage
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS inference failed: {str(e)}")

    finally:
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
        # Auto-unload VRAM after inference
        with model_lock:
            auto_unload_if_enabled()


# =============================================================================
# Streaming TTS Endpoint
# =============================================================================

def create_wav_header(sample_rate: int = 22050, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """Create a WAV header for streaming (with placeholder size)"""
    # Use max size placeholder - will be updated or client handles chunked transfer
    data_size = 0x7FFFFFFF
    file_size = data_size + 36

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # PCM format
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,  # byte rate
        channels * bits_per_sample // 8,  # block align
        bits_per_sample,
        b'data',
        data_size
    )
    return header


@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    Generate speech from text with streaming response.

    Returns chunked WAV audio as it's generated.
    Each segment is streamed as soon as it's ready.
    """
    # Ensure model is loaded (auto-load if offloaded)
    with model_lock:
        ensure_model_loaded()

    # Handle reference audio
    prompt_audio = None
    temp_audio_file = None

    try:
        if request.prompt_audio_base64:
            audio_bytes = base64.b64decode(request.prompt_audio_base64)
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()
            prompt_audio = temp_audio_file.name
        elif request.prompt_audio_path:
            if not os.path.exists(request.prompt_audio_path):
                raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.prompt_audio_path}")
            prompt_audio = request.prompt_audio_path

        # Auto-estimate max_mel_tokens for streaming
        effective_max_mel_tokens = request.max_mel_tokens
        if effective_max_mel_tokens <= 0:
            effective_max_mel_tokens = estimate_max_mel_tokens(request.text)

        def audio_generator():
            import torch
            try:
                # Send WAV header first
                yield create_wav_header(sample_rate=22050)

                # Use streaming inference
                generator = tts_model.infer(
                    spk_audio_prompt=prompt_audio,
                    text=request.text,
                    output_path=None,
                    emo_alpha=request.emo_weight,
                    emo_vector=request.emo_vector,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k if request.top_k > 0 else None,
                    repetition_penalty=request.repetition_penalty,
                    length_penalty=request.length_penalty,
                    max_mel_tokens=effective_max_mel_tokens,
                    max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                    stream_return=True,
                    verbose=False,
                    num_beams=1  # Reduce VRAM usage
                )

                for wav_chunk in generator:
                    if wav_chunk is not None and isinstance(wav_chunk, torch.Tensor):
                        # Convert to int16 bytes
                        wav_int16 = wav_chunk.type(torch.int16).numpy()
                        yield wav_int16.tobytes()

            finally:
                # Cleanup temp file
                if temp_audio_file and os.path.exists(temp_audio_file.name):
                    os.remove(temp_audio_file.name)
                # Auto-unload VRAM after streaming completes
                with model_lock:
                    auto_unload_if_enabled()

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=tts_stream.wav",
                "Transfer-Encoding": "chunked"
            }
        )

    except Exception as e:
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
        raise HTTPException(status_code=500, detail=f"TTS streaming failed: {str(e)}")


# =============================================================================
# Batch TTS Endpoints
# =============================================================================

def process_batch_item(
    item: BatchTTSItem,
    index: int,
    batch_id: str,
    common_params: dict
) -> BatchItemResult:
    """Process a single batch item"""
    import time
    import torch

    temp_audio_file = None
    prompt_audio = None

    try:
        # Update status to processing
        batch_jobs[batch_id]["results"][index].status = "processing"

        # Handle reference audio
        if item.prompt_audio_base64:
            audio_bytes = base64.b64decode(item.prompt_audio_base64)
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()
            prompt_audio = temp_audio_file.name
        elif item.prompt_audio_path:
            if not os.path.exists(item.prompt_audio_path):
                return BatchItemResult(
                    index=index,
                    status="failed",
                    error=f"Reference audio not found: {item.prompt_audio_path}"
                )
            prompt_audio = item.prompt_audio_path

        # Generate speech
        output_path = tempfile.mktemp(suffix=".wav")

        # Auto-estimate max_mel_tokens for each batch item
        item_max_mel_tokens = common_params["max_mel_tokens"]
        if item_max_mel_tokens <= 0:
            item_max_mel_tokens = estimate_max_mel_tokens(item.text)

        with model_lock:
            tts_model.infer(
                spk_audio_prompt=prompt_audio,
                text=item.text,
                output_path=output_path,
                emo_alpha=item.emo_weight,
                emo_vector=item.emo_vector,
                temperature=common_params["temperature"],
                top_p=common_params["top_p"],
                top_k=common_params["top_k"] if common_params["top_k"] > 0 else None,
                repetition_penalty=common_params["repetition_penalty"],
                length_penalty=common_params["length_penalty"],
                max_mel_tokens=item_max_mel_tokens,
                verbose=False,
                num_beams=1  # Reduce VRAM usage
            )

        # Read and encode audio
        with open(output_path, 'rb') as f:
            audio_data = f.read()

        # Get duration
        with wave.open(output_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

        # Cleanup
        os.remove(output_path)

        return BatchItemResult(
            index=index,
            status="completed",
            audio_base64=audio_base64,
            duration=duration
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return BatchItemResult(
            index=index,
            status="failed",
            error=str(e)
        )

    finally:
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)


def run_batch_job(batch_id: str, request: BatchTTSRequest):
    """Run batch job in background"""
    # Ensure model is loaded before starting batch
    with model_lock:
        ensure_model_loaded()

    batch_jobs[batch_id]["status"] = "processing"

    common_params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "repetition_penalty": request.repetition_penalty,
        "length_penalty": request.length_penalty,
        "max_mel_tokens": request.max_mel_tokens
    }

    for idx, item in enumerate(request.items):
        result = process_batch_item(item, idx, batch_id, common_params)
        batch_jobs[batch_id]["results"][idx] = result

        if result.status == "completed":
            batch_jobs[batch_id]["completed_items"] += 1

    # Update final status
    all_completed = all(r.status == "completed" for r in batch_jobs[batch_id]["results"])
    batch_jobs[batch_id]["status"] = "completed" if all_completed else "completed_with_errors"

    # Auto-unload VRAM after batch completes
    with model_lock:
        auto_unload_if_enabled()


@app.post("/tts/batch", response_model=BatchTTSResponse)
async def batch_text_to_speech(request: BatchTTSRequest, background_tasks: BackgroundTasks):
    """
    Submit a batch TTS job.

    Parameters:
    - items: List of TTS items to process
    - temperature, top_p, etc.: Common generation parameters

    Returns:
    - batch_id: ID to track the job
    - status: Current job status
    """

    if not request.items:
        raise HTTPException(status_code=400, detail="No items provided")

    if len(request.items) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 items per batch")

    # Create batch job
    batch_id = str(uuid.uuid4())
    batch_jobs[batch_id] = {
        "status": "queued",
        "total_items": len(request.items),
        "completed_items": 0,
        "results": [
            BatchItemResult(index=i, status="pending")
            for i in range(len(request.items))
        ]
    }

    # Start background processing
    background_tasks.add_task(run_batch_job, batch_id, request)

    return BatchTTSResponse(
        batch_id=batch_id,
        status="queued",
        total_items=len(request.items),
        completed_items=0,
        results=batch_jobs[batch_id]["results"]
    )


@app.get("/tts/batch/{batch_id}", response_model=BatchTTSResponse)
async def get_batch_status(batch_id: str):
    """Get the status of a batch job"""
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")

    job = batch_jobs[batch_id]
    return BatchTTSResponse(
        batch_id=batch_id,
        status=job["status"],
        total_items=job["total_items"],
        completed_items=job["completed_items"],
        results=job["results"]
    )


@app.delete("/tts/batch/{batch_id}")
async def delete_batch_job(batch_id: str):
    """Delete a completed batch job to free memory"""
    if batch_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")

    del batch_jobs[batch_id]
    return {"status": "deleted", "batch_id": batch_id}


@app.get("/tts/batch")
async def list_batch_jobs():
    """List all batch jobs"""
    return {
        "jobs": [
            {
                "batch_id": bid,
                "status": job["status"],
                "total_items": job["total_items"],
                "completed_items": job["completed_items"]
            }
            for bid, job in batch_jobs.items()
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="~/models/index-tts-ko/checkpoints",
                        help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload")

    args = parser.parse_args()

    model_dir = os.path.expanduser(args.model_dir)

    print(f"Starting IndexTTS API Server")
    print(f"Model directory: {model_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
