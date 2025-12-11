#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import torch
import soundfile as sf

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from indextts.infer_v2 import IndexTTS2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to GPT checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("--output", type=str, required=True, help="Output wav path")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to <ckpt dir>/config.yaml)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing checkpoints and assets (defaults to ckpt parent)",
    )
    # vocoder_ckpt is not used by IndexTTS2; it loads from config/model_dir.
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else ckpt_path.parent
    cfg_path = Path(args.config).expanduser().resolve() if args.config else model_dir / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}. Pass --config or ensure it sits next to the checkpoint.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[InferCPU] Loading model from {ckpt_path} on CPU...")
    print(f"[InferCPU] Using config: {cfg_path}")
    print(f"[InferCPU] Model directory: {model_dir}")
    
    # Force CPU
    device = torch.device("cpu")
    
    # Initialize TTS
    try:
        tts = IndexTTS2(
            device=device,
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            gpt_ckpt_override=str(ckpt_path),
        )
        
        print(f"[InferCPU] Synthesizing: {args.text}")
        
        if not os.path.exists(args.ref_audio):
            print(f"[InferCPU] Warning: Reference audio {args.ref_audio} not found.")
            # Create a dummy file to prevent crash if file missing, or just exit
            # For now, let's try to proceed, maybe infer handles missing ref? No, it needs it.
            # We will assume the user provides a valid path in the training script.
        
        # Run inference
        sr, audio = tts.infer(
            spk_audio_prompt=args.ref_audio,
            text=args.text,
            output_path=None,
            use_emo_text=False,
            use_random=False,
            verbose=True
        )
        
        # Save
        sf.write(str(output_path), audio, sr)
        print(f"[InferCPU] Saved sample to {output_path}")

    except Exception as e:
        print(f"[InferCPU] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
