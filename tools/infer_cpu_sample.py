#!/usr/bin/env python3
import argparse
import os
import sys
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
    # vocoder_ckpt is not used by IndexTTS2; it loads from config/model_dir.
    args = parser.parse_args()

    print(f"[InferCPU] Loading model from {args.ckpt} on CPU...")
    
    # Force CPU
    device = torch.device("cpu")
    
    # Initialize TTS
    try:
        tts = IndexTTS2(
            device=device,
            gpt_ckpt_override=args.ckpt,
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
        sf.write(args.output, audio, sr)
        print(f"[InferCPU] Saved sample to {args.output}")

    except Exception as e:
        print(f"[InferCPU] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
