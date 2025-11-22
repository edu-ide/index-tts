#!/usr/bin/env python3
"""
Convert a training checkpoint (model-only or full) into an inference-ready checkpoint.

Usage:
  python tools/export_inference_checkpoint.py \
    --ckpt /mnt/sda1/models/index-tts-ko/checkpoints/best_model_step44200.pth \
    --config /mnt/sda1/models/index-tts-ko/checkpoints/config.yaml \
    --tokenizer /mnt/sda1/models/index-tts-ko/checkpoints/bpe.model \
    --base-tokenizer /mnt/sda1/models/IndexTTS-2/bpe.model \
    --out /mnt/sda1/models/index-tts-ko/checkpoints/gpt_infer_44200.pth

The script builds the model on CPU by default, loads weights from --ckpt (handling
_orig_mod. prefixes), instantiates inference_model, and saves an inference-ready
state dict under the "model" key.
"""

import argparse
from pathlib import Path

import torch

# Reuse build helpers from the trainer
from trainers.train_gpt_v2 import build_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export training checkpoint to inference format")
    p.add_argument("--ckpt", type=Path, required=True, help="Training checkpoint (.pth)")
    p.add_argument("--config", type=Path, required=True, help="Config YAML used for the model")
    p.add_argument("--tokenizer", type=Path, required=True, help="Target tokenizer (BPE model)")
    p.add_argument(
        "--base-tokenizer",
        type=Path,
        default=None,
        help="Base tokenizer for vocab remap (optional, e.g., original 12k BPE)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output inference checkpoint path")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (defaults to cpu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    if args.base_tokenizer and not args.base_tokenizer.exists():
        raise FileNotFoundError(f"Base tokenizer not found: {args.base_tokenizer}")

    print(f"[Export] Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"[Export] Building model from config: {args.config}")
    model = build_model(
        cfg_path=args.config,
        tokenizer=tokenizer,
        base_checkpoint=args.ckpt,
        base_tokenizer_path=args.base_tokenizer,
        device=device,
        enable_grl=False,
        load_ckpt_to_device=device.type != "cpu",
    )
    model.eval()

    # Save inference-ready state dict (includes inference_model.*)
    state = {"model": model.state_dict()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(args.out)
    print(f"[Export] Saved inference checkpoint to: {args.out}")


if __name__ == "__main__":
    main()
