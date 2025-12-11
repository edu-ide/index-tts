#!/usr/bin/env python3
"""
Extract inference-only model weights from training checkpoint.
Removes optimizer, scheduler, and other training-only states.
"""

import torch
import sys
from pathlib import Path

def extract_inference_weights(input_path, output_path):
    """Extract model weights from training checkpoint"""

    print(f"Loading checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Display checkpoint structure
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            print(f"  {key}: dict with {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {type(checkpoint[key])}")

    # Extract only model weights
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"\nExtracting 'model' state dict with {len(model_state)} keys")
    else:
        # If no 'model' key, assume the entire checkpoint is the model
        model_state = checkpoint
        print(f"\nUsing entire checkpoint as model state dict with {len(model_state)} keys")

    # Strip _orig_mod. prefix if present (from torch.compile)
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        print("Removing _orig_mod. prefix from keys")
        model_state = {k.replace("_orig_mod.", "", 1): v for k, v in model_state.items()}

    # Calculate sizes
    original_size = Path(input_path).stat().st_size / (1024**3)  # GB

    # Save inference-only checkpoint
    print(f"\nSaving inference model to: {output_path}")
    torch.save(model_state, output_path)

    new_size = Path(output_path).stat().st_size / (1024**3)  # GB

    print(f"\nOriginal size: {original_size:.2f} GB")
    print(f"New size: {new_size:.2f} GB")
    print(f"Reduction: {original_size - new_size:.2f} GB ({(1 - new_size/original_size)*100:.1f}%)")

    return new_size

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_inference_model.py <input_checkpoint> <output_checkpoint>")
        print("Example: python extract_inference_model.py best_model.pth best_model_inference.pth")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    extract_inference_weights(input_path, output_path)
    print("\nDone!")
