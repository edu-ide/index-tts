#!/usr/bin/env python3
"""
Preprocess mel-spectrograms for Stage 2 GRL training.

Features:
- Progress bar with ETA (tqdm)
- Resume support (skip existing mel files)
- Parallel processing with chunked batches

Usage:
    python tools/preprocess_mel_for_stage2.py \
        --input-manifest /path/to/gpt_pairs_train.jsonl \
        --output-manifest /path/to/gpt_pairs_train_mel.jsonl \
        --data-dir /path/to/KO_preprocessed \
        --num-workers 32
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchaudio.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


# Mel-spectrogram parameters from config.yaml (s2mel section)
MEL_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "f_min": 0,
    "f_max": None,  # Nyquist
}

# Global variable for data_dir (for multiprocessing)
_data_dir = None
_skip_existing = True


def init_worker(data_dir: str, skip_existing: bool):
    """Initialize worker with shared state."""
    global _data_dir, _skip_existing
    _data_dir = Path(data_dir)
    _skip_existing = skip_existing
    # Suppress warnings in worker processes
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torchaudio.*")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")


def compute_mel_spectrogram(audio_path: str, target_sr: int = 22050) -> Optional[np.ndarray]:
    """Compute mel-spectrogram from audio file."""
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Compute mel-spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=MEL_CONFIG["n_fft"],
            win_length=MEL_CONFIG["win_length"],
            hop_length=MEL_CONFIG["hop_length"],
            n_mels=MEL_CONFIG["n_mels"],
            f_min=MEL_CONFIG["f_min"],
            f_max=MEL_CONFIG["f_max"],
            power=2.0,
            normalized=False,
        )

        mel = mel_transform(waveform)  # [1, n_mels, time]

        # Convert to log scale
        mel = torch.log(torch.clamp(mel, min=1e-5))

        # Remove batch dim and transpose to [time, n_mels]
        mel = mel.squeeze(0).transpose(0, 1).numpy()

        return mel.astype(np.float32)

    except Exception as e:
        return None


def process_sample(sample: Dict[str, Any]) -> Tuple[Optional[Dict], str]:
    """Process a single sample."""
    global _data_dir, _skip_existing

    # Get audio path
    audio_rel_path = sample.get("prompt_audio_path")
    if not audio_rel_path:
        return None, "no_audio_path"

    audio_path = _data_dir / audio_rel_path
    if not audio_path.exists():
        return None, "audio_not_found"

    # Determine mel output path
    prompt_id = sample.get("prompt_id", sample.get("id", "unknown"))
    mel_filename = f"{prompt_id}.npy"
    mel_rel_path = f"mel/{mel_filename}"
    mel_full_path = _data_dir / mel_rel_path

    # Skip if already exists (resume support)
    if _skip_existing and mel_full_path.exists():
        try:
            mel = np.load(mel_full_path)
            sample["prompt_mel_path"] = mel_rel_path
            sample["prompt_mel_len"] = mel.shape[0]
            return sample, "skipped"
        except:
            pass  # Corrupted file, recompute

    # Compute mel-spectrogram
    mel = compute_mel_spectrogram(str(audio_path))
    if mel is None:
        return None, "mel_failed"

    # Save mel-spectrogram
    mel_full_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_full_path, mel)

    # Update sample with mel path
    sample["prompt_mel_path"] = mel_rel_path
    sample["prompt_mel_len"] = mel.shape[0]

    return sample, "success"


def main():
    parser = argparse.ArgumentParser(description="Preprocess mel-spectrograms for Stage 2")
    parser.add_argument("--input-manifest", type=str, required=True,
                        help="Input manifest file (jsonl)")
    parser.add_argument("--output-manifest", type=str, required=True,
                        help="Output manifest file with mel paths (jsonl)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Base data directory")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume (recompute all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mel_dir = data_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = not args.no_resume

    # Load input manifest
    print(f"[Info] Loading manifest: {args.input_manifest}")
    samples = []
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.limit:
        samples = samples[:args.limit]

    total = len(samples)
    print(f"[Info] Total samples: {total:,}")
    print(f"[Info] Mel output: {mel_dir}")
    print(f"[Info] Workers: {args.num_workers}")
    print(f"[Info] Resume: {'enabled' if skip_existing else 'disabled'}")
    print(f"[Info] Mel config: sr={MEL_CONFIG['sample_rate']}, n_mels={MEL_CONFIG['n_mels']}")
    print()

    # Process samples in parallel with progress bar
    processed = []
    stats = {"success": 0, "skipped": 0, "no_audio_path": 0, "audio_not_found": 0, "mel_failed": 0}

    # Use imap_unordered for streaming results
    with Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(str(data_dir), skip_existing)
    ) as pool:
        with tqdm(total=total, desc="Processing", unit="samples",
                  dynamic_ncols=True, smoothing=0.05) as pbar:

            # Use imap_unordered with chunksize for efficiency
            chunksize = max(1, total // (args.num_workers * 100))
            chunksize = min(chunksize, 1000)  # Cap at 1000

            for result, status in pool.imap_unordered(process_sample, samples, chunksize=chunksize):
                if status in stats:
                    stats[status] += 1

                if result is not None:
                    processed.append(result)

                # Update progress bar
                pbar.set_postfix({
                    "ok": stats["success"],
                    "skip": stats["skipped"],
                    "err": stats["audio_not_found"] + stats["mel_failed"]
                })
                pbar.update(1)

    print()
    print("=" * 60)
    print("📊 Results:")
    print(f"   ✅ New processed: {stats['success']:,}")
    print(f"   ⏭️  Skipped (resume): {stats['skipped']:,}")
    print(f"   ❌ Audio not found: {stats['audio_not_found']:,}")
    print(f"   ❌ Mel failed: {stats['mel_failed']:,}")
    print(f"   📁 Total output: {len(processed):,}")
    print("=" * 60)

    # Save output manifest
    print(f"\n[Info] Saving: {args.output_manifest}")
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        for sample in processed:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print()
    print("✅ Complete!")
    print(f"   Output manifest: {args.output_manifest}")
    print(f"   Mel directory: {mel_dir}")


if __name__ == "__main__":
    main()
