#!/usr/bin/env python3
"""
Fast mel manifest generator - uses file size instead of loading numpy arrays.

For numpy .npy files with float32 data and shape (mel_len, 80):
- Header size: 128 bytes (fixed for simple arrays)
- Data size: mel_len * 80 * 4 bytes
- mel_len = (file_size - 128) / 320
"""

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

# Constants
NPY_HEADER_SIZE = 128
MEL_BINS = 80
FLOAT32_SIZE = 4
BYTES_PER_FRAME = MEL_BINS * FLOAT32_SIZE  # 320 bytes


def get_mel_len_from_size(file_size: int) -> int:
    """Calculate mel length from file size."""
    return (file_size - NPY_HEADER_SIZE) // BYTES_PER_FRAME


def process_sample(sample: Dict[str, Any], data_dir: Path) -> Tuple[Optional[Dict], str]:
    """Add mel path and length to sample using file size."""
    prompt_id = sample.get("prompt_id")
    if not prompt_id:
        return None, "no_prompt_id"

    mel_rel_path = f"mel/{prompt_id}.npy"
    mel_full_path = data_dir / mel_rel_path

    if not mel_full_path.exists():
        return None, "mel_not_found"

    try:
        file_size = mel_full_path.stat().st_size
        mel_len = get_mel_len_from_size(file_size)

        if mel_len <= 0:
            return None, "invalid_mel_len"

        sample["prompt_mel_path"] = mel_rel_path
        sample["prompt_mel_len"] = mel_len
        return sample, "ok"
    except Exception as e:
        return None, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Fast mel manifest generator")
    parser.add_argument("--input-manifest", required=True, help="Input manifest (jsonl)")
    parser.add_argument("--output-manifest", required=True, help="Output manifest (jsonl)")
    parser.add_argument("--data-dir", required=True, help="Directory containing mel/ folder")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of workers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mel_dir = data_dir / "mel"

    print(f"[Info] Input: {args.input_manifest}")
    print(f"[Info] Output: {args.output_manifest}")
    print(f"[Info] Data dir: {data_dir}")
    print(f"[Info] Mel dir: {mel_dir}")
    print(f"[Info] Workers: {args.num_workers}")

    # Check mel directory
    if not mel_dir.exists():
        print(f"[Error] Mel directory not found: {mel_dir}")
        return

    # Load input manifest
    print(f"[Info] Loading manifest...")
    samples = []
    with open(args.input_manifest, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"[Info] Total samples: {len(samples):,}")

    # Process samples in parallel
    stats = {"ok": 0, "mel_not_found": 0, "error": 0}
    results = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_sample, sample, data_dir): i
            for i, sample in enumerate(samples)
        }

        with tqdm(total=len(samples), desc="Processing", unit="samples") as pbar:
            for future in as_completed(futures):
                result, status = future.result()
                if status == "ok":
                    stats["ok"] += 1
                    results.append((futures[future], result))
                elif status == "mel_not_found":
                    stats["mel_not_found"] += 1
                else:
                    stats["error"] += 1

                pbar.update(1)
                pbar.set_postfix(ok=stats["ok"], miss=stats["mel_not_found"], err=stats["error"])

    # Sort results by original order
    results.sort(key=lambda x: x[0])

    # Write output manifest
    print(f"[Info] Writing output manifest...")
    with open(args.output_manifest, "w") as f:
        for _, sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[Done]")
    print(f"  OK: {stats['ok']:,}")
    print(f"  Mel not found: {stats['mel_not_found']:,}")
    print(f"  Errors: {stats['error']:,}")
    print(f"  Output: {args.output_manifest}")


if __name__ == "__main__":
    main()
