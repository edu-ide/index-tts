#!/usr/bin/env python3
"""
Copy only 500K subset audio files from HDD to SSD.

Reads manifest to extract unique audio paths, then copies from HDD to SSD.
Estimated: 500K files, ~25GB, ~10-15 minutes.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Set, Tuple


def extract_audio_paths(manifest_path: str) -> Set[str]:
    """Extract unique audio paths from manifest."""
    audio_paths = set()

    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Get target audio path (relative path like KO-B000011/xxx.mp3)
                audio_path = sample.get("target_audio_path")
                if audio_path:
                    audio_paths.add(audio_path)

    return audio_paths


def copy_file(args: Tuple[str, Path, Path]) -> Tuple[bool, str]:
    """Copy a single file from HDD to SSD."""
    rel_path, hdd_dir, ssd_dir = args

    src = hdd_dir / rel_path
    dst = ssd_dir / rel_path

    if not src.exists():
        return False, f"not_found:{rel_path}"

    if dst.exists():
        return True, "exists"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True, "copied"
    except Exception as e:
        return False, f"error:{e}"


def main():
    parser = argparse.ArgumentParser(description="Copy 500K subset audio files to SSD")
    parser.add_argument("--manifest", required=True, help="Input manifest (jsonl)")
    parser.add_argument("--hdd-dir", required=True, help="HDD base directory")
    parser.add_argument("--ssd-dir", required=True, help="SSD base directory")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually copy")
    args = parser.parse_args()

    hdd_dir = Path(args.hdd_dir)
    ssd_dir = Path(args.ssd_dir)

    print(f"[Info] Manifest: {args.manifest}")
    print(f"[Info] HDD dir: {hdd_dir}")
    print(f"[Info] SSD dir: {ssd_dir}")
    print(f"[Info] Workers: {args.num_workers}")
    print(f"[Info] Dry run: {args.dry_run}")
    print()

    # Extract unique audio paths
    print("[1/3] Extracting audio paths from manifest...")
    audio_paths = extract_audio_paths(args.manifest)
    print(f"      Found {len(audio_paths):,} unique audio files")

    # Estimate size
    print()
    print("[2/3] Estimating total size...")
    total_size = 0
    sample_count = min(1000, len(audio_paths))
    sample_paths = list(audio_paths)[:sample_count]

    for rel_path in sample_paths:
        src = hdd_dir / rel_path
        if src.exists():
            total_size += src.stat().st_size

    avg_size = total_size / sample_count if sample_count > 0 else 0
    estimated_total = avg_size * len(audio_paths)
    print(f"      Sample avg size: {avg_size/1024:.1f} KB")
    print(f"      Estimated total: {estimated_total/1024/1024/1024:.1f} GB")

    if args.dry_run:
        print()
        print("[Dry run] Would copy files. Exiting.")
        return

    # Copy files
    print()
    print("[3/3] Copying files...")

    stats = {"copied": 0, "exists": 0, "not_found": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(copy_file, (path, hdd_dir, ssd_dir)): path
            for path in audio_paths
        }

        with tqdm(total=len(audio_paths), desc="Copying", unit="files") as pbar:
            for future in as_completed(futures):
                success, status = future.result()

                if status == "copied":
                    stats["copied"] += 1
                elif status == "exists":
                    stats["exists"] += 1
                elif status.startswith("not_found"):
                    stats["not_found"] += 1
                else:
                    stats["error"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    copied=stats["copied"],
                    exists=stats["exists"],
                    miss=stats["not_found"]
                )

    print()
    print("[Done]")
    print(f"  Copied: {stats['copied']:,}")
    print(f"  Already exists: {stats['exists']:,}")
    print(f"  Not found: {stats['not_found']:,}")
    print(f"  Errors: {stats['error']:,}")


if __name__ == "__main__":
    main()
