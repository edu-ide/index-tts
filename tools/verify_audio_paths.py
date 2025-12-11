#!/usr/bin/env python3
"""
Verify all audio paths in manifest are accessible.
Run this before Stage 2 training to ensure all files exist.

Usage:
    python tools/verify_audio_paths.py \
        --manifest /mnt/sda1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl \
        --limit 10000
"""

import argparse
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def check_file(args):
    """Check if file exists and is readable."""
    base_dir, rel_path = args
    if not rel_path:
        return None, "empty_path"

    full_path = base_dir / rel_path
    if not full_path.exists():
        return str(full_path), "not_found"

    try:
        # Try to open and read first byte
        with open(full_path, 'rb') as f:
            f.read(1)
        return None, "ok"
    except Exception as e:
        return str(full_path), f"read_error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Verify audio paths in manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to check")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent

    print(f"[Info] Manifest: {manifest_path}")
    print(f"[Info] Base dir: {base_dir}")

    # Load manifest
    samples = []
    with open(manifest_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                audio_path = record.get("prompt_audio_path")
                if audio_path:
                    samples.append((base_dir, audio_path))

    if args.limit:
        samples = samples[:args.limit]

    print(f"[Info] Checking {len(samples):,} audio paths...")

    # Check files in parallel
    stats = {"ok": 0, "not_found": 0, "read_error": 0, "empty_path": 0}
    missing_files = []

    with Pool(args.workers) as pool:
        for result, status in tqdm(pool.imap_unordered(check_file, samples, chunksize=1000),
                                   total=len(samples), desc="Checking"):
            if status == "ok":
                stats["ok"] += 1
            elif status == "not_found":
                stats["not_found"] += 1
                missing_files.append(result)
            elif status == "empty_path":
                stats["empty_path"] += 1
            else:
                stats["read_error"] += 1
                missing_files.append(f"{result} ({status})")

    print()
    print("=" * 60)
    print("Results:")
    print(f"  ✅ OK: {stats['ok']:,}")
    print(f"  ❌ Not found: {stats['not_found']:,}")
    print(f"  ❌ Read error: {stats['read_error']:,}")
    print(f"  ⚠️  Empty path: {stats['empty_path']:,}")
    print("=" * 60)

    if missing_files:
        print(f"\nFirst 20 missing/error files:")
        for f in missing_files[:20]:
            print(f"  {f}")

    if stats["not_found"] > 0 or stats["read_error"] > 0:
        print(f"\n❌ FAILED: {stats['not_found'] + stats['read_error']} files inaccessible")
        return 1
    else:
        print(f"\n✅ All {stats['ok']:,} audio files verified successfully")
        return 0


if __name__ == "__main__":
    exit(main())
