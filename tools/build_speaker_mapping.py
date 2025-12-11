#!/usr/bin/env python3
"""
Build speaker ID mapping for Stage 2 GRL training.

This script creates a mapping from speaker names to integer IDs,
selecting the top N speakers by sample count.

Usage:
    python tools/build_speaker_mapping.py \
        --manifest /path/to/train_manifest.jsonl \
        --output /path/to/speaker_mapping.json \
        --top-k 500
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict


def build_speaker_mapping(
    manifest_path: Path,
    output_path: Path,
    top_k: int = 500,
    min_samples: int = 10
) -> Dict[str, int]:
    """
    Build speaker ID mapping from manifest.

    Args:
        manifest_path: Path to train_manifest.jsonl
        output_path: Path to save speaker_mapping.json
        top_k: Number of top speakers to include (by sample count)
        min_samples: Minimum samples required for a speaker

    Returns:
        Dictionary mapping speaker names to IDs
    """
    print(f"📊 Building speaker mapping from {manifest_path}")
    print(f"   Top-K: {top_k}")
    print(f"   Min samples: {min_samples}")
    print()

    # Count speaker samples
    speaker_counts = Counter()

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            speaker_counts[data['speaker']] += 1

    print(f"✅ Total speakers found: {len(speaker_counts):,}")

    # Filter by min_samples
    filtered_speakers = {
        spk: count for spk, count in speaker_counts.items()
        if count >= min_samples
    }

    print(f"✅ Speakers with >= {min_samples} samples: {len(filtered_speakers):,}")

    # Get top-k speakers
    top_speakers = sorted(
        filtered_speakers.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Create mapping (sorted by count for consistency)
    speaker_to_id = {
        spk: idx for idx, (spk, count) in enumerate(top_speakers)
    }

    # Statistics
    total_samples = sum(count for spk, count in top_speakers)
    all_samples = sum(speaker_counts.values())

    print()
    print(f"📈 Statistics:")
    print(f"   Selected speakers: {len(speaker_to_id):,}")
    print(f"   Total samples (selected): {total_samples:,}")
    print(f"   Total samples (all): {all_samples:,}")
    print(f"   Coverage: {100 * total_samples / all_samples:.2f}%")
    print()

    # Show top 10
    print("🏆 Top 10 speakers:")
    for idx, (spk, count) in enumerate(top_speakers[:10], 1):
        print(f"   {idx:2d}. {spk}: {count:,} samples")
    print()

    # Save mapping
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_to_id, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved speaker mapping to {output_path}")
    print(f"   Total speakers in mapping: {len(speaker_to_id):,}")

    # Also save reverse mapping (for debugging)
    reverse_path = output_path.with_name(output_path.stem + '_reverse.json')
    id_to_speaker = {str(idx): spk for spk, idx in speaker_to_id.items()}

    with open(reverse_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_speaker, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved reverse mapping to {reverse_path}")
    print()

    return speaker_to_id


def main():
    parser = argparse.ArgumentParser(description="Build speaker ID mapping")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to train_manifest.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save speaker_mapping.json"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Number of top speakers to include (default: 500)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required for a speaker (default: 10)"
    )

    args = parser.parse_args()

    build_speaker_mapping(
        args.manifest,
        args.output,
        args.top_k,
        args.min_samples
    )


if __name__ == "__main__":
    main()
