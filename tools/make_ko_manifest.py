#!/usr/bin/env python3
"""Create or resume a JSONL manifest for the Emilia-Yodas Korean dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/sda1/emilia-yodas/KO")
DEFAULT_OUTPUT = DEFAULT_ROOT / "ko_manifest_raw.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Dataset root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest path (default: <root>/ko_manifest_raw.jsonl).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-listed samples when the manifest exists.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Emit progress logs after every N newly written samples.",
    )
    return parser.parse_args()


def load_existing_ids(output: Path) -> set[str]:
    existing: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            if isinstance(sample_id, str):
                existing.add(sample_id)
    return existing


def build_manifest(root: Path, output: Path, resume: bool, log_interval: int) -> int:
    """Scan `root` for JSON metadata files and emit/extend a JSONL manifest."""
    print(f"Scanning {root} for JSON metadata files...")

    existing_ids: set[str] = set()
    mode = "w"
    if resume and output.exists():
        print(f"--resume enabled, loading existing manifest: {output}")
        existing_ids = load_existing_ids(output)
        print(f"Found {len(existing_ids)} existing samples.")
        mode = "a"
    elif output.exists():
        print(f"Output {output} exists and --resume not set; it will be overwritten.")

    written = 0
    with output.open(mode, encoding="utf-8") as out:
        for json_path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Failed to read {json_path}: {exc}")
                continue

            audio_path = json_path.with_suffix(".mp3")
            if not audio_path.exists():
                print(f"Missing audio for {json_path}")
                continue

            text = (data.get("text") or "").strip()
            if not text:
                continue

            sample_id = data.get("_id") or json_path.stem
            if sample_id in existing_ids:
                continue

            record = {
                "id": sample_id,
                "text": text,
                "audio": str(audio_path.relative_to(root)),
                "speaker": data.get("speaker") or "",
                "language": data.get("language") or "ko",
            }

            if "duration" in data:
                record["duration"] = data["duration"]
            if "dnsmos" in data:
                record["dnsmos"] = data["dnsmos"]

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written and written % log_interval == 0:
                print(f"Processed {written} new samples so far...")

    total = len(existing_ids) + written
    if written:
        print(f"Added {written} new samples.")
    else:
        print("No new samples were added.")
    return total


def main() -> None:
    args = parse_args()
    root = args.root
    if args.output is not None:
        output = args.output
    elif root == DEFAULT_ROOT:
        output = DEFAULT_OUTPUT
    else:
        output = root / "ko_manifest_raw.jsonl"
    total = build_manifest(root, output, args.resume, args.log_interval)
    print(f"Manifest now contains {total} records at {output}")


if __name__ == "__main__":
    main()
