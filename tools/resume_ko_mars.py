#!/usr/bin/env python3
"""
Python wrapper to launch the MARS resume bash script.
Usage: set env vars as usual (CKPT, FRESH_OPT, LR, WSD_WARMUP, etc.) and run:
    python tools/resume_ko_mars.py
This simply shells out to tools/resume_ko_mars.sh so that Python-first runners
don't accidentally invoke the bash script with the wrong interpreter.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).with_suffix(".sh")
    if not script_path.exists():
        sys.stderr.write(f"[ERROR] Missing script: {script_path}\n")
        return 1

    try:
        subprocess.run(["bash", str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
