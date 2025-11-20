#!/usr/bin/env python3
"""
IndexTTS2 Stage 2 Training Wrapper.

This wrapper enables Stage 2 (GRL + Emotion Disentanglement) training
by modifying the base trainer with:
  1. GRL (Gradient Reversal Layer) enabled
  2. Speaker classification loss
  3. Speaker perceiver frozen
  4. Speaker ID mapping

Usage:
    STAGE2_MODE=1 SPEAKER_MAPPING=/path/to/speaker_mapping.json \
    GRL_LAMBDA=1.0 SPEAKER_LOSS_WEIGHT=0.1 \
    python trainers/train_gpt_v2.py [args...]

Stage 2 Environment Variables:
    STAGE2_MODE: Set to "1" to enable Stage 2 mode
    SPEAKER_MAPPING: Path to speaker_mapping.json
    GRL_LAMBDA: Gradient reversal strength (default: 1.0)
    SPEAKER_LOSS_WEIGHT: Weight for speaker classification loss (default: 0.1)
    GRL_SCHEDULE: Lambda scheduling ("constant", "linear", "exponential")
    FREEZE_SPEAKER_PERCEIVER: Freeze speaker perceiver (default: 1)

Example:
    export STAGE2_MODE=1
    export SPEAKER_MAPPING=/mnt/sda1/models/index-tts-ko/speaker_mapping.json
    export GRL_LAMBDA=1.0
    export SPEAKER_LOSS_WEIGHT=0.1
    export GRL_SCHEDULE=exponential

    python trainers/train_gpt_v2.py \
        --train-manifest /mnt/sda1/emilia-yodas/KO_preprocessed/train_manifest.jsonl \
        --val-manifest /mnt/sda1/emilia-yodas/KO_preprocessed/val_manifest.jsonl \
        --tokenizer /mnt/sda1/models/IndexTTS-2/tokenizer.model \
        --config /mnt/sda1/models/IndexTTS-2/config.yaml \
        --base-checkpoint /mnt/sda1/models/index-tts-ko/checkpoints/stage1_final.pth \
        --output-dir /mnt/sda1/models/index-tts-ko/stage2 \
        --batch-size 8 \
        --grad-accumulation 8 \
        --learning-rate 2e-4 \
        --warmup-steps 5000 \
        --epochs 2
"""

print("""
================================================================
⚠️  STAGE 2 WRAPPER DEPRECATION NOTICE
================================================================

This wrapper script is deprecated. Please use the integrated
Stage 2 mode in train_gpt_v2.py instead.

To enable Stage 2, set environment variables:
    export STAGE2_MODE=1
    export SPEAKER_MAPPING=/path/to/speaker_mapping.json

Then run train_gpt_v2.py normally:
    python trainers/train_gpt_v2.py [args...]

See tools/train_ko_stage2.sh for a complete example.
================================================================
""")
