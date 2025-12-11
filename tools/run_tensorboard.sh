#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${LOGDIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
EXTRA_FLAGS=("${@}")

export TB_PLUGINS_DISABLE="${TB_PLUGINS_DISABLE:-projector}"
echo "[TensorBoard] logdir=${LOGDIR} host=${HOST} port=${PORT} (TB_PLUGINS_DISABLE=${TB_PLUGINS_DISABLE})" >&2
exec tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}" "${EXTRA_FLAGS[@]}"
