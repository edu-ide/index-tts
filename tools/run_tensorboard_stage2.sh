#!/usr/bin/env bash
# TensorBoard launcher for Stage2 (binds 0.0.0.0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Activate venv if present
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

LOGDIR="${LOGDIR:-/mnt/sda1/models/index-tts-ko/stage2_lowlr/logs}"
PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"

echo "Starting TensorBoard"
echo "  Logdir: ${LOGDIR}"
echo "  Host:   ${HOST}"
echo "  Port:   ${PORT}"
echo ""

exec tensorboard --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}"
