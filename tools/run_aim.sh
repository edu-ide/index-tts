#!/usr/bin/env bash
# Start Aim web UI pointing to the current repo's .aim
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[ERROR] 먼저 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' 로 가상환경 활성화 후 실행하세요." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-43800}"

echo "Starting Aim at ${HOST}:${PORT} (repo=${REPO_DIR})"
exec aim up --repo "${REPO_DIR}" --host "${HOST}" --port "${PORT}"
