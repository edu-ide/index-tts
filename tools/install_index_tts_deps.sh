#!/usr/bin/env bash
# Install IndexTTS2 training dependencies for the current environment.
# Prefers uv (fast, deterministic). Falls back to pip.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if command -v uv >/dev/null 2>&1; then
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log "uv detected but no virtualenv active; please run 'source /mnt/sdc1/ws/workspace/.venv_indextts/bin/activate' first."
        exit 1
    fi
    log "Using uv to install dependencies."
    uv pip install --editable "${PROJECT_DIR}"
else
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log "uv not available and no virtualenv active; please activate /mnt/sdc1/ws/workspace/.venv_indextts first."
        exit 1
    fi
    log "uv not available; using pip."
    python3 -m pip install --upgrade pip
    python3 -m pip install --editable "${PROJECT_DIR}"
fi

log "IndexTTS dependencies installed."
