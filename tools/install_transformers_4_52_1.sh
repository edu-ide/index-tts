#!/usr/bin/env bash
# Install the transformers version required by IndexTTS2 (>=4.52.1,<4.53).
# The script prefers uv when available, otherwise falls back to `python -m pip`.

set -euo pipefail

TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.52.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMEOUT_BIN="${TIMEOUT_BIN:-timeout}"
INSTALL_TIMEOUT="${INSTALL_TIMEOUT:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_install() {
    local cmd=()
    if command -v uv >/dev/null 2>&1; then
        log "Detected uv; installing transformers==${TRANSFORMERS_VERSION} with uv."
        cmd=(uv pip install --no-deps "transformers==${TRANSFORMERS_VERSION}")
    else
        log "uv not found; falling back to ${PYTHON_BIN} -m pip install."
        cmd=("${PYTHON_BIN}" -m pip install --no-deps --upgrade "transformers==${TRANSFORMERS_VERSION}")
    fi

    if [[ "${INSTALL_TIMEOUT}" -gt 0 ]]; then
        "${TIMEOUT_BIN}" "${INSTALL_TIMEOUT}" "${cmd[@]}"
    else
        "${cmd[@]}"
    fi
}

log "Ensuring transformers==${TRANSFORMERS_VERSION} is installed."
run_install

log "Verifying installed version."
"${PYTHON_BIN}" - <<'PY'
import transformers
print(f"transformers {transformers.__version__}")
PY

log "Done."
