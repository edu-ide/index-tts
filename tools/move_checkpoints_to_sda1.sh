#!/usr/bin/env bash
# Move GPT checkpoints back to /mnt/sda1 and remove the /mnt/sdb1 symlink setup.

set -euo pipefail

SRC_LINK="${SRC_LINK:-/mnt/sda1/models/index-tts-ko/checkpoints}"
REAL_SRC="${REAL_SRC:-/mnt/sdb1/index-tts-ko/checkpoints}"
DEST_DIR="${DEST_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints_new}"
FINAL_DIR="/mnt/sda1/models/index-tts-ko/checkpoints"
BACKUP_DIR="${FINAL_DIR}.backup.$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "${REAL_SRC}" ]]; then
  echo "[ERROR] Source checkpoint dir not found: ${REAL_SRC}" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"
echo "[INFO] Copying checkpoints from ${REAL_SRC} -> ${DEST_DIR}" >&2
rsync -avh --progress "${REAL_SRC}/" "${DEST_DIR}/"

if [[ -L "${SRC_LINK}" ]]; then
  echo "[INFO] Renaming current symlink/original to ${BACKUP_DIR}" >&2
  mv "${SRC_LINK}" "${BACKUP_DIR}"
fi

echo "[INFO] Moving ${DEST_DIR} -> ${FINAL_DIR}" >&2
mv "${DEST_DIR}" "${FINAL_DIR}"

echo "[DONE] Checkpoints now stored at ${FINAL_DIR}" >&2
echo "[INFO] Previous link saved at ${BACKUP_DIR}" >&2
