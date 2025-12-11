#!/usr/bin/env bash
# Move GPT checkpoints off /mnt/sda1 onto /mnt/sdb1 and leave a symlink behind.

set -euo pipefail

SRC_DIR="${SRC_DIR:-/mnt/sda1/models/index-tts-ko/checkpoints}"
DST_BASE="${DST_BASE:-/mnt/sdb1/index-tts-ko}"
DST_DIR="${DST_BASE}/checkpoints"
BACKUP_SUFFIX="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${SRC_DIR}.backup.${BACKUP_SUFFIX}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "[ERROR] Source checkpoint directory not found: ${SRC_DIR}" >&2
  exit 1
fi

if mountpoint -q /mnt/sdb1; then
  echo "[INFO] /mnt/sdb1 mounted."
else
  echo "[WARN] /mnt/sdb1 is not a mountpoint (continuing anyway)." >&2
fi

mkdir -p "${DST_BASE}"

if [[ -d "${DST_DIR}" ]]; then
  echo "[INFO] Destination already exists: ${DST_DIR}" >&2
else
  echo "[INFO] Copying checkpoints to ${DST_DIR} (rsync)." >&2
  rsync -avh --progress "${SRC_DIR}/" "${DST_DIR}/"
fi

if [[ ! -d "${DST_DIR}" ]]; then
  echo "[ERROR] Destination copy failed: ${DST_DIR}" >&2
  exit 1
fi

if [[ -L "${SRC_DIR}" ]]; then
  echo "[ERROR] Source path is already a symlink: ${SRC_DIR}" >&2
  exit 1
fi

echo "[INFO] Renaming original ${SRC_DIR} -> ${BACKUP_DIR}" >&2
mv "${SRC_DIR}" "${BACKUP_DIR}"

echo "[INFO] Creating symlink ${SRC_DIR} -> ${DST_DIR}" >&2
ln -s "${DST_DIR}" "${SRC_DIR}"

echo "[DONE] Checkpoints now live at ${DST_DIR}" >&2
echo "       Original data saved at ${BACKUP_DIR}" >&2
echo "       Verify everything looks good, then remove the backup to reclaim space." >&2
