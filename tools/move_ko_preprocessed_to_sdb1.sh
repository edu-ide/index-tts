#!/usr/bin/env bash

set -euo pipefail

SRC=${SRC:-/mnt/sda1/emilia-yodas/KO_preprocessed}
DST=${DST:-/mnt/sdb1/emilia-yodas/KO_preprocessed}

if [[ ! -d "$SRC" ]]; then
  echo "[move_ko_preprocessed] Source directory '$SRC' not found or already moved." >&2
  exit 1
fi

if [[ -L "$SRC" ]]; then
  echo "[move_ko_preprocessed] '$SRC' is already a symlink. Nothing to do." >&2
  exit 0
fi

echo "[move_ko_preprocessed] Copying contents from $SRC to $DST (this can take a while, ~490GB)..."
mkdir -p "$DST"

rsync -a --info=progress2 --partial --delete "$SRC/" "$DST/"

BACKUP="${SRC%/}.backup.$(date +%Y%m%d_%H%M%S)"
echo "[move_ko_preprocessed] Copy complete. Renaming original to $BACKUP"
mv "$SRC" "$BACKUP"

echo "[move_ko_preprocessed] Creating symlink $SRC -> $DST"
ln -s "$DST" "$SRC"

cat <<MSG
[move_ko_preprocessed] Done.
  - New data lives in: $DST
  - Original directory kept as: $BACKUP (remove once satisfied)
  - Symlink created: $SRC -> $DST

To verify and remove backup:
  sudo rm -rf "$BACKUP"
MSG
