#!/usr/bin/env bash
# mel 데이터를 sda1에서 sdb1으로 복사
set -euo pipefail

SRC="/mnt/sda1/emilia-yodas/KO_preprocessed/mel"
DST="/mnt/sdb1/emilia-yodas/KO_preprocessed/"

echo "=== mel 데이터 복사 ==="
echo "Source: $SRC ($(du -sh $SRC | cut -f1))"
echo "Destination: $DST"
echo ""

rsync -av --progress "$SRC" "$DST"

echo ""
echo "=== 복사 완료 ==="
du -sh "$DST/mel"
