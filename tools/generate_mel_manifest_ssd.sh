#!/usr/bin/env bash
# sdb1 manifest를 mel 있는 샘플 우선으로 정렬 (incremental)
# - 기존 manifest 순서 유지
# - 새로 추가된 mel 샘플만 앞에 삽입
# - WSD 스케줄은 전체 데이터셋 기준
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/.venv/bin/activate"

python << 'EOF'
"""
Incremental manifest 갱신
- 기존 gpt_pairs_train_mel.jsonl 순서 유지
- 새로 추가된 mel 샘플만 기존 mel 섹션 뒤에 삽입
"""
import json
import os
from pathlib import Path

BASE_MANIFEST = "/mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl"
OUTPUT = "/mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_train_mel.jsonl"
MEL_DIR = "/mnt/sdb1/emilia-yodas/KO_preprocessed/mel"

# mel 파일 목록 로드
print("Loading mel file list...")
mel_files = set(os.listdir(MEL_DIR))
print(f"Found {len(mel_files)} mel files")

# 기존 manifest 확인
existing_manifest = Path(OUTPUT)
if existing_manifest.exists():
    print(f"\n기존 manifest 발견: {OUTPUT}")

    # 기존 manifest에서 mel 유무 분리
    existing_with_mel = []
    existing_without_mel = []

    print("기존 manifest 분석...")
    with open(OUTPUT, 'r') as f:
        for line in f:
            data = json.loads(line)
            target_id = data['target_id']
            if data.get('target_mel_path'):
                existing_with_mel.append(data)
            else:
                existing_without_mel.append(data)

    print(f"  기존 mel 샘플: {len(existing_with_mel)}개")
    print(f"  기존 non-mel 샘플: {len(existing_without_mel)}개")

    # 기존 mel 샘플 중 실제 파일 없는 것 걸러내기 (manifest 부풀림 방지)
    existing_with_mel_valid = []
    missing_mel_from_existing = []
    for data in existing_with_mel:
        target_id = data['target_id']
        mel_file = f"{target_id}.npy"
        if mel_file in mel_files:
            existing_with_mel_valid.append(data)
        else:
            # 실제 mel 파일이 없으므로 non-mel로 이동
            data.pop('target_mel_path', None)
            missing_mel_from_existing.append(data)

    if missing_mel_from_existing:
        print(f"  ⚠️  기존 manifest에 mel 경로가 있지만 파일이 없는 샘플: {len(missing_mel_from_existing)}개 → non-mel로 이동")

    # existing_without_mel 앞에 누락분을 합쳐서 다시 처리
    existing_without_mel = missing_mel_from_existing + existing_without_mel
    existing_with_mel = existing_with_mel_valid

    # 새로 추가된 mel 찾기 (기존 without_mel 중에서)
    newly_added = []
    still_without_mel = []

    print("\n새로 추가된 mel 확인...")
    for data in existing_without_mel:
        target_id = data['target_id']
        mel_file = f"{target_id}.npy"

        if mel_file in mel_files:
            data['target_mel_path'] = f"mel/{mel_file}"
            newly_added.append(data)
        else:
            still_without_mel.append(data)

    print(f"  새로 추가된 mel: {len(newly_added)}개")
    print(f"  여전히 mel 없음: {len(still_without_mel)}개")

    # 최종 순서: [기존 mel] + [새 mel] + [mel 없음]
    final_with_mel = existing_with_mel + newly_added
    final_without_mel = still_without_mel

else:
    print(f"\n기존 manifest 없음. 새로 생성...")

    # 처음 생성: 전체 정렬
    final_with_mel = []
    final_without_mel = []

    print("Base manifest 분류...")
    with open(BASE_MANIFEST, 'r') as fin:
        for i, line in enumerate(fin):
            data = json.loads(line)
            target_id = data['target_id']
            mel_file = f"{target_id}.npy"

            if mel_file in mel_files:
                data['target_mel_path'] = f"mel/{mel_file}"
                final_with_mel.append(data)
            else:
                final_without_mel.append(data)

            if (i + 1) % 500000 == 0:
                print(f"  Processed {i+1} samples...")

print(f"\n=== 최종 결과 ===")
print(f"mel 있는 샘플: {len(final_with_mel)}개")
print(f"mel 없는 샘플: {len(final_without_mel)}개")
print(f"전체: {len(final_with_mel) + len(final_without_mel)}개")

# 저장
print(f"\nWriting to {OUTPUT}...")
with open(OUTPUT, 'w') as fout:
    for data in final_with_mel:
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    for data in final_without_mel:
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"\n✅ Done!")
print(f"  → mel 샘플 {len(final_with_mel)}개 (앞)")
print(f"  → non-mel 샘플 {len(final_without_mel)}개 (뒤)")
print(f"  → WSD 스케줄: 전체 {len(final_with_mel) + len(final_without_mel)}개 기준")
EOF
