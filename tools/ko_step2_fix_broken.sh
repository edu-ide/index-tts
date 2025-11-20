#!/usr/bin/env bash
# Utility to regenerate only the broken manifest entries reported by report.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATASET_DIR="${DATASET_DIR:-/mnt/sda1/emilia-yodas/KO_preprocessed}"
RAW_MANIFEST="${RAW_MANIFEST:-/mnt/sda1/emilia-yodas/KO/ko_manifest_raw.jsonl}"
REPORT_PATH="${REPORT_PATH:-${DATASET_DIR}/report.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
KEEP_TMP=0
SCAN_EMPTY=0
SCAN_MISSING=0
CLEAN_CACHE=1
MANUAL_IDS=()

usage() {
  cat <<'USAGE'
Usage: ko_step2_fix_broken.sh [options]

Options:
  --dataset <dir>       Dataset directory containing train/val manifests (default: $DATASET_DIR)
  --raw-manifest <path> Original raw manifest to source clean entries from
  --report <path>       Path to report.json (default: <dataset>/report.json)
  --id <SAMPLE_ID>      Regenerate an explicit ID (repeatable)
  --keep-tmp            Keep the temporary manifest instead of deleting it
  --no-clean-cache      Do not delete existing .npy artifacts (not recommended)
  -h, --help            Show this message

Environment:
  VAL_RATIO, BATCH_SIZE, DEVICE 등은 ko_step2_preprocess.sh 에 전달하고 싶다면
  환경변수로 설정해서 호출하면 됩니다.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_DIR="$2"
      shift
      ;;
    --raw-manifest)
      RAW_MANIFEST="$2"
      shift
      ;;
    --report)
      REPORT_PATH="$2"
      shift
      ;;
    --id)
      MANUAL_IDS+=("$2")
      shift
      ;;
    --keep-tmp)
      KEEP_TMP=1
      ;;
    --scan-empty)
      SCAN_EMPTY=1
      ;;
    --scan-missing)
      SCAN_MISSING=1
      ;;
    --no-clean-cache)
      CLEAN_CACHE=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

RAW_PARENT="$(dirname "${RAW_MANIFEST}")"
TMP_MANIFEST="$(mktemp "${RAW_PARENT}/ko_manifest_fix.XXXXXX.jsonl")"
cleanup() {
  if [[ ${KEEP_TMP} -eq 0 && -n "${TMP_MANIFEST:-}" && -f "${TMP_MANIFEST}" ]]; then
    rm -f "${TMP_MANIFEST}"
  fi
}
trap cleanup EXIT

if [[ ! -f "${RAW_MANIFEST}" ]]; then
  echo "[ERROR] Raw manifest not found: ${RAW_MANIFEST}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "[ERROR] Dataset directory not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${REPORT_PATH}" && ${#MANUAL_IDS[@]} -eq 0 ]]; then
  echo "[ERROR] report.json not found (${REPORT_PATH}) and no --id provided." >&2
  exit 1
fi

if [[ ${#MANUAL_IDS[@]} -gt 0 ]]; then
  MANUAL_IDS_CSV=$(IFS=,; printf '%s' "${MANUAL_IDS[*]}")
else
MANUAL_IDS_CSV=""
fi

mapfile -t BROKEN_IDS < <(
  DATASET_DIR="${DATASET_DIR}" \
  RAW_MANIFEST="${RAW_MANIFEST}" \
  REPORT_PATH="${REPORT_PATH}" \
  TMP_MANIFEST="${TMP_MANIFEST}" \
  MANUAL_IDS_CSV="${MANUAL_IDS_CSV}" \
  SCAN_EMPTY="${SCAN_EMPTY}" \
  SCAN_MISSING="${SCAN_MISSING}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import re
import sys
from pathlib import Path

pattern = re.compile(r'"id"\s*:\s*"([^"]+)"')

def build_tmp_manifest(raw_manifest: Path, ids: set[str], tmp_manifest: Path) -> None:
    found: set[str] = set()
    with raw_manifest.open('r', encoding='utf-8') as src, tmp_manifest.open('w', encoding='utf-8') as dst:
        for line in src:
            record = json.loads(line)
            rec_id = record.get('id')
            if rec_id in ids:
                dst.write(line)
                found.add(rec_id)
    missing = ids - found
    if missing:
        raise SystemExit(
            f"Unable to locate {len(missing)} id(s) in raw manifest: {', '.join(sorted(missing))}"
        )


def strip_lines(manifest_path: Path, line_numbers: list[int]) -> list[str]:
    if not line_numbers:
        return []
    bad_set = set(line_numbers)
    seen: set[int] = set()
    collected: list[str] = []
    tmp_path = manifest_path.with_suffix('.cleaning')
    with manifest_path.open('r', encoding='utf-8', errors='ignore') as src, tmp_path.open(
        'w', encoding='utf-8'
    ) as dst:
        for lineno, raw in enumerate(src, 1):
            if lineno in bad_set:
                collected.append(raw.rstrip('\n'))
                seen.add(lineno)
                continue
            dst.write(raw)
    tmp_path.replace(manifest_path)
    missing = sorted(bad_set - seen)
    if missing:
        raise SystemExit(
            f"Failed to find line(s) {missing} in {manifest_path}. Did the file change since report.json was generated?"
        )
    return collected


def extract_ids(text: str) -> list[str]:
    return pattern.findall(text)


def main() -> None:
    dataset_dir = Path(os.environ['DATASET_DIR']).expanduser().resolve()
    raw_manifest = Path(os.environ['RAW_MANIFEST']).expanduser().resolve()
    report_path = Path(os.environ['REPORT_PATH']).expanduser()
    tmp_manifest = Path(os.environ['TMP_MANIFEST']).resolve()
    manual_ids = {token for token in os.environ.get('MANUAL_IDS_CSV', '').split(',') if token}
    scan_empty = os.environ.get('SCAN_EMPTY', '0') == '1'
    scan_missing = os.environ.get('SCAN_MISSING', '0') == '1'

    ids_to_fix: set[str] = set(manual_ids)

    if report_path.exists():
        with report_path.open('r', encoding='utf-8') as handle:
            report = json.load(handle)
    else:
        report = {}

    manifests_to_clean: list[tuple[Path, list[int]]] = []
    for key, filename in ('train_manifest', 'train_manifest.jsonl'), ('val_manifest', 'val_manifest.jsonl'):
        info = report.get(key)
        errors = (info or {}).get('errors') or []
        if not errors:
            continue
        manifest_path = Path((info or {}).get('path') or dataset_dir / filename)
        line_numbers = [int(err['line']) for err in errors if 'line' in err]
        if line_numbers:
            manifests_to_clean.append((manifest_path, line_numbers))

    for manifest_path, line_numbers in manifests_to_clean:
        collected_lines = strip_lines(manifest_path, line_numbers)
        for text in collected_lines:
            found_ids = extract_ids(text)
            if not found_ids:
                print(
                    f"[WARN] Unable to extract ID from corrupted line in {manifest_path}.",
                    file=sys.stderr,
                )
            ids_to_fix.update(found_ids)

    if scan_empty:
        subdirs = ['text_ids', 'codes', 'condition', 'emo_vec']
        for sub in subdirs:
            folder = dataset_dir / sub
            if not folder.exists():
                continue
            for path in folder.glob('*.npy'):
                try:
                    if path.stat().st_size == 0:
                        ids_to_fix.add(path.stem)
                except OSError:
                    continue

    if scan_missing:
        def scan_manifest_file(manifest_path: Path) -> set[str]:
            missing_ids: set[str] = set()
            if not manifest_path.exists():
                return missing_ids
            base = dataset_dir
            with manifest_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rel_path = record.get('text_ids_path')
                    sample_id = record.get('id')
                    if not rel_path or not sample_id:
                        continue
                    target = (base / rel_path).resolve()
                    if not target.exists() or target.stat().st_size == 0:
                        missing_ids.add(sample_id)
            return missing_ids

        for manifest_name in ('train_manifest.jsonl', 'val_manifest.jsonl'):
            missing = scan_manifest_file(dataset_dir / manifest_name)
            if missing:
                print(f"[INFO] scan_missing: {manifest_name} missing {len(missing)} id(s)", file=sys.stderr)
                ids_to_fix.update(missing)

    if not ids_to_fix:
        return

    build_tmp_manifest(raw_manifest, ids_to_fix, tmp_manifest)

    for sample_id in sorted(ids_to_fix):
        print(sample_id)


if __name__ == '__main__':
    main()
PY
)

if [[ ${#BROKEN_IDS[@]} -eq 0 ]]; then
  echo "[INFO] 재생성할 ID가 없습니다. (report가 이미 정리된 상태일 수 있음)" >&2
  exit 0
fi

echo "[INFO] 대상 ID (${#BROKEN_IDS[@]}): ${BROKEN_IDS[*]}" >&2

if [[ ${CLEAN_CACHE} -eq 1 ]]; then
  for sample_id in "${BROKEN_IDS[@]}"; do
    for subdir in text_ids codes condition emo_vec; do
      target="${DATASET_DIR}/${subdir}/${sample_id}.npy"
      if [[ -f "${target}" ]]; then
        rm -f "${target}"
      fi
    done
  done
else
  echo "[WARN] --no-clean-cache 사용 중. 기존 특징 파일이 있으면 --skip-existing 때문에 재처리가 건너뛰어질 수 있습니다." >&2
fi

echo "[INFO] ko_step2_preprocess.sh 를 ${#BROKEN_IDS[@]}개 ID에 대해 실행합니다." >&2
MANIFEST="${TMP_MANIFEST}" OUTPUT_DIR="${DATASET_DIR}" \
  "${SCRIPT_DIR}/ko_step2_preprocess.sh"

if [[ ${KEEP_TMP} -eq 1 ]]; then
  echo "[INFO] 임시 manifest 유지: ${TMP_MANIFEST}" >&2
else
  rm -f "${TMP_MANIFEST}"
fi
trap - EXIT
cleanup
