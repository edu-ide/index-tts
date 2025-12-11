#!/usr/bin/env python3
"""
추가 mel 전처리 스크립트 (target_audio → mel)

mel이 없는 샘플에 대해 target_audio에서 mel-spectrogram을 추출합니다.
기존 mel 파일은 건너뛰고 새로운 mel만 생성합니다.

Usage:
    python tools/preprocess_target_mel.py \
        --manifest /mnt/sdb1/emilia-yodas/KO_preprocessed/gpt_pairs_train.jsonl \
        --data-dir /mnt/sdb1/emilia-yodas/KO_preprocessed \
        --num-workers 32 \
        --batch-size 50000

After preprocessing, run:
    bash tools/generate_mel_manifest_ssd.sh
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*")

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


# s2mel config.yaml 기준 mel 설정
MEL_CONFIG = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 256,
    "n_mels": 80,
    "f_min": 0,
    "f_max": None,
}

# Global state for workers
_data_dir = None


def init_worker(data_dir: str):
    """Worker 초기화"""
    global _data_dir
    _data_dir = Path(data_dir)
    warnings.filterwarnings("ignore")


def compute_mel_spectrogram(audio_path: str) -> Optional[np.ndarray]:
    """오디오 파일에서 mel-spectrogram 계산"""
    try:
        waveform, sr = torchaudio.load(audio_path)

        # mono로 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        target_sr = MEL_CONFIG["sample_rate"]
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # mel-spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=MEL_CONFIG["n_fft"],
            win_length=MEL_CONFIG["win_length"],
            hop_length=MEL_CONFIG["hop_length"],
            n_mels=MEL_CONFIG["n_mels"],
            f_min=MEL_CONFIG["f_min"],
            f_max=MEL_CONFIG["f_max"],
            power=2.0,
            normalized=False,
        )

        mel = mel_transform(waveform)  # [1, n_mels, time]
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.squeeze(0).transpose(0, 1).numpy()  # [time, n_mels]

        return mel.astype(np.float32)
    except Exception as e:
        return None


def process_sample(sample: dict) -> Tuple[str, str]:
    """단일 샘플 처리 - target_audio에서 mel 추출"""
    global _data_dir

    target_id = sample.get("target_id")
    if not target_id:
        return target_id or "unknown", "no_target_id"

    # mel 파일 경로
    mel_path = _data_dir / "mel" / f"{target_id}.npy"

    # 이미 존재하면 skip
    if mel_path.exists():
        return target_id, "exists"

    # target_audio 경로
    audio_rel_path = sample.get("target_audio_path")
    if not audio_rel_path:
        return target_id, "no_audio_path"

    audio_path = _data_dir / audio_rel_path
    if not audio_path.exists():
        return target_id, "audio_not_found"

    # mel 계산
    mel = compute_mel_spectrogram(str(audio_path))
    if mel is None:
        return target_id, "mel_failed"

    # 저장
    mel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_path, mel)

    return target_id, "success"


def main():
    parser = argparse.ArgumentParser(description="추가 mel 전처리 (target_audio → mel)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="입력 manifest (gpt_pairs_train.jsonl)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="데이터 디렉토리 (KO_preprocessed)")
    parser.add_argument("--num-workers", type=int, default=32,
                        help="병렬 worker 수")
    parser.add_argument("--batch-size", type=int, default=100000,
                        help="한 번에 처리할 샘플 수 (메모리 관리)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="시작 라인 번호 (재시작용)")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 최대 샘플 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mel_dir = data_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    # 기존 mel 파일 수 확인
    existing_mel = len(list(mel_dir.glob("*.npy")))
    print(f"[Info] 기존 mel 파일: {existing_mel:,}개")

    # manifest 로드
    print(f"[Info] Manifest 로드: {args.manifest}")
    samples = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < args.start_from:
                continue
            if args.limit and len(samples) >= args.limit:
                break
            if line.strip():
                samples.append(json.loads(line))

    total = len(samples)
    print(f"[Info] 처리할 샘플: {total:,}개")
    print(f"[Info] Workers: {args.num_workers}")
    print(f"[Info] Batch size: {args.batch_size:,}")
    print(f"[Info] Mel config: sr={MEL_CONFIG['sample_rate']}, n_mels={MEL_CONFIG['n_mels']}")
    print()

    # 통계
    stats = {"success": 0, "exists": 0, "no_audio_path": 0, "audio_not_found": 0, "mel_failed": 0}

    # 배치 단위로 처리 (메모리 관리)
    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch = samples[batch_start:batch_end]

        print(f"\n[Batch] {batch_start:,} ~ {batch_end:,} ({len(batch):,}개)")

        with Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(str(data_dir),)
        ) as pool:
            with tqdm(total=len(batch), desc="Processing", unit="samples",
                      dynamic_ncols=True, smoothing=0.05) as pbar:

                chunksize = max(1, len(batch) // (args.num_workers * 100))
                chunksize = min(chunksize, 1000)

                for target_id, status in pool.imap_unordered(process_sample, batch, chunksize=chunksize):
                    if status in stats:
                        stats[status] += 1

                    pbar.set_postfix({
                        "new": stats["success"],
                        "skip": stats["exists"],
                        "err": stats["audio_not_found"] + stats["mel_failed"]
                    })
                    pbar.update(1)

    # 결과 출력
    print()
    print("=" * 60)
    print("📊 Results:")
    print(f"   ✅ 새로 생성: {stats['success']:,}")
    print(f"   ⏭️  이미 존재: {stats['exists']:,}")
    print(f"   ❌ 오디오 없음: {stats['audio_not_found']:,}")
    print(f"   ❌ mel 실패: {stats['mel_failed']:,}")
    print("=" * 60)

    # 최종 mel 파일 수
    final_mel = len(list(mel_dir.glob("*.npy")))
    print(f"\n[Info] 최종 mel 파일: {final_mel:,}개 (+{final_mel - existing_mel:,})")

    print()
    print("✅ Complete!")
    print()
    print("다음 단계:")
    print("  1. manifest 재생성:")
    print("     bash tools/generate_mel_manifest_ssd.sh")
    print()
    print("  2. 학습 재시작:")
    print("     bash tools/run_stage2_training.sh")


if __name__ == "__main__":
    main()
