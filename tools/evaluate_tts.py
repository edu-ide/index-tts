#!/usr/bin/env python3
"""
TTS 자동 평가 스크립트
- 체크포인트에서 샘플 음성 생성
- Whisper ASR로 WER (Word Error Rate) 계산
- 결과를 WandB에 업로드

사용법:
    python evaluate_tts.py --checkpoint <path> --test-manifest <path>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torchaudio
from jiwer import wer, cer  # pip install jiwer

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Whisper for ASR (optional but recommended)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[Warning] Whisper not available. Install with: pip install openai-whisper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TTS checkpoint quality")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint to evaluate")
    parser.add_argument(
        "--test-manifest",
        type=Path,
        required=True,
        help="Test manifest (JSONL with ground truth text)"
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--output-dir", type=Path, default=Path("./eval_results"), help="Output directory")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--wandb-project", type=str, default="indextts-korean", help="WandB project")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_test_samples(manifest_path: Path, num_samples: int) -> List[Dict]:
    """Load test samples from manifest"""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                record = json.loads(line.strip())
                samples.append(record)
            except json.JSONDecodeError:
                print(f"[Warning] Failed to parse line {i+1}")
                continue
    return samples


def generate_audio_from_checkpoint(
    checkpoint_path: Path,
    text: str,
    device: str
) -> Tuple[np.ndarray, int]:
    """
    체크포인트에서 오디오 생성

    참고: IndexTTS-2의 실제 inference 코드로 교체 필요
    현재는 placeholder 구현
    """
    # TODO: 실제 IndexTTS-2 inference 코드로 교체
    # from indextts.inference import generate_audio
    # audio, sr = generate_audio(checkpoint_path, text, device)

    print(f"[TODO] Generate audio for: {text[:50]}...")
    # Placeholder: 빈 오디오 반환
    sr = 24000
    audio = np.zeros(sr * 2, dtype=np.float32)  # 2초 무음
    return audio, sr


def transcribe_with_whisper(
    audio: np.ndarray,
    sample_rate: int,
    whisper_model_name: str,
    device: str
) -> str:
    """Whisper로 음성 인식"""
    if not WHISPER_AVAILABLE:
        return "[Whisper not available]"

    # Whisper 모델 로드 (캐싱됨)
    if not hasattr(transcribe_with_whisper, '_model'):
        print(f"[Info] Loading Whisper model: {whisper_model_name}")
        transcribe_with_whisper._model = whisper.load_model(whisper_model_name, device=device)

    model = transcribe_with_whisper._model

    # Whisper는 16kHz 요구
    if sample_rate != 16000:
        import librosa
        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    else:
        audio_16k = audio

    # Whisper 추론
    result = model.transcribe(audio_16k, language="ko", fp16=(device == "cuda"))
    return result["text"].strip()


def calculate_metrics(
    ground_truth: str,
    hypothesis: str
) -> Dict[str, float]:
    """WER/CER 계산"""
    # 정규화 (공백, 소문자 등)
    gt_normalized = ground_truth.lower().strip()
    hyp_normalized = hypothesis.lower().strip()

    # WER (Word Error Rate)
    word_error_rate = wer(gt_normalized, hyp_normalized)

    # CER (Character Error Rate)
    char_error_rate = cer(gt_normalized, hyp_normalized)

    return {
        "wer": word_error_rate * 100,  # Percentage
        "cer": char_error_rate * 100,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    test_samples: List[Dict],
    args: argparse.Namespace
) -> Dict[str, float]:
    """체크포인트 평가"""
    device = args.device
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_wer = []
    all_cer = []
    results = []

    print(f"[Info] Evaluating {len(test_samples)} samples from {checkpoint_path}")

    for i, sample in enumerate(test_samples):
        # Ground truth 텍스트
        ground_truth = sample.get("text", "")
        if not ground_truth:
            print(f"[Warning] Sample {i+1} has no text, skipping")
            continue

        # 1. 음성 생성
        try:
            audio, sr = generate_audio_from_checkpoint(checkpoint_path, ground_truth, device)
        except Exception as e:
            print(f"[Error] Failed to generate audio for sample {i+1}: {e}")
            continue

        # 2. 음성 인식 (Whisper)
        try:
            hypothesis = transcribe_with_whisper(audio, sr, args.whisper_model, device)
        except Exception as e:
            print(f"[Error] Failed to transcribe sample {i+1}: {e}")
            hypothesis = "[transcription failed]"

        # 3. 메트릭 계산
        if hypothesis != "[transcription failed]" and hypothesis != "[Whisper not available]":
            metrics = calculate_metrics(ground_truth, hypothesis)
            all_wer.append(metrics["wer"])
            all_cer.append(metrics["cer"])

            result = {
                "sample_id": i + 1,
                "ground_truth": ground_truth,
                "hypothesis": hypothesis,
                "wer": metrics["wer"],
                "cer": metrics["cer"],
            }
            results.append(result)

            print(f"[Sample {i+1}/{len(test_samples)}] WER: {metrics['wer']:.2f}% CER: {metrics['cer']:.2f}%")

        # 오디오 저장 (처음 5개만)
        if i < 5:
            audio_path = output_dir / f"sample_{i+1}.wav"
            torchaudio.save(str(audio_path), torch.from_numpy(audio).unsqueeze(0), sr)

    # 평균 메트릭
    avg_metrics = {
        "avg_wer": np.mean(all_wer) if all_wer else 0.0,
        "avg_cer": np.mean(all_cer) if all_cer else 0.0,
        "std_wer": np.std(all_wer) if all_wer else 0.0,
        "std_cer": np.std(all_cer) if all_cer else 0.0,
        "num_samples": len(all_wer),
    }

    # 결과 저장
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "metrics": avg_metrics,
            "samples": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[Results] Average WER: {avg_metrics['avg_wer']:.2f}% ± {avg_metrics['std_wer']:.2f}%")
    print(f"[Results] Average CER: {avg_metrics['avg_cer']:.2f}% ± {avg_metrics['std_cer']:.2f}%")
    print(f"[Results] Results saved to {results_file}")

    return avg_metrics


def main():
    args = parse_args()

    # WandB 초기화
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval_{args.checkpoint.stem}",
            config=vars(args),
            job_type="evaluation",
        )
        print(f"[Info] WandB initialized for evaluation")

    # 테스트 샘플 로드
    test_samples = load_test_samples(args.test_manifest, args.num_samples)
    print(f"[Info] Loaded {len(test_samples)} test samples")

    if len(test_samples) == 0:
        print("[Error] No valid test samples found")
        sys.exit(1)

    # 평가 실행
    metrics = evaluate_checkpoint(args.checkpoint, test_samples, args)

    # WandB 업로드
    if use_wandb:
        wandb.log({
            "eval/wer": metrics["avg_wer"],
            "eval/cer": metrics["avg_cer"],
            "eval/wer_std": metrics["std_wer"],
            "eval/cer_std": metrics["std_cer"],
            "eval/num_samples": metrics["num_samples"],
        })

        # 오디오 샘플 업로드 (처음 5개)
        audio_files = sorted(args.output_dir.glob("sample_*.wav"))
        for audio_file in audio_files[:5]:
            wandb.log({f"eval/audio_{audio_file.stem}": wandb.Audio(str(audio_file))})

        print("[Info] Results uploaded to WandB")
        wandb.finish()

    print("\n[Info] Evaluation complete!")

    # 다음 단계 안내
    print("\n다음 단계:")
    print("  1. WER < 10%: 매우 우수한 품질")
    print("  2. WER 10-20%: 우수한 품질")
    print("  3. WER 20-30%: 양호한 품질")
    print("  4. WER > 30%: 추가 학습 필요")

    if metrics["avg_wer"] > 30:
        print("\n⚠️  WER이 30% 이상입니다. 학습을 더 진행하거나 하이퍼파라미터 조정이 필요합니다.")


if __name__ == "__main__":
    main()
