#!/usr/bin/env python3
"""
End-to-end finetuning entry point for IndexTTS2 (GPT module) with Japanese data.

This trainer expects the preprocessing pipeline to have produced manifests where each
sample record stores paths to:
  - text token ids (.npy, int32)
  - semantic codes (.npy, int32)
  - conditioning latent (.npy, float32 [32, hidden])
  - emotion vector (.npy, float32 [hidden])

The model is optimised with cross-entropy losses over text tokens and semantic codes,
with optional gradient accumulation and mixed-precision support. Checkpoints are
emitted every 1k optimiser steps (`model_step{N}.pth`), keeping only the three most
recent snapshots. TensorBoard summaries track losses and learning rate under the
chosen output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from prodigyopt import Prodigy
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.typical_sampling import TypicalLogitsWarper

try:
    from trainers.mars_optimizer_official import MARS
except ImportError:
    try:
        from mars_optimizer import MARS
    except ImportError:
        print("[Warning] MARS optimizer not found. Please ensure trainers/mars_optimizer.py exists.")

# Schedule-Free Optimizer Integration
try:
    from schedulefree import AdamWScheduleFree
    SCHEDULEFREE_AVAILABLE = True
except ImportError:
    SCHEDULEFREE_AVAILABLE = False
    print("[Warning] schedulefree not installed. Run: pip install schedulefree")

from indextts.utils.front import TextNormalizer, TextTokenizer
try:
    from transformers import get_wsd_schedule  # Official Huggingface implementation
except ImportError:
    from indextts.utils.scheduler import get_wsd_schedule_with_warmup as get_wsd_schedule

# Liger Kernel Integration (Memory & Speed Optimization)
try:
    import liger_kernel
    from liger_kernel.transformers import apply_liger_kernel_to_gpt2
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("[Warning] Liger Kernel not found. Install with `pip install liger-kernel` for memory savings.")

# Aim integration for experiment tracking
try:
    import aim
    from aim.sdk.errors import MissingRunError
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("[Warning] aim not installed. Run: pip install aim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune IndexTTS2 GPT on Japanese data.")
    parser.add_argument(
        "--train-manifest",
        dest="train_manifests",
        action="append",
        type=str,
        required=True,
        help="Training manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument(
        "--val-manifest",
        dest="val_manifests",
        action="append",
        type=str,
        required=True,
        help="Validation manifest JSONL. Repeat to mix multiple datasets; optionally suffix with '::lang' to force a language hint.",
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model path.")
    parser.add_argument(
        "--base-tokenizer",
        type=Path,
        default=None,
        help="Original SentencePiece model used by the base checkpoint (for embedding remap).",
    )
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"), help="Model config YAML.")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"), help="Base GPT checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"), help="Directory for checkpoints/logs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size per optimisation step.")
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="LR warmup steps.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional max optimiser steps (0 = unlimited).")
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between training log entries.")
    parser.add_argument("--val-interval", type=int, default=0, help="Validation frequency in steps (0 = once per epoch).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping value.")
    parser.add_argument("--text-loss-weight", type=float, default=0.2, help="Weight for text CE loss.")
    parser.add_argument("--mel-loss-weight", type=float, default=0.8, help="Weight for semantic CE loss.")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from, or 'auto'.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "prodigy", "mars", "schedulefree"], help="Optimizer to use.")
    parser.add_argument("--adamw-no-fused", action="store_true", help="Disable fused AdamW even if available.")
    parser.add_argument("--scheduler", type=str, default="wsd", choices=["cosine", "wsd", "none"], help="Scheduler choice: cosine, wsd (Warmup-Stable-Decay), or none.")
    parser.add_argument("--wsd-stable-ratio", type=float, default=0.9, help="Ratio of stable phase for WSD scheduler (default: 0.9).")
    parser.add_argument("--wsd-min-lr-ratio", type=float, default=0.0, help="Minimum LR ratio for WSD scheduler (default: 0.0).")
    parser.add_argument(
        "--cpu-ckpt-load",
        action="store_true",
        help="Force loading checkpoint to CPU instead of device (default: load to device for faster startup).",
    )
    parser.add_argument(
        "--duration-conditioning",
        type=str,
        default="binary",
        choices=["binary", "length"],
        help="Duration conditioning mode: binary uses legacy speed_emb 0/1; length ties to mel positional embeddings using target code length (per IndexTTS2).",
    )
    # Aim options
    parser.add_argument("--aim-experiment", type=str, default="indextts-korean", help="Aim experiment name.")
    parser.add_argument("--aim-run-name", type=str, default=None, help="Aim run name (auto-generated if not specified).")
    parser.add_argument("--aim-repo", type=str, default=".aim", help="Aim repository path.")
    parser.add_argument("--no-aim", action="store_true", help="Disable Aim tracking even if available.")
    # Stage 2 options
    parser.add_argument("--enable-grl", action="store_true", help="Enable GRL for Stage 2 emotion disentanglement.")
    parser.add_argument("--speaker-mapping", type=Path, default=None, help="Path to speaker_mapping.json for Stage 2.")
    parser.add_argument("--grl-lambda", type=float, default=1.0, help="GRL reversal strength (Stage 2).")
    parser.add_argument("--speaker-loss-weight", type=float, default=0.1, help="Weight for speaker classification loss (Stage 2).")
    parser.add_argument("--grl-schedule", type=str, default="exponential", choices=["constant", "linear", "exponential"], help="GRL lambda scheduling (Stage 2).")
    parser.add_argument("--enable-stage2-realtime-emo", action="store_true", help="Compute emo_vec in real-time during Stage 2 for proper gradient flow through emo encoder.")
    # Stage 3 options
    parser.add_argument("--freeze-conditioners", action="store_true", help="Freeze feature conditioners (speaker + emotion perceiver) for Stage 3 fine-tuning.")
    return parser.parse_args()


@dataclass
class ManifestSpec:
    path: Path
    language: Optional[str] = None


def parse_manifest_specs(entries: Sequence[str], flag_name: str) -> List[ManifestSpec]:
    if not entries:
        raise ValueError(f"{flag_name} requires at least one manifest path.")
    specs: List[ManifestSpec] = []
    for raw in entries:
        value = raw.strip()
        lang: Optional[str] = None
        for separator in ("::", "@", "="):
            if separator in value:
                path_str, lang_part = value.rsplit(separator, 1)
                value = path_str.strip()
                lang = lang_part.strip().lower() or None
                break
        path = Path(value).expanduser()
        specs.append(ManifestSpec(path=path, language=lang))
    return specs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)


@dataclass
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emo_vec_path: Path
    text_len: int
    code_len: int
    condition_len: int
    sample_type: str = "single"
    prompt_id: Optional[str] = None
    target_id: Optional[str] = None
    language: Optional[str] = None
    prompt_language: Optional[str] = None
    manifest_path: Optional[Path] = None
    speaker: Optional[str] = None  # Stage 2: Speaker name for GRL


class JapaneseGPTDataset(Dataset):
    def __init__(self, manifests: Sequence[ManifestSpec]):
        if isinstance(manifests, ManifestSpec):
            manifests = [manifests]
        manifest_list = list(manifests)
        if not manifest_list:
            raise ValueError("No manifest paths supplied.")

        self.samples: List[Sample] = []
        self.sample_type: str = "unknown"
        self.manifest_summaries: List[Dict[str, object]] = []
        self.bad_indices: Set[int] = set()

        for spec in manifest_list:
            self._load_single_manifest(spec)

        if not self.samples:
            manifest_paths = ", ".join(str(spec.path) for spec in manifest_list)
            raise RuntimeError(f"No entries found in the provided manifests: {manifest_paths}")
        if self.sample_type != "paired":
            raise RuntimeError(
                "The GPT trainer expects prompt/target pair manifests.\n"
                "Generate paired manifests with tools/build_gpt_prompt_pairs.py and retry."
            )

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        if not value:
            raise ValueError("Empty path provided in manifest record.")
        path = Path(value)
        if path.is_absolute():
            return path
        return (base_dir / path).expanduser()

    @staticmethod
    def _normalize_language(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped.lower() if stripped else None

    def _load_single_manifest(self, spec: ManifestSpec) -> None:
        manifest_path = spec.path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        local_count = 0
        local_languages: set[str] = set()
        manifest_sample_type: Optional[str] = None
        base_dir = manifest_path.parent

        print(f"[Info] Parsing manifest {manifest_path} ...")
        processed = 0
        progress_interval = 10000

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                processed += 1
                is_paired = "prompt_condition_path" in record and "target_codes_path" in record
                if is_paired:
                    emo_path_value = record.get("prompt_emo_vec_path") or record.get("target_emo_vec_path")
                    if not emo_path_value:
                        raise RuntimeError(
                            f"Paired manifest entry {record.get('id')} missing prompt_emo_vec_path."
                        )
                    target_language = self._normalize_language(
                        record.get("target_language") or record.get("language") or spec.language
                    )
                    prompt_language = self._normalize_language(record.get("prompt_language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["target_text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["target_codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["prompt_condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, emo_path_value),
                        text_len=int(record["target_text_len"]),
                        code_len=int(record["target_code_len"]),
                        condition_len=int(record.get("prompt_condition_len", 32)),
                        sample_type="paired",
                        prompt_id=record.get("prompt_id"),
                        target_id=record.get("target_id"),
                        language=target_language,
                        prompt_language=prompt_language,
                        manifest_path=manifest_path,
                        speaker=record.get("speaker"),  # Stage 2: Load speaker name
                    )
                else:
                    language = self._normalize_language(record.get("language") or spec.language)
                    sample = Sample(
                        id=record["id"],
                        text_ids_path=self._resolve_path(base_dir, record["text_ids_path"]),
                        codes_path=self._resolve_path(base_dir, record["codes_path"]),
                        condition_path=self._resolve_path(base_dir, record["condition_path"]),
                        emo_vec_path=self._resolve_path(base_dir, record["emo_vec_path"]),
                        text_len=int(record["text_len"]),
                        code_len=int(record["code_len"]),
                        condition_len=int(record.get("condition_len", 32)),
                        sample_type="single",
                        manifest_path=manifest_path,
                        language=language,
                        speaker=record.get("speaker"),  # Stage 2: Load speaker name
                    )

                if manifest_sample_type is None:
                    manifest_sample_type = sample.sample_type
                elif manifest_sample_type != sample.sample_type:
                    raise RuntimeError(
                        f"Manifest {manifest_path} mixes sample types ({manifest_sample_type} vs {sample.sample_type})."
                    )

                self.samples.append(sample)
                local_count += 1
                if sample.language:
                    local_languages.add(sample.language)
                if sample.prompt_language:
                    local_languages.add(sample.prompt_language)

                if processed % progress_interval == 0:
                    print(
                        f"  • processed {processed:,} entries "
                        f"(kept {local_count:,}) in {manifest_path.name}"
                    )

        if local_count:
            if processed % progress_interval != 0:
                print(
                    f"  • processed {processed:,} entries "
                    f"(kept {local_count:,}) in {manifest_path.name}"
                )
            if manifest_sample_type and manifest_sample_type != "paired":
                raise RuntimeError(
                    f"Manifest {manifest_path} contains '{manifest_sample_type}' entries. "
                    "This trainer expects prompt/target pair manifests (see tools/build_gpt_prompt_pairs.py)."
                )
            if self.sample_type == "unknown":
                self.sample_type = manifest_sample_type or "unknown"
            elif manifest_sample_type and self.sample_type != manifest_sample_type:
                raise RuntimeError(
                    f"Mixed sample types encountered across manifests: {self.sample_type} vs {manifest_sample_type} (from {manifest_path})"
                )

            languages_display = sorted(local_languages)
            if not languages_display and spec.language:
                languages_display = [spec.language]
            language_text = ", ".join(languages_display) if languages_display else "unspecified"
            print(
                f"[Info] Loaded {local_count} samples ({manifest_sample_type}) from {manifest_path} "
                f"(languages: {language_text})"
            )
            self.manifest_summaries.append(
                {"path": manifest_path, "count": local_count, "languages": languages_display}
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError("Dataset is empty.")

        if len(self.bad_indices) >= len(self.samples):
            raise RuntimeError("All samples were marked invalid; cannot continue.")

        attempts = 0
        max_attempts = len(self.samples)
        sample_count = len(self.samples)

        while attempts < max_attempts:
            current_idx = idx % sample_count

            sample = self.samples[current_idx]
            if sample is None:
                idx += 1
                attempts += 1
                continue

            try:
                text_ids = np.load(sample.text_ids_path, allow_pickle=False)
                codes = np.load(sample.codes_path, allow_pickle=False)
                condition = np.load(sample.condition_path, allow_pickle=False)
                emo_vec = np.load(sample.emo_vec_path, allow_pickle=False)

                if text_ids.size == 0 or codes.size == 0 or condition.size == 0 or emo_vec.size == 0:
                    raise ValueError("Encountered empty feature file.")

                text_ids = text_ids.astype(np.int64, copy=False)
                codes = codes.astype(np.int64, copy=False)
                condition = condition.astype(np.float32, copy=False)
                emo_vec = emo_vec.astype(np.float32, copy=False)

                return {
                    "id": sample.id,
                    "text_ids": torch.from_numpy(text_ids),
                    "codes": torch.from_numpy(codes),
                    "condition": torch.from_numpy(condition),  # [cond_len, dim]
                    "emo_vec": torch.from_numpy(emo_vec),
                    "text_len": torch.tensor(sample.text_len, dtype=torch.long),
                    "code_len": torch.tensor(sample.code_len, dtype=torch.long),
                    "condition_len": torch.tensor(sample.condition_len, dtype=torch.long),
                    "prompt_id": sample.prompt_id if sample.prompt_id else sample.id,
                    "target_id": sample.target_id if sample.target_id else sample.id,
                    "language": sample.language,
                    "prompt_language": sample.prompt_language,
                    "manifest_path": str(sample.manifest_path) if sample.manifest_path else "",
                    "speaker": sample.speaker,  # Stage 2: Speaker name
                }

            except (FileNotFoundError, OSError, ValueError, EOFError) as exc:
                if current_idx not in self.bad_indices:
                    message = (
                        f"[Warn] Skipping sample '{sample.id}' due to load failure: {exc}. "
                        "It will be removed from the dataset for this run."
                    )
                    print(message)
                    self.bad_indices.add(current_idx)

                self.samples[current_idx] = None
                if len(self.bad_indices) >= len(self.samples):
                    raise RuntimeError("All samples were marked invalid; cannot continue.")

                idx = current_idx + 1
                attempts += 1
                continue

        raise RuntimeError("Exceeded retry budget while sampling training data.")


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    text_tensors = [item["text_ids"] for item in batch]
    code_tensors = [item["codes"] for item in batch]
    condition_tensors = [item["condition"] for item in batch]
    emo_tensors = [item["emo_vec"] for item in batch]

    text_padded = pad_sequence(text_tensors, batch_first=True, padding_value=0)
    code_padded = pad_sequence(code_tensors, batch_first=True, padding_value=0)
    condition_stacked = torch.stack(condition_tensors, dim=0)
    emo_stacked = torch.stack(emo_tensors, dim=0)

    text_lengths = torch.stack([item["text_len"] for item in batch])
    code_lengths = torch.stack([item["code_len"] for item in batch])
    cond_lengths = torch.stack([item["condition_len"] for item in batch])

    ids = [item["id"] for item in batch]
    prompt_ids = [item.get("prompt_id", item["id"]) for item in batch]
    target_ids = [item.get("target_id", item["id"]) for item in batch]
    languages = [item.get("language") for item in batch]
    prompt_languages = [item.get("prompt_language") for item in batch]
    manifest_paths = [item.get("manifest_path") for item in batch]
    speakers = [item.get("speaker") for item in batch]  # Stage 2: Speaker names

    return {
        "ids": ids,
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "text_ids": text_padded,
        "codes": code_padded,
        "condition": condition_stacked,
        "emo_vec": emo_stacked,
        "text_lengths": text_lengths,
        "code_lengths": code_lengths,
        "condition_lengths": cond_lengths,
        "languages": languages,
        "prompt_languages": prompt_languages,
        "manifest_paths": manifest_paths,
        "speakers": speakers,  # Stage 2: Speaker names
    }


def load_tokenizer(tokenizer_path: Path) -> TextTokenizer:
    normalizer = TextNormalizer()
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    return tokenizer


def build_model(
    cfg_path: Path,
    tokenizer: TextTokenizer,
    base_checkpoint: Path,
    base_tokenizer_path: Path | None,
    device: torch.device,
    enable_grl: bool = False,
    num_speakers: int = 500,
    grl_lambda: float = 1.0,
    load_ckpt_to_device: bool = True,
) -> UnifiedVoice:
    cfg = OmegaConf.load(cfg_path)
    vocab_size = tokenizer.vocab_size
    if cfg.gpt.number_text_tokens != vocab_size:
        cfg.gpt.number_text_tokens = vocab_size

    # Stage 2: Add GRL parameters
    model = UnifiedVoice(
        **cfg.gpt,
        enable_grl=enable_grl,
        num_speakers=num_speakers,
        grl_lambda=grl_lambda
    )
    map_loc = device if load_ckpt_to_device else "cpu"
    checkpoint = torch.load(base_checkpoint, map_location=map_loc)
    raw_state_dict = checkpoint.get("model", checkpoint)

    filtered_state_dict = {}
    for key, value in raw_state_dict.items():
        if key.startswith("inference_model."):
            continue
        if ".lora_" in key:
            continue
        new_key = key.replace(".base_layer.", ".")
        if new_key == "gpt.wte.weight":
            continue
        filtered_state_dict[new_key] = value
    state_dict = filtered_state_dict

    base_vocab: Dict[str, int] = {}
    if base_tokenizer_path:
        if not base_tokenizer_path.exists():
            raise FileNotFoundError(f"Base tokenizer not found: {base_tokenizer_path}")
        base_tokenizer = TextTokenizer(str(base_tokenizer_path))
        base_vocab = base_tokenizer.get_vocab()

    new_vocab = tokenizer.get_vocab()

    def remap_by_token(
        key: str,
        module_weight: torch.Tensor,
        fallback_copy: bool = True,
    ) -> None:
        weight = state_dict.pop(key, None)
        if weight is None:
            return

        target = module_weight.detach().clone()
        copied = 0

        if base_vocab:
            if weight.ndim == 2 and target.ndim == 2:
                copy_cols = min(weight.size(1), target.size(1))
            else:
                copy_cols = None

            for token, new_idx in new_vocab.items():
                base_idx = base_vocab.get(token)
                if base_idx is None:
                    continue
                if base_idx >= weight.size(0) or new_idx >= target.size(0):
                    continue
                if target.ndim == 1:
                    target[new_idx] = weight[base_idx]
                elif copy_cols is not None:
                    target[new_idx, :copy_cols] = weight[base_idx, :copy_cols]
                else:
                    target[new_idx].copy_(weight[base_idx])
                copied += 1

            if copied:
                print(f"[Info] Remapped {key}: copied {copied} shared tokens.")

        if copied == 0 and fallback_copy:
            with torch.no_grad():
                slices = tuple(min(a, b) for a, b in zip(target.shape, weight.shape))
                if target.ndim == 1:
                    target[: slices[0]].copy_(weight[: slices[0]])
                else:
                    target[: slices[0], : slices[1]].copy_(weight[: slices[0], : slices[1]])

        state_dict[key] = target

    use_fallback = not base_vocab

    remap_by_token("text_embedding.weight", model.text_embedding.weight, fallback_copy=use_fallback)
    remap_by_token("text_head.weight", model.text_head.weight, fallback_copy=use_fallback)
    remap_by_token("text_head.bias", model.text_head.bias, fallback_copy=use_fallback)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] Missing keys during load: {missing}")
    if unexpected:
        print(f"[Warn] Unexpected keys during load: {unexpected}")

    return model.to(device)


def compute_losses(
    model: UnifiedVoice,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    speaker_to_id: Optional[Dict[str, int]] = None,
    speaker_loss_weight: float = 0.1,
    enable_stage2_realtime_emo: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
    condition = batch["condition"].to(device)
    text_ids = batch["text_ids"].to(device)
    codes = batch["codes"].to(device)
    emo_vec_precomputed = batch["emo_vec"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    code_lengths = batch["code_lengths"].to(device)
    condition_lengths = batch["condition_lengths"].to(device)

    batch_size = text_ids.size(0)
    use_speed = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Stage 2: Compute emo_vec in real-time from condition (mel-spectrogram)
    # This allows gradients to flow through emo encoder and enables proper GRL training
    if enable_stage2_realtime_emo and hasattr(model, 'enable_grl') and model.enable_grl:
        # condition: [batch, cond_len, 1024] -> transpose to [batch, 1024, cond_len]
        condition_transposed = condition.transpose(1, 2)

        # Pass through emo conditioning encoder to get emotion features
        emo_features = model.emo_conditioning_encoder(condition_transposed, condition_lengths)

        # Pass through emo perceiver to get final emo_vec: [batch, 1, output_dim]
        emo_vec_raw = model.emo_perceiver_encoder(emo_features)

        # Transform: [batch, 1, 1024] -> [batch, 1024] -> [batch, model_dim]
        emo_vec_syn_ori = emo_vec_raw.squeeze(1)
        emo_vec_syn = model.emovec_layer(emo_vec_syn_ori)
        emo_vec = model.emo_layer(emo_vec_syn)

        # Store raw emo_vec for GRL (before final transformation)
        emo_vec_for_grl = emo_vec
    else:
        # Stage 1: Use pre-computed emo_vec
        emo_vec = emo_vec_precomputed

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )

    # Duration conditioning:
    # - "binary" keeps legacy 0/1 speed_emb tokens (free-mode vs constrained)
    # - "length" follows IndexTTS2 paper: tie duration embedding to mel positional embedding
    duration_zero = model.speed_emb(torch.zeros_like(use_speed))
    if args.duration_conditioning == "length":
        # Clamp to valid range; +1 to account for stop token alignment
        length_idx = (code_lengths + 1).clamp_min(0).clamp_max(model.max_mel_tokens + 1)
        duration_one = model.mel_pos_embedding(length_idx)
    else:
        duration_one = model.speed_emb(torch.ones_like(use_speed))

    conds = torch.cat(
        (condition + emo_vec.unsqueeze(1), duration_one.unsqueeze(1), duration_zero.unsqueeze(1)),
        dim=1,
    )

    text_emb = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    mel_emb = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(conds, text_emb, model.text_head, mel_emb, model.mel_head)

    text_mask = (
        torch.arange(text_targets.size(1), device=device).unsqueeze(0)
        < (text_lengths + 1).unsqueeze(1)
    )
    mel_mask = (
        torch.arange(mel_targets.size(1), device=device).unsqueeze(0)
        < (code_lengths + 1).unsqueeze(1)
    )

    text_ce = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_ce = F.cross_entropy(mel_logits, mel_targets, reduction="none")

    text_loss = (text_ce * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_ce * mel_mask).sum() / mel_mask.sum().clamp_min(1)

    metrics = {}
    with torch.no_grad():
        mel_logits_flat = mel_logits.permute(0, 2, 1).reshape(-1, mel_logits.size(1))
        mel_targets_flat = mel_targets.reshape(-1)
        mel_mask_flat = mel_mask.reshape(-1)
        if mel_mask_flat.any():
            valid_logits = mel_logits_flat[mel_mask_flat]
            valid_targets = mel_targets_flat[mel_mask_flat]
            top1 = (valid_logits.argmax(dim=-1) == valid_targets).float().mean().item()
        else:
            top1 = 0.0
        metrics["mel_top1"] = top1

    # Stage 2: Speaker classification loss
    speaker_loss = None
    if speaker_to_id is not None and hasattr(model, 'enable_grl') and model.enable_grl:
        speakers = batch.get("speakers", [])
        speaker_labels = []
        for spk in speakers:
            if spk and spk in speaker_to_id:
                speaker_labels.append(speaker_to_id[spk])
            else:
                speaker_labels.append(-1)  # Ignore unknown speakers

        speaker_labels_tensor = torch.tensor(speaker_labels, dtype=torch.long, device=device)

        try:
            with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for speaker classifier
                # Use emo_vec computed in real-time (if enabled) or transform pre-computed one
                if enable_stage2_realtime_emo:
                    # Real-time mode: emo_vec_for_grl already computed above
                    # Gradients will flow through emo encoder -> proper adversarial training
                    emo_vec_to_reverse = emo_vec_for_grl
                else:
                    # Fallback mode: Transform pre-computed emo_vec
                    # This is a practical compromise when real-time computation is disabled
                    emo_vec_to_reverse = model.emo_layer(model.emovec_layer(emo_vec_precomputed))

                # Apply GRL: reverses gradients during backprop
                emo_vec_reversed = model.grl(emo_vec_to_reverse)

                # Classify speaker from reversed emo_vec
                speaker_logits = model.speaker_classifier(emo_vec_reversed)

            # Calculate speaker classification loss (ignore -1 labels)
            valid_mask = speaker_labels_tensor != -1
            if valid_mask.any():
                speaker_loss = F.cross_entropy(
                    speaker_logits[valid_mask],
                    speaker_labels_tensor[valid_mask],
                    reduction="mean"
                )
                metrics["speaker_loss"] = speaker_loss.item()
                metrics["speaker_acc"] = (
                    speaker_logits[valid_mask].argmax(dim=-1) == speaker_labels_tensor[valid_mask]
                ).float().mean().item()
            else:
                speaker_loss = torch.tensor(0.0, device=device)
        except Exception as e:
            import logging
            logging.warning(f"Failed to compute speaker loss: {e}")
            speaker_loss = None

    return text_loss, mel_loss, metrics, speaker_loss


def get_current_lr(optimizer: torch.optim.Optimizer, scheduler) -> float:
    """Return the current learning rate, even when no scheduler is used."""
    if scheduler is not None:
        return scheduler.get_last_lr()[0]
    # Fall back to first param group lr when running schedule-free
    return optimizer.param_groups[0].get("lr", 0.0)


def set_optimizer_train_mode(optimizer: torch.optim.Optimizer, is_train: bool) -> None:
    """Some optimizers (e.g., Schedule-Free) require explicit train/eval toggles."""
    if hasattr(optimizer, "train"):
        try:
            # AdamWScheduleFree.train(self) takes no arg beyond self
            optimizer.train()
        except TypeError:
            optimizer.train(is_train)


def normalize_state_dict_for_compile(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Align checkpoint keys with the current model (compiled vs. non-compiled).
    - If model expects _orig_mod.* but checkpoint has plain keys, add prefix.
    - If model expects plain keys but checkpoint has _orig_mod.*, strip prefix.
    """
    target_keys = model.state_dict().keys()
    target_has_prefix = any(k.startswith("_orig_mod.") for k in target_keys)
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if target_has_prefix and not ckpt_has_prefix:
        print("[Info] Compiled model detected; adding _orig_mod. prefix to checkpoint keys")
        return {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    if ckpt_has_prefix and not target_has_prefix:
        print("[Info] Non-compiled model detected; stripping _orig_mod. prefix from checkpoint keys")
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def get_effective_lr(optimizer: torch.optim.Optimizer, reported_lr: float) -> float:
    """
    For Prodigy, actual step size is scaled by internal state (d). For others, use reported lr.
    """
    if optimizer.__class__.__name__ == "Prodigy":
        d = getattr(optimizer, "d", None)
        if isinstance(d, torch.Tensor):
            return d.mean().item() * reported_lr
        if isinstance(d, (float, int)):
            return float(d) * reported_lr
    return reported_lr


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    recent_checkpoints: List[str],
    extra: Dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Detect optimizer type from class name
    optimizer_type = "prodigy" if optimizer.__class__.__name__ == "Prodigy" else "adamw"
    if optimizer.__class__.__name__ == "MARS":
        optimizer_type = "mars"
    elif optimizer.__class__.__name__ == "AdamWScheduleFree":
        optimizer_type = "schedulefree"

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_type": optimizer_type,  # Save optimizer type
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "recent_checkpoints": recent_checkpoints,
    }
    if extra:
        state["extra"] = extra
    torch.save(state, path)


def evaluate(model: UnifiedVoice, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            # Evaluation uses pre-computed emo_vec for speed
            text_loss, mel_loss, metrics, _ = compute_losses(
                model, batch, device, args, enable_stage2_realtime_emo=False
            )
            bsz = batch["text_ids"].size(0)
            totals["text_loss"] += text_loss.item() * bsz
            totals["mel_loss"] += mel_loss.item() * bsz
            totals["mel_top1"] += metrics["mel_top1"] * bsz
            count += bsz
    model.train()
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable Flash Attention 2 (PyTorch 2.0+ SDPA backend)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("[Info] Flash Attention 2 enabled via SDPA backend")
        print("[Info] Expected speedup: 2-4× faster attention, 50% less memory")

        # Additional GPU optimizations
        torch.backends.cudnn.benchmark = True  # Auto-select fastest cuDNN algorithm (5-10% faster)
        torch.set_float32_matmul_precision("high")  # Faster matmul with minor precision loss (20-30% faster)

        # TF32 optimization (Ampere+ GPUs: RTX 30xx/40xx, A100+)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print("[Info] cuDNN benchmark enabled (5-10% speedup)")
        print("[Info] TF32 enabled for matmul and cuDNN (20-30% speedup on RTX 4090)")
        print("[Info] Matrix multiplication precision set to 'high' (20-30% speedup)")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_root = output_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if os.environ.get("INDEXTTS_RUN_NAME") is None
        else os.environ["INDEXTTS_RUN_NAME"]
    )
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Initialize Aim if available and not disabled
    use_aim = AIM_AVAILABLE and not args.no_aim
    aim_run = None
    if use_aim:
        aim_run_name = args.aim_run_name or run_name
        # Create Aim Run
        try:
            # Try to resume the run
            aim_run = aim.Run(
                repo=args.aim_repo,
                experiment=args.aim_experiment,
                run_hash=aim_run_name,
            )
        except MissingRunError:
            # If the run doesn't exist, create a new one
            aim_run = aim.Run(
                repo=args.aim_repo,
                experiment=args.aim_experiment,
            )
            # Set the custom run name
            aim_run.name = aim_run_name
        # Track hyperparameters
        aim_run['hparams'] = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'grad_accumulation': args.grad_accumulation,
            'epochs': args.epochs,
            'warmup_steps': args.warmup_steps,
            'max_steps': args.max_steps,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'text_loss_weight': args.text_loss_weight,
            'mel_loss_weight': args.mel_loss_weight,
            'amp': args.amp,
            'seed': args.seed,
            'tokenizer': str(args.tokenizer),
            'base_checkpoint': str(args.base_checkpoint),
            'device': str(device),
            'output_dir': str(output_dir),
        }
        print(f"[Info] Aim initialized: experiment={args.aim_experiment}, run={aim_run_name}")
        print(f"[Info] View results: aim up --repo {args.aim_repo}")
    else:
        if not AIM_AVAILABLE:
            print("[Info] Aim not available. Install with: pip install aim")
        else:
            print("[Info] Aim disabled by --no-aim flag")

    # Stage 2: Load speaker mapping if provided
    speaker_to_id = None
    if args.enable_grl and args.speaker_mapping:
        if not args.speaker_mapping.exists():
            raise FileNotFoundError(f"Speaker mapping not found: {args.speaker_mapping}")
        with open(args.speaker_mapping, 'r', encoding='utf-8') as f:
            speaker_to_id = json.load(f)
        print(f"[Stage 2] Loaded speaker mapping: {len(speaker_to_id)} speakers")

    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(
        args.config,
        tokenizer,
        args.base_checkpoint,
        args.base_tokenizer,
        device,
        enable_grl=args.enable_grl,
        num_speakers=len(speaker_to_id) if speaker_to_id else 500,
        grl_lambda=args.grl_lambda,
        load_ckpt_to_device=not args.cpu_ckpt_load,
    )

    # Apply Liger Kernel Optimization (Before compilation)
    if LIGER_AVAILABLE:
        print("[Info] Applying Liger Kernel optimizations (Fused CrossEntropy, RMSNorm, etc.)")
        apply_liger_kernel_to_gpt2()

    # torch.compile with reduce-overhead mode for maximum speed
    print("[Info] Compiling model with torch.compile (reduce-overhead mode)...")
    model = torch.compile(model, mode="reduce-overhead")
    print("[Info] Model compilation complete - expect 15-30% speed boost")

    # DISABLED: Gradient Checkpointing (causing OOM during backward pass)
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     print("[Info] Enabling Gradient Checkpointing for memory efficiency")
    #     model.gradient_checkpointing_enable()
    # elif hasattr(model, "transformer") and hasattr(model.transformer, "gradient_checkpointing_enable"):
    #     print("[Info] Enabling Gradient Checkpointing for Transformer backbone")
    #     model.transformer.gradient_checkpointing_enable()

    # Stage 3: Freeze feature conditioners (speaker + emotion perceiver)
    if args.freeze_conditioners:
        print("[Stage 3] Freezing feature conditioners...")

        # Freeze speaker conditioning encoder and perceiver
        if hasattr(model, 'conditioning_encoder'):
            for param in model.conditioning_encoder.parameters():
                param.requires_grad = False
            print("  ✅ Speaker conditioning encoder frozen")

        if hasattr(model, 'perceiver_encoder'):
            for param in model.perceiver_encoder.parameters():
                param.requires_grad = False
            print("  ✅ Speaker perceiver encoder frozen")

        # Freeze emotion conditioning encoder and perceiver
        if hasattr(model, 'emo_conditioning_encoder'):
            for param in model.emo_conditioning_encoder.parameters():
                param.requires_grad = False
            print("  ✅ Emotion conditioning encoder frozen")

        if hasattr(model, 'emo_perceiver_encoder'):
            for param in model.emo_perceiver_encoder.parameters():
                param.requires_grad = False
            print("  ✅ Emotion perceiver encoder frozen")

        # Also freeze transformation layers
        if hasattr(model, 'emovec_layer'):
            for param in model.emovec_layer.parameters():
                param.requires_grad = False
            print("  ✅ Emovec layer frozen")

        if hasattr(model, 'emo_layer'):
            for param in model.emo_layer.parameters():
                param.requires_grad = False
            print("  ✅ Emo layer frozen")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[Stage 3] Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    train_specs = parse_manifest_specs(args.train_manifests, "--train-manifest")
    val_specs = parse_manifest_specs(args.val_manifests, "--val-manifest")

    print("[Info] Loading training manifests...")
    train_dataset = JapaneseGPTDataset(train_specs)
    print("[Info] Loading validation manifests...")
    val_dataset = JapaneseGPTDataset(val_specs)

    manifest_metadata = {
        "train": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in train_dataset.manifest_summaries
        ],
        "val": [
            {
                "path": str(entry["path"]),
                "count": entry["count"],
                "languages": list(entry["languages"]),
            }
            for entry in val_dataset.manifest_summaries
        ],
    }

    def checkpoint_extra(extra_type: str) -> Dict[str, object]:
        return {"type": extra_type, "manifests": manifest_metadata}

    use_cuda = torch.cuda.is_available()

    # DataLoader optimization (FFCV alternative for audio)
    dataloader_kwargs = {
        "persistent_workers": args.num_workers > 0,  # Reuse workers (20-30% faster)
        "prefetch_factor": 2,  # Prefetch 2 batches ahead
        "multiprocessing_context": "fork",  # Faster fork on Linux
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=use_cuda,
        **dataloader_kwargs,
    )

    # Optimizer selection
    if args.optimizer == "prodigy":
        print(f"[Info] Using Prodigy optimizer (parameter-free learning)")
        optimizer = Prodigy(
            model.parameters(),
            lr=1.0,
            betas=(0.9, 0.999),
            beta3=None,
            eps=1e-8,
            weight_decay=args.weight_decay,
            decouple=True,
            use_bias_correction=True,
            d_coef=1.0,
        )
    elif args.optimizer == "mars":
        print(f"[Info] Using MARS optimizer (MARS-AdamW) with LR={args.learning_rate}")
        optimizer = MARS(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            gamma=0.025,
            optimize_1d=True,
        )
    elif args.optimizer == "schedulefree":
        if not SCHEDULEFREE_AVAILABLE:
            raise RuntimeError("schedulefree optimizer requested but not installed. Run: pip install schedulefree")
        print(f"[Info] Using Schedule-Free AdamW optimizer with LR={args.learning_rate}")
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
    else:
        fused_ok = torch.cuda.is_available()
        print(f"[Info] Using AdamW optimizer with LR={args.learning_rate} fused={fused_ok}")
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=fused_ok,
        )

    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * max(1, len(train_loader)) // max(1, args.grad_accumulation)
    total_steps = max(total_steps, 1)

    scheduler = None
    if args.scheduler != "none":
        if args.optimizer == "schedulefree":
            print("[Info] Schedule-Free optimizer selected: disabling LR scheduler (constant LR after warmup).")
        elif args.scheduler == "wsd":
            # Calculate stable and decay steps from ratio
            remaining_steps = total_steps - args.warmup_steps
            num_stable_steps = int(remaining_steps * args.wsd_stable_ratio)
            num_decay_steps = remaining_steps - num_stable_steps

            print(f"[Info] Using WSD Scheduler")
            print(f"  - Warmup: {args.warmup_steps} steps")
            print(f"  - Stable: {num_stable_steps} steps ({args.wsd_stable_ratio*100:.0f}%)")
            print(f"  - Decay: {num_decay_steps} steps")
            print(f"  - Min LR Ratio: {args.wsd_min_lr_ratio}")

            scheduler = get_wsd_schedule(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                min_lr_ratio=args.wsd_min_lr_ratio,
            )
        else:
            print(f"[Info] Using Cosine Scheduler")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=total_steps,
            )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    global_step = 0
    start_epoch = 0
    recent_checkpoints: List[str] = []
    last_saved_step: int | None = None

    resume_path: str | None = None
    if args.resume:
        if args.resume == "auto":
            candidate = output_dir / "latest.pth"
            if candidate.exists():
                resume_path = str(candidate)
        else:
            resume_path = args.resume
    if resume_path:
        map_loc = device if not args.cpu_ckpt_load else "cpu"
        print(f"[Info] Loading checkpoint from {resume_path} to {map_loc} ...")
        checkpoint = torch.load(resume_path, map_location=map_loc)

        # Strip _orig_mod. prefix from torch.compile checkpoints
        state_dict = checkpoint["model"]
        state_dict = normalize_state_dict_for_compile(model, state_dict)

        print("[Info] Loading model state_dict...")
        model.load_state_dict(state_dict)
        print("[Info] Model state loaded successfully")

        # Load optimizer state only if optimizer type matches
        ckpt_optimizer_type = checkpoint.get("optimizer_type", "adamw")  # Default to adamw for old checkpoints
        if args.optimizer == ckpt_optimizer_type:
            print(f"[Info] Loading {args.optimizer} optimizer state (matching checkpoint)")
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print(f"[Info] Skipping optimizer state loading (checkpoint: {ckpt_optimizer_type}, current: {args.optimizer})")
            print(f"[Info] Starting with fresh {args.optimizer} optimizer state")

        if scheduler and checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        recent_checkpoints = checkpoint.get("recent_checkpoints", [])
        last_saved_step = checkpoint.get("step")
        print(f"[Info] Resumed from {resume_path} at epoch {start_epoch}, step {global_step}.")

    model.train()
    set_optimizer_train_mode(optimizer, True)
    optimizer.zero_grad(set_to_none=True)

    save_every = 1000
    best_val = math.inf

    # Training speed tracking
    import time
    step_start_time = time.time()
    last_log_step = global_step
    last_log_time = time.time()

    # Data loading time tracking
    data_load_time = 0.0
    compute_time = 0.0
    batch_start_time = time.time()

    if args.val_interval > 0 and global_step > 0:
        # If we resumed exactly on a validation boundary we postpone evaluation until
        # after the next training step to avoid running validation before training.
        print("[Info] Skipping startup validation; will evaluate after next training interval.")

    for epoch in range(start_epoch, args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            # Measure data loading time
            data_load_end = time.time()
            data_load_time = data_load_end - batch_start_time

            # Start compute timing
            compute_start = time.time()

            # Use new torch.amp.autocast (PyTorch 2.4+)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
                text_loss, mel_loss, metrics, speaker_loss = compute_losses(
                    model, batch, device, args, speaker_to_id, args.speaker_loss_weight,
                    enable_stage2_realtime_emo=args.enable_stage2_realtime_emo
                )
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
                # Stage 2: Add speaker classification loss
                if speaker_loss is not None:
                    loss = loss + args.speaker_loss_weight * speaker_loss

            if use_amp:
                scaler.scale(loss / args.grad_accumulation).backward()
            else:
                (loss / args.grad_accumulation).backward()

            if (batch_idx + 1) % args.grad_accumulation == 0:
                # Calculate gradient norm before clipping
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5  # pre-clip norm
                clipped_grad_norm = total_norm

                if args.grad_clip > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    clipped_grad_norm = min(total_norm, args.grad_clip)
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                # MARS: Update last gradient for variance reduction
                if hasattr(optimizer, "update_last_grad"):
                    optimizer.update_last_grad()

                # Measure compute time
                compute_time = time.time() - compute_start

                # Collect GPU metrics
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB (actually used)
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB (reserved by PyTorch)
                    # Note: torch.cuda.utilization() requires NVML, fallback to memory-based estimate
                    try:
                        gpu_utilization = torch.cuda.utilization()
                    except:
                        gpu_utilization = -1  # Not available
                else:
                    gpu_mem_allocated = 0
                    gpu_mem_reserved = 0
                    gpu_utilization = 0

                global_step += 1

                if global_step % args.log_interval == 0:
                    # Calculate training speed metrics
                    current_time = time.time()
                    steps_since_log = global_step - last_log_step
                    time_since_log = current_time - last_log_time

                    if time_since_log > 0 and steps_since_log > 0:
                        steps_per_min = (steps_since_log / time_since_log) * 60
                        time_per_step = time_since_log / steps_since_log
                        samples_per_sec = (steps_since_log * args.batch_size) / time_since_log
                        current_lr = get_current_lr(optimizer, scheduler)

                        # Calculate ETA
                        if args.max_steps > 0:
                            remaining_steps = args.max_steps - global_step
                        else:
                            total_steps = args.epochs * len(train_loader) // args.grad_accumulation
                            remaining_steps = total_steps - global_step
                        eta_seconds = remaining_steps * time_per_step
                        eta_hours = eta_seconds / 3600
                    else:
                        steps_per_min = 0
                        time_per_step = 0
                        samples_per_sec = 0
                        eta_hours = 0
                        current_lr = get_current_lr(optimizer, scheduler)

                    writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                    writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                    writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                    effective_lr = get_effective_lr(optimizer, current_lr)
                    writer.add_scalar("train/lr", current_lr, global_step)
                    writer.add_scalar("train/lr_effective", effective_lr, global_step)
                    writer.add_scalar("train/steps_per_min", steps_per_min, global_step)
                    writer.add_scalar("train/samples_per_sec", samples_per_sec, global_step)
                    writer.add_scalar("train/time_per_step", time_per_step, global_step)

                    # Resource utilization metrics
                    writer.add_scalar("train/gradient_norm_preclip", total_norm, global_step)
                    writer.add_scalar("train/gradient_norm_clipped", clipped_grad_norm, global_step)
                    writer.add_scalar("train/data_load_time_ms", data_load_time * 1000, global_step)
                    writer.add_scalar("train/compute_time_ms", compute_time * 1000, global_step)
                    writer.add_scalar("train/gpu_memory_allocated_gb", gpu_mem_allocated, global_step)  # Actual usage
                    writer.add_scalar("train/gpu_memory_reserved_gb", gpu_mem_reserved, global_step)  # Reserved (matches nvidia-smi)
                    if gpu_utilization >= 0:
                        writer.add_scalar("train/gpu_utilization_pct", gpu_utilization, global_step)

                    # Stage 2: Log speaker loss metrics
                    if speaker_loss is not None:
                        writer.add_scalar("train/speaker_loss", speaker_loss.item(), global_step)
                        if "speaker_acc" in metrics:
                            writer.add_scalar("train/speaker_acc", metrics["speaker_acc"], global_step)

                    # Aim logging
                    if use_aim:
                        aim_run.track(text_loss.item(), name='text_loss', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(mel_loss.item(), name='mel_loss', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(metrics["mel_top1"], name='mel_top1', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(current_lr, name='learning_rate', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(steps_per_min, name='steps_per_min', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(samples_per_sec, name='samples_per_sec', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)

                        # Resource utilization metrics
                        aim_run.track(total_norm, name='gradient_norm', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(data_load_time * 1000, name='data_load_time_ms', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(compute_time * 1000, name='compute_time_ms', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(gpu_mem_allocated, name='gpu_memory_gb', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                        if gpu_utilization >= 0:
                            aim_run.track(gpu_utilization, name='gpu_utilization_pct', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)

                        # Stage 2: Track speaker metrics
                        if speaker_loss is not None:
                            aim_run.track(speaker_loss.item(), name='speaker_loss', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)
                            if "speaker_acc" in metrics:
                                aim_run.track(metrics["speaker_acc"], name='speaker_acc', context={'subset': 'train'}, step=global_step, epoch=epoch + 1)

                    # Build log message
                    log_msg = (
                        f"[Train] epoch={epoch + 1} step={global_step} "
                        f"text_loss={text_loss.item():.4f} mel_loss={mel_loss.item():.4f} "
                        f"mel_top1={metrics['mel_top1']:.4f}"
                    )
                    if speaker_loss is not None:
                        log_msg += f" speaker_loss={speaker_loss.item():.4f}"
                        if "speaker_acc" in metrics:
                            log_msg += f" speaker_acc={metrics['speaker_acc']:.4f}"
                    log_msg += f" lr={current_lr:.2e}"
                    log_msg += f" | {steps_per_min:.1f}steps/min {samples_per_sec:.1f}samples/s {time_per_step:.2f}s/step ETA:{eta_hours:.1f}h"

                    # Add resource metrics
                    data_compute_ratio = data_load_time / (data_load_time + compute_time) if (data_load_time + compute_time) > 0 else 0
                    log_msg += (
                        f" | grad_norm={total_norm:.2f} (clipped {clipped_grad_norm:.2f}) "
                        f"lr_eff={effective_lr:.3e} GPU={gpu_mem_allocated:.1f}GB data={data_compute_ratio*100:.1f}%"
                    )

                    print(log_msg)

                    # Update tracking variables
                    last_log_step = global_step
                    last_log_time = current_time

                if args.val_interval > 0 and global_step > 0 and global_step % args.val_interval == 0:
                    val_start_time = time.time()
                    val_metrics = evaluate(model, val_loader, device, args)
                    val_time = time.time() - val_start_time
                    val_time_min = val_time / 60

                    writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
                    writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
                    writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
                    writer.add_scalar("val/time_minutes", val_time_min, global_step)

                    # Aim logging
                    if use_aim:
                        aim_run.track(val_metrics["text_loss"], name='text_loss', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(val_metrics["mel_loss"], name='mel_loss', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(val_metrics["mel_top1"], name='mel_top1', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                        aim_run.track(val_time_min, name='val_time_minutes', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)

                    print(
                        f"[Val] epoch={epoch + 1} step={global_step} "
                        f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                        f"mel_top1={val_metrics['mel_top1']:.4f} | val_time={val_time_min:.2f}min"
                    )
                    if val_metrics["mel_loss"] < best_val:
                        best_val = val_metrics["mel_loss"]
                        # Track best validation loss in Aim
                        if use_aim:
                            aim_run.track(best_val, name='best_val_mel_loss', context={'metric': 'best'})

                if global_step % save_every == 0:
                    ckpt_path = output_dir / f"model_step{global_step}.pth"
                    recent_checkpoints.append(str(ckpt_path))
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        recent_checkpoints,
                        extra=checkpoint_extra("step"),
                    )
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "optimizer_type": args.optimizer,  # Save optimizer type for resume
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "scaler": scaler.state_dict() if scaler else None,
                            "epoch": epoch,
                            "step": global_step,
                            "recent_checkpoints": recent_checkpoints,
                            "manifests": manifest_metadata,
                        },
                        output_dir / "latest.pth",
                    )

                    # Log checkpoint save to Aim
                    if use_aim:
                        aim_run.track(global_step, name='checkpoint_saved', context={'checkpoint': 'latest'})
                        print(f"[Info] Checkpoint saved at step {global_step}")

                    while len(recent_checkpoints) > 3:
                        obsolete = recent_checkpoints.pop(0)
                        try:
                            os.remove(obsolete)
                        except OSError:
                            pass
                    last_saved_step = global_step

                # Reset data loading timer for next batch
                batch_start_time = time.time()

                if args.max_steps and global_step >= args.max_steps:
                    break

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        if args.val_interval == 0:
            val_start_time = time.time()
            set_optimizer_train_mode(optimizer, False)
            val_metrics = evaluate(model, val_loader, device, args)
            set_optimizer_train_mode(optimizer, True)
            val_time = time.time() - val_start_time
            val_time_min = val_time / 60

            writer.add_scalar("val/text_loss", val_metrics["text_loss"], global_step)
            writer.add_scalar("val/mel_loss", val_metrics["mel_loss"], global_step)
            writer.add_scalar("val/mel_top1", val_metrics["mel_top1"], global_step)
            writer.add_scalar("val/time_minutes", val_time_min, global_step)

            # Aim logging for end-of-epoch validation
            if use_aim:
                aim_run.track(val_metrics["text_loss"], name='text_loss', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                aim_run.track(val_metrics["mel_loss"], name='mel_loss', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                aim_run.track(val_metrics["mel_top1"], name='mel_top1', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)
                aim_run.track(val_time_min, name='val_time_minutes', context={'subset': 'val'}, step=global_step, epoch=epoch + 1)

            print(
                f"[Val] epoch={epoch + 1} step={global_step} "
                f"text_loss={val_metrics['text_loss']:.4f} mel_loss={val_metrics['mel_loss']:.4f} "
                f"mel_top1={val_metrics['mel_top1']:.4f} | val_time={val_time_min:.2f}min"
            )
            if val_metrics["mel_loss"] < best_val:
                best_val = val_metrics["mel_loss"]
                # Track best validation loss in Aim
                if use_aim:
                    aim_run.track(best_val, name='best_val_mel_loss', context={'metric': 'best'})


    if global_step > 0 and last_saved_step != global_step:
        ckpt_path = output_dir / f"model_step{global_step}.pth"
        recent_checkpoints.append(str(ckpt_path))
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            recent_checkpoints,
            extra=checkpoint_extra("step-final"),
        )
        torch.save(
            {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_type": args.optimizer,  # Save optimizer type for resume
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "step": global_step,
            "recent_checkpoints": recent_checkpoints,
            "manifests": manifest_metadata,
            },
            output_dir / "latest.pth",
        )
        while len(recent_checkpoints) > 3:
            obsolete = recent_checkpoints.pop(0)
            try:
                os.remove(obsolete)
            except OSError:
                pass

    # Finalize Aim run
    if use_aim:
        aim_run.close()
        print("[Info] Aim run finalized")
        print(f"[Info] View results: aim up --repo {args.aim_repo}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
