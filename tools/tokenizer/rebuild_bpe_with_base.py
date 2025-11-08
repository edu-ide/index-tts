#!/usr/bin/env python3
"""
Rebuild a SentencePiece model by retaining the leading tokens from a base model
and filling the remainder with tokens learned from a new model.

This allows us to keep a subset of the original vocabulary (to preserve
pre-trained embeddings) while appending newly learned tokens for a different
language.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Set

try:  # pragma: no cover - import side-effect for SentencePiece protobuf builders
    from google.protobuf.internal import builder as _builder  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - shim for older protobuf
    from tools.tokenizer import protobuf_builder_compat as _builder  # type: ignore  # noqa: F401
    sys.modules["google.protobuf.internal.builder"] = _builder

from sentencepiece import sentencepiece_model_pb2 as sp_model  # noqa: E402


def load_model(path: Path) -> sp_model.ModelProto:
    proto = sp_model.ModelProto()
    proto.ParseFromString(path.read_bytes())
    return proto


def write_model(proto: sp_model.ModelProto, path: Path) -> None:
    path.write_bytes(proto.SerializeToString())


def write_vocab(proto: sp_model.ModelProto, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for piece in proto.pieces:
            handle.write(f"{piece.piece}\t{piece.score}\n")


def copy_piece(destination: sp_model.ModelProto, piece: sp_model.ModelProto.SentencePiece) -> None:
    new_piece = destination.pieces.add()
    new_piece.piece = piece.piece
    new_piece.score = piece.score
    new_piece.type = piece.type


def rebuild_vocab(
    base_proto: sp_model.ModelProto,
    new_proto: sp_model.ModelProto,
    retain_count: int,
    target_size: int,
) -> sp_model.ModelProto:
    if retain_count < 0:
        raise ValueError("retain_count must be non-negative.")
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    result = sp_model.ModelProto()
    result.CopyFrom(new_proto)
    result.ClearField("pieces")

    base_pieces = list(base_proto.pieces)
    new_pieces = list(new_proto.pieces)
    new_lookup: Dict[str, sp_model.ModelProto.SentencePiece] = {
        piece.piece: piece for piece in new_pieces
    }

    used: Set[str] = set()

    # Stage 1: keep the first `retain_count` tokens from the base model.
    retained = 0
    for base_piece in base_pieces:
        if retained >= retain_count:
            break
        text = base_piece.piece
        if text in used:
            continue
        source_piece = new_lookup.get(text, base_piece)
        copy_piece(result, source_piece)
        used.add(text)
        retained += 1

    if retained < retain_count:
        print(
            f"[rebuild_bpe] Warning: only retained {retained} tokens "
            f"out of requested {retain_count}. Consider lowering retain_count."
        )

    # Stage 2: append tokens from the new model.
    for piece in new_pieces:
        if len(result.pieces) >= target_size:
            break
        if piece.piece in used:
            continue
        copy_piece(result, piece)
        used.add(piece.piece)

    # Stage 3: if we still have room, fall back to remaining base tokens.
    if len(result.pieces) < target_size:
        for piece in base_pieces:
            if len(result.pieces) >= target_size:
                break
            if piece.piece in used:
                continue
            copy_piece(result, piece)
            used.add(piece.piece)

    if len(result.pieces) < target_size:
        raise RuntimeError(
            f"Unable to reach target size {target_size}. "
            f"Only {len(result.pieces)} pieces available."
        )

    result.trainer_spec.vocab_size = len(result.pieces)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild SentencePiece model with base retention.")
    parser.add_argument("--base-model", type=Path, required=True, help="Original SentencePiece model.")
    parser.add_argument("--new-model", type=Path, required=True, help="Newly trained SentencePiece model.")
    parser.add_argument("--output-model", type=Path, required=True, help="Output model path.")
    parser.add_argument("--output-vocab", type=Path, required=True, help="Output vocab path.")
    parser.add_argument(
        "--retain-count",
        type=int,
        default=2000,
        help="Number of leading tokens to keep from the base model (default: 2000).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=16000,
        help="Total vocabulary size for the rebuilt model (default: 16000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_proto = load_model(args.base_model)
    new_proto = load_model(args.new_model)

    rebuilt = rebuild_vocab(base_proto, new_proto, args.retain_count, args.target_size)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    write_model(rebuilt, args.output_model)
    write_vocab(rebuilt, args.output_vocab)
    print(
        f"[rebuild_bpe] Wrote rebuilt model ({len(rebuilt.pieces)} pieces) to {args.output_model}"
    )


if __name__ == "__main__":
    main()
