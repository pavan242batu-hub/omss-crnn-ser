#!/usr/bin/env python3
"""Extract noisy or OMSS MFCC features from a supplied manifest.

Preserves the recovered mfcc_long processing procedure.
This script does not perform OMSS enhancement or train a classifier.
"""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=["noisy", "omss"], required=True
    )
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract_mfcc(path, frames):
    y, _ = librosa.load(str(path), sr=16000, mono=True)

    m = librosa.feature.mfcc(
        y=y,
        sr=16000,
        n_mfcc=40,
        n_fft=512,
        win_length=512,
        hop_length=160,
        n_mels=64,
        center=True,
    ).T

    if not len(m) or not np.isfinite(m).all():
        raise ValueError(f"Invalid extracted features: {path}")

    m = (
        (m - m.mean(0, keepdims=True))
        / (m.std(0, keepdims=True) + 1e-8)
    )

    if len(m) < frames:
        m = np.pad(m, ((0, frames - len(m)), (0, 0)))
    else:
        m = m[:frames]

    return m.astype(np.float32)[..., None], len(y) / 16000


def index_audio(root):
    index = {}

    for path in sorted(root.rglob("*.wav")):
        if path.name in index:
            raise ValueError(
                f"Duplicate audio filename: {path.name}\n"
                f"{index[path.name]}\n{path}"
            )
        index[path.name] = path

    return index


def main():
    args = parse_args()

    if args.frames < 1:
        raise ValueError("--frames must be positive.")

    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    if not args.audio_root.is_dir():
        raise FileNotFoundError(args.audio_root)

    if args.output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; nothing overwritten: {args.output}"
        )

    manifest = pd.read_csv(args.manifest)
    required = ["FileName", "OriginalID"]

    if not set(required).issubset(manifest.columns):
        raise ValueError(
            f"Manifest must contain these columns: {required}"
        )

    if manifest.empty or manifest[required].isna().any().any():
        raise ValueError("Manifest is empty or contains missing identifiers.")

    filenames = manifest["FileName"].astype(str).to_numpy()
    original_ids = manifest["OriginalID"].astype(str).to_numpy()

    if len(set(filenames)) != len(filenames):
        raise ValueError("Manifest contains duplicate filenames.")

    audio_index = index_audio(args.audio_root)
    missing = [name for name in filenames if name not in audio_index]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files missing. First examples: "
            + ", ".join(missing[:5])
        )

    X = np.empty(
        (len(manifest), args.frames, 40, 1), dtype=np.float32
    )
    durations = np.empty(len(manifest), dtype=np.float64)

    for row, name in enumerate(filenames):
        X[row], durations[row] = extract_mfcc(
            audio_index[name], args.frames
        )

        if (row + 1) % 100 == 0 or row + 1 == len(manifest):
            print(f"Extracted {row + 1}/{len(manifest)}", flush=True)

    info = {
        "condition": args.condition,
        "frames": args.frames,
        "sample_rate": 16000,
        "n_mfcc": 40,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "center": True,
        "normalization": "per-recording coefficient mean/std before length adjustment",
        "epsilon": 1e-8,
        "length_adjustment": "end zero-padding or first-frame truncation",
        "unspecified_options": "library defaults in the recorded environment",
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "librosa", "soundfile", "scipy"]
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting an existing cache.
    with args.output.open("xb") as handle:
        np.savez(
            handle,
            X=X,
            dur=durations,
            filenames=filenames.astype(str),
            original_ids=original_ids.astype(str),
            metadata_json=np.array(json.dumps(info, sort_keys=True)),
        )

    print(f"Saved: {args.output}")
    print(f"Feature shape: {X.shape}")
    print("Rows follow the supplied manifest order.")


if __name__ == "__main__":
    main()
