#!/usr/bin/env python3
"""Validate inputs and optionally run one historical SER training fold."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VARIANTS = (
    "noisy_gap", "noisy_att", "omss_gap", "omss_att",
    "fusion_gap", "fusion_att", "fusion_guided",
)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_integer_column(frame, column):
    values = pd.to_numeric(frame[column], errors="raise")
    require(
        np.isfinite(values.to_numpy(dtype=float)).all(),
        f"{column} contains nonfinite values.",
    )
    require(
        np.equal(values, np.floor(values)).all(),
        f"{column} contains noninteger values.",
    )
    return values.astype(int)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a historical grouped SER fold. "
            "Training runs only when --train is supplied."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--noisy-cache", type=Path, required=True)
    parser.add_argument("--omss-cache", type=Path, required=True)
    parser.add_argument("--name", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    require(
        args.frames > 0 and args.frames % 4 == 0,
        "--frames must be positive and divisible by four.",
    )
    require(args.epochs > 0, "--epochs must be positive.")
    require(args.patience >= 0, "--patience must be nonnegative.")
    require(args.seed >= 0, "--seed must be nonnegative.")

    full = pd.read_csv(args.manifest, dtype={"OriginalID": str})
    assignment = pd.read_csv(
        args.assignments, dtype={"OriginalID": str}
    )

    require(
        {"OriginalID", "FileName"}.issubset(full.columns),
        "Manifest requires OriginalID and FileName.",
    )
    require(
        {"OriginalID", "fold", "y"}.issubset(assignment.columns),
        "Assignments require OriginalID, fold, and y.",
    )
    require(
        not full[["OriginalID", "FileName"]].isna().any().any(),
        "Manifest contains missing IDs or filenames.",
    )
    require(
        not assignment[["OriginalID", "fold", "y"]].isna().any().any(),
        "Assignments contain missing values.",
    )
    require(
        not assignment["OriginalID"].duplicated().any(),
        "Assignments contain duplicate OriginalIDs.",
    )
    require(len(full) > 0, "Manifest is empty.")

    for column in ("fold", "y"):
        assignment[column] = read_integer_column(assignment, column)

    require(
        set(assignment["y"]) == {0, 1, 2, 3},
        "Assignments must contain exactly labels 0, 1, 2, 3.",
    )

    lookup = assignment.set_index("OriginalID")
    for column in ("fold", "y"):
        full[column] = full["OriginalID"].map(lookup[column])

    require(
        not full[["fold", "y"]].isna().any().any(),
        "Some manifest IDs lack assignments.",
    )
    full[["fold", "y"]] = full[["fold", "y"]].astype(int)
    full["row"] = np.arange(len(full))

    require(
        args.fold in set(full["fold"]),
        "Requested fold is absent from the manifest.",
    )

    XC = {}
    cache_paths = {
        "noisy": args.noisy_cache,
        "omss": args.omss_cache,
    }
    for condition, path in cache_paths.items():
        with np.load(path, allow_pickle=False) as cache:
            require(
                {"X", "filenames", "original_ids", "metadata_json"}
                .issubset(cache.files),
                f"{condition}: use a metadata-bearing extracted cache.",
            )
            require(
                np.array_equal(
                    cache["filenames"],
                    full["FileName"].astype(str).to_numpy(),
                ),
                f"{condition}: filename order mismatch.",
            )
            require(
                np.array_equal(
                    cache["original_ids"],
                    full["OriginalID"].to_numpy(),
                ),
                f"{condition}: OriginalID order mismatch.",
            )
            metadata = json.loads(str(cache["metadata_json"].item()))
            require(
                metadata["condition"] == condition
                and metadata["frames"] == args.frames,
                f"{condition}: cache metadata mismatch.",
            )
            X = cache["X"]
            require(
                X.shape == (len(full), args.frames, 40, 1),
                f"{condition}: unexpected feature shape {X.shape}.",
            )
            require(X.dtype == np.float32, f"{condition}: expected float32.")
            for start in range(0, len(X), 64):
                require(
                    np.isfinite(X[start:start + 64]).all(),
                    f"{condition}: nonfinite features.",
                )
            XC[condition] = X

    outer_train = full[full.fold != args.fold]
    test = full[full.fold == args.fold]
    groups = (
        outer_train.groupby("OriginalID")
        .agg(y=("y", "first"))
        .reset_index()
    )
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=42
    )
    _, indices = next(
        splitter.split(groups, groups.y, groups=groups.OriginalID)
    )
    validation_ids = set(groups.OriginalID.iloc[indices])
    validation = outer_train[
        outer_train.OriginalID.isin(validation_ids)
    ]
    train = outer_train[
        ~outer_train.OriginalID.isin(validation_ids)
    ]

    partitions = {
        "train": train, "validation": validation, "test": test
    }
    id_sets = {
        name: set(frame.OriginalID)
        for name, frame in partitions.items()
    }
    require(
        not (id_sets["train"] & id_sets["validation"])
        and not (id_sets["train"] & id_sets["test"])
        and not (id_sets["validation"] & id_sets["test"]),
        "OriginalID leakage between partitions.",
    )
    require(
        set.union(*id_sets.values()) == set(full.OriginalID),
        "Partitions do not cover all manifest IDs.",
    )

    split_summary = {}
    for name, frame in partitions.items():
        require(len(frame) > 0, f"Empty {name} partition.")
        counts = frame.y.value_counts().reindex(range(4), fill_value=0)
        split_summary[name] = {
            "rows": len(frame),
            "original_ids": frame.OriginalID.nunique(),
            "class_counts": {
                str(label): int(counts[label]) for label in range(4)
            },
        }
        missing = counts[counts == 0].index.tolist()
        if missing:
            warnings.warn(
                f"{name} lacks labels {missing}. Historical macro "
                "recall includes all four labels with zero_division=0."
            )

    print("=== VALIDATED SPLIT ===")
    print(json.dumps(split_summary, indent=2))
    print("Input alignment and OriginalID separation: PASS")

    if not args.train:
        print("VALIDATION ONLY: no model built, no training, no files written.")
        return

    require(args.out_dir is not None, "--train requires --out-dir.")
    require(
        not args.out_dir.exists(),
        "Training requires a new output directory to avoid stale-run skips.",
    )

    if __package__:
        from .training_core import run_fold
    else:
        from training_core import run_fold

    args.out_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["split_summary"] = split_summary
    configuration["wrapper_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    (args.out_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_fold(
        args.name, args.fold, args.seed, args.frames,
        full, XC,
        epochs=args.epochs,
        patience=args.patience,
        out_dir=str(args.out_dir),
    )
    (args.out_dir / "run_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
