from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .stats import write_results


RUN_KEY = ("experiment", "x_value", "algorithm", "seed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge non-overlapping, partitioned STG-DDQN experiment outputs"
    )
    parser.add_argument("--inputs", required=True, help="Comma-separated result directories")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--experiments",
        default=None,
        help="Optional comma-separated experiment filter",
    )
    args = parser.parse_args()
    directories = [Path(value.strip()) for value in args.inputs.split(",") if value.strip()]
    raw_frames: list[pd.DataFrame] = []
    request_frames: list[pd.DataFrame] = []
    for directory in directories:
        raw_path = directory / "evaluation_raw.csv"
        request_path = directory / "request_records.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_frames.append(pd.read_csv(raw_path))
        if request_path.exists() and request_path.stat().st_size:
            request_frames.append(pd.read_csv(request_path))
    raw = pd.concat(raw_frames, ignore_index=True)
    if 'schema_version' in raw and raw['schema_version'].nunique(dropna=False) != 1:
        raise ValueError('Cannot merge results with different accounting schemas')
    if args.experiments:
        selected = {
            value.strip() for value in args.experiments.split(",") if value.strip()
        }
        raw = raw[raw["experiment"].isin(selected)].copy()
        request_frames = [
            frame[frame["experiment"].isin(selected)].copy()
            for frame in request_frames
        ]
        if raw.empty:
            raise ValueError(f"No runs matched experiments {sorted(selected)}")
    missing = set(RUN_KEY) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw result partitions lack run keys: {sorted(missing)}")
    if raw.duplicated(list(RUN_KEY)).any():
        duplicates = raw.loc[raw.duplicated(list(RUN_KEY), keep=False), list(RUN_KEY)]
        raise ValueError(f"Duplicate partitioned runs:\n{duplicates.to_string(index=False)}")
    hashes = {
        str(value) for value in raw.get("checkpoint_sha256", pd.Series(dtype=str)).dropna()
        if str(value) != "not_applicable"
    }
    if len(hashes) > 1:
        raise ValueError("Partitions were evaluated with different STG-DDQN checkpoints")
    requests = (
        pd.concat(request_frames, ignore_index=True).to_dict("records")
        if request_frames else []
    )
    paths = write_results(raw.to_dict("records"), requests, args.output)
    print("\n".join(str(path.resolve()) for path in paths))


if __name__ == "__main__":
    main()
