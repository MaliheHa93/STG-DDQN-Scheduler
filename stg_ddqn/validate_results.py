from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate STG-DDQN result accounting")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--strict-curves", action="store_true")
    parser.add_argument("--minimum-points", type=int, default=4)
    parser.add_argument("--expected-algorithms", default="stg_ddqn,im,dlapm")
    args = parser.parse_args()
    raw = pd.read_csv(args.raw)
    summary = pd.read_csv(args.summary)
    validate_raw(raw)
    validate_summary(summary)
    if args.strict_curves:
        from .plot_results import validate_curve_data

        validate_curve_data(
            summary,
            expected_algorithms=tuple(
                value.strip()
                for value in args.expected_algorithms.split(",")
                if value.strip()
            ),
            minimum_points=int(args.minimum_points),
            allow_degenerate=False,
        )
    print(
        f"validated runs={len(raw)} summary_rows={len(summary)} "
        f"algorithms={raw['algorithm'].nunique()}"
    )


def validate_raw(raw: pd.DataFrame) -> None:
    required = {
        "algorithm", "seed", "experiment", "x_value",
        "checkpoint_sha256",
        "generated_requests", "admitted_requests",
        "continuity_resolved_requests", "continuity_satisfied_requests",
        "acceptance_ratio_pct", "continuity_satisfaction_ratio_pct",
        "migration_volume_total_mb", "migration_mb_per_admitted_sfc",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw results are missing columns: {sorted(missing)}")
    run_key = ["experiment", "x_value", "algorithm", "seed"]
    if raw.duplicated(run_key).any():
        raise ValueError("Raw results contain duplicate algorithm/scenario/seed runs")
    checkpoint_hashes = {
        str(value)
        for value in raw["checkpoint_sha256"].dropna()
        if str(value) != "not_applicable"
    }
    if raw["algorithm"].isin(("stg_ddqn", "graph_ddqn")).any() and (
        len(checkpoint_hashes) != 1
    ):
        raise ValueError(
            "Learning-policy runs must share exactly one frozen checkpoint fingerprint"
        )
    if (raw["admitted_requests"] > raw["generated_requests"]).any():
        raise ValueError("Admitted requests exceed generated requests")
    if (raw["continuity_satisfied_requests"] > raw["continuity_resolved_requests"]).any():
        raise ValueError("Satisfied requests exceed the resolved continuity denominator")
    expected_acceptance = np.where(
        raw["generated_requests"] > 0,
        100.0 * raw["admitted_requests"] / raw["generated_requests"],
        np.nan,
    )
    _assert_close(raw["acceptance_ratio_pct"], expected_acceptance, "acceptance ratio")
    expected_continuity = np.where(
        raw["continuity_resolved_requests"] > 0,
        100.0 * raw["continuity_satisfied_requests"] / raw["continuity_resolved_requests"],
        np.nan,
    )
    _assert_close(
        raw["continuity_satisfaction_ratio_pct"], expected_continuity,
        "continuity ratio",
    )
    expected_migration = np.where(
        raw["admitted_requests"] > 0,
        raw["migration_volume_total_mb"] / raw["admitted_requests"],
        np.nan,
    )
    _assert_close(
        raw["migration_mb_per_admitted_sfc"], expected_migration,
        "migration volume per admitted SFC",
    )
    grouping = ["experiment", "x_value"]
    for _, group in raw.groupby(grouping, dropna=False):
        seed_sets = [set(values) for _, values in group.groupby("algorithm")["seed"]]
        if seed_sets and any(values != seed_sets[0] for values in seed_sets[1:]):
            raise ValueError("Algorithms do not use identical paired seeds")


def validate_summary(summary: pd.DataFrame) -> None:
    required = {"mean", "ci95_low", "ci95_high", "n", "metric"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    finite = summary[["mean", "ci95_low", "ci95_high"]].notna().all(axis=1)
    if (summary.loc[finite, "ci95_low"] > summary.loc[finite, "mean"]).any():
        raise ValueError("A CI lower bound exceeds its mean")
    if (summary.loc[finite, "mean"] > summary.loc[finite, "ci95_high"]).any():
        raise ValueError("A CI upper bound is below its mean")
    if (summary["n"] < 1).any():
        raise ValueError("Every summary row must contain at least one realization")


def _assert_close(actual, expected, label: str) -> None:
    actual_values = np.asarray(actual, dtype=float)
    expected_values = np.asarray(expected, dtype=float)
    finite = np.isfinite(actual_values) & np.isfinite(expected_values)
    if not np.allclose(actual_values[finite], expected_values[finite], rtol=1e-9, atol=1e-9):
        raise ValueError(f"Incorrect {label} accounting")
    nan_mismatch = np.isnan(actual_values) ^ np.isnan(expected_values)
    if nan_mismatch.any():
        raise ValueError(f"Incorrect {label} NaN handling")


if __name__ == "__main__":
    main()
