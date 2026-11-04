from __future__ import annotations

import unittest

import pandas as pd

from stg_ddqn.validate_results import validate_raw


def result_row(algorithm: str, seed: int, checkpoint_hash: str) -> dict:
    return {
        "algorithm": algorithm,
        "seed": seed,
        "experiment": "traffic",
        "x_value": 0.5,
        "checkpoint_sha256": checkpoint_hash,
        "generated_requests": 10,
        "admitted_requests": 8,
        "continuity_resolved_requests": 7,
        "continuity_satisfied_requests": 6,
        "acceptance_ratio_pct": 80.0,
        "continuity_satisfaction_ratio_pct": 100.0 * 6.0 / 7.0,
        "migration_volume_total_mb": 16.0,
        "migration_mb_per_admitted_sfc": 2.0,
    }


class RawResultValidationTests(unittest.TestCase):
    def test_rejects_duplicate_run_keys(self):
        row = result_row("stg_ddqn", 31, "hash-a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_raw(pd.DataFrame((row, dict(row))))

    def test_rejects_mixed_learning_checkpoints(self):
        rows = (
            result_row("stg_ddqn", 31, "hash-a"),
            result_row("stg_ddqn", 47, "hash-b"),
        )
        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            validate_raw(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
