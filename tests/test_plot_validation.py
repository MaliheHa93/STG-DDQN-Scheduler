from __future__ import annotations

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from stg_ddqn.plot_results import PANELS, _draw_panel, validate_curve_data


ALGORITHMS = ("stg_ddqn", "im", "dlapm")


def summary_frame(flat: bool) -> pd.DataFrame:
    rows = []
    for panel_index, (experiment, metric, *_rest) in enumerate(PANELS):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for point in range(4):
                if flat:
                    mean = 100.0 if metric.endswith("_pct") else 0.0
                else:
                    mean = 80.0 - 3.0 * point + algorithm_index
                    if metric == "migration_mb_per_admitted_sfc":
                        mean = 5.0 + 4.0 * point + 2.0 * algorithm_index
                    elif metric == "p95_decision_runtime_ms":
                        mean = 2.0 + point + 0.5 * algorithm_index
                rows.append({
                    "experiment": experiment,
                    "x_value": point,
                    "algorithm": algorithm,
                    "metric": metric,
                    "n": 3,
                    "mean": mean,
                    "ci95_low": mean - 0.5,
                    "ci95_high": mean + 0.5,
                })
    return pd.DataFrame(rows)


class CurveValidationTests(unittest.TestCase):
    def test_accepts_valid_flat_overlapping_zero_migration_results(self):
        with self.assertWarns(UserWarning):
            validate_curve_data(summary_frame(flat=True))

    def test_accepts_complete_nontrivial_paired_curve_grid(self):
        validate_curve_data(summary_frame(flat=False))

    def test_panel_contains_curves_without_shaded_collections(self):
        frame = summary_frame(flat=False)
        figure, axis = plt.subplots()
        try:
            _draw_panel(axis, frame, PANELS[0])
            self.assertEqual(len(axis.lines), len(ALGORITHMS))
            self.assertEqual(len(axis.collections), 0)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
