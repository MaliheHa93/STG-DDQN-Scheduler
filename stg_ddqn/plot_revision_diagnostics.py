"""Python preview of the MATLAB diagnostics; never labeled as MATLAB output."""
from __future__ import annotations

import argparse
from pathlib import Path

from .plot_results import _configure_style, _save_figure, STYLE
import matplotlib.pyplot as plt
import pandas as pd


PANELS = [
    ('mobility', 'failures_per_service_hour', 'Maximum vehicle speed (m/s)', 'Failures per service-hour'),
    ('traffic', 'peak_temporary_cpu', 'Arrival rate (requests/s)', 'Peak temporary CPU (model units)'),
    ('traffic', 'temporary_cpu_seconds', 'Arrival rate (requests/s)', 'Temporary CPU reservation (unit s)'),
    ('mobility', 'canceled_transfer_total_mb', 'Maximum vehicle speed (m/s)', 'Canceled transfer volume (MB)'),
    ('scalability', 'mean_candidate_runtime_ms', 'Number of fog RSUs', 'Mean candidate construction time (ms)'),
    ('scalability', 'complete_fallback_count', 'Number of fog RSUs', 'Fallback decisions'),
    ('traffic', 'search_budget_blocked_requests', 'Arrival rate (requests/s)', 'Admissions lost to search budget'),
    ('mobility', 'search_budget_continuity_failures', 'Maximum vehicle speed (m/s)', 'Continuity failures from search budget'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evidence-label', required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _configure_style()
    figure, axes = plt.subplots(4, 2, figsize=(13.8, 15), constrained_layout=True)
    for axis, (experiment, metric, xlabel, ylabel) in zip(axes.flat, PANELS):
        subset = frame[(frame.experiment == experiment) & (frame.metric == metric)]
        if subset.empty:
            raise ValueError(f'Missing diagnostic metric: {experiment}/{metric}')
        for name in ('stg_ddqn', 'im', 'dlapm'):
            rows = subset[subset.algorithm == name].sort_values('x_value')
            if len(rows) == 0:
                raise ValueError(f'Missing diagnostic algorithm: {name}')
            style = STYLE[name]
            axis.errorbar(rows.x_value, rows['mean'],
                yerr=[rows['mean']-rows.ci95_low, rows.ci95_high-rows['mean']],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                label=style['label'], capsize=3)
        axis.set(xlabel=xlabel, ylabel=ylabel, xticks=sorted(subset.x_value.unique()))
        axis.grid(True)
        axis.legend(fontsize=9)
    figure.suptitle(args.evidence_label, fontsize=12)
    _save_figure(figure, output / 'revision_diagnostics_python_preview.pdf')
    _save_figure(figure, output / 'revision_diagnostics_python_preview.png', dpi=200)
    plt.close(figure)


if __name__ == '__main__':
    main()
