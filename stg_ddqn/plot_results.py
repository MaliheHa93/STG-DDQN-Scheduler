from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stg-ddqn-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {
    "stg_ddqn": {"label": "STG-DDQN", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "graph_ddqn": {"label": "Graph-DDQN", "color": "#D55E00", "marker": "s", "linestyle": "-"},
    "im": {"label": "IM", "color": "#56B4E9", "marker": "^", "linestyle": "--"},
    "dlapm": {"label": "DLAPM", "color": "#E69F00", "marker": "D", "linestyle": "--"},
    "delay_greedy": {"label": "Delay-Greedy", "color": "#009E73", "marker": "v", "linestyle": ":"},
}

PANELS = (
    ("traffic", "acceptance_ratio_pct", "Arrival rate $\\lambda$ (requests/s)", "Request acceptance ratio (%)", "lower left"),
    ("mobility", "continuity_satisfaction_ratio_pct", "Maximum vehicle speed (m/s)", "SFC continuity satisfaction (%)", "lower left"),
    ("mobility", "migration_mb_per_admitted_sfc", "Maximum vehicle speed (m/s)", "Migration volume per admitted SFC (MB)", "upper left"),
    ("scalability", "p95_decision_runtime_ms", "Number of fog RSUs", "95th-percentile decision runtime (ms)", "upper left"),
)
DEFAULT_ALGORITHMS = ("stg_ddqn", "im", "dlapm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw paper figures in the supplied sample style")
    parser.add_argument("--input", default="results/paper/evaluation_summary.csv")
    parser.add_argument("--output", default="plots/paper")
    parser.add_argument(
        "--basename",
        default="stg_ddqn_paper_panels",
        help="Base filename for the combined PDF/PNG export",
    )
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--show-ci", action="store_true")
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms required in every panel",
    )
    parser.add_argument("--minimum-points", type=int, default=4)
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Permit smoke-test grids; never use this for paper figures",
    )
    parser.add_argument("--evidence-label", default="")
    args = parser.parse_args()
    summary = pd.read_csv(args.input)
    expected_algorithms = tuple(
        value.strip() for value in args.algorithms.split(",") if value.strip()
    )
    validate_curve_data(
        summary,
        expected_algorithms=expected_algorithms,
        minimum_points=int(args.minimum_points),
        allow_degenerate=bool(args.allow_degenerate),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _configure_style()
    figure, axes = plt.subplots(
        2, 2, figsize=(13.8, 8.6), constrained_layout=True, facecolor="white"
    )
    for index, (axis, panel) in enumerate(zip(axes.flat, PANELS)):
        _draw_panel(axis, summary, panel, show_ci=args.show_ci)
    if args.evidence_label:
        figure.suptitle(args.evidence_label, fontsize=12)
    _save_figure(figure, output / f"{args.basename}.pdf")
    _save_figure(figure, output / f"{args.basename}.png", dpi=600)
    plt.close(figure)
    if args.separate:
        for index, panel in enumerate(PANELS):
            figure, axis = plt.subplots(
                figsize=(7.2, 5.0), constrained_layout=True, facecolor="white"
            )
            _draw_panel(axis, summary, panel, show_ci=args.show_ci)
            if args.evidence_label:
                figure.suptitle(args.evidence_label, fontsize=10)
            _save_figure(figure, output / f"figure_{chr(97 + index)}.pdf")
            _save_figure(
                figure,
                output / f"figure_{chr(97 + index)}.png",
                dpi=600,
            )
            plt.close(figure)


def _save_figure(figure, path: Path, dpi: int | None = None) -> None:
    """Write a complete figure before atomically publishing its final name."""

    temporary = path.with_name(f".{path.stem}.writing{path.suffix}")
    figure.savefig(
        temporary,
        format=path.suffix.lstrip("."),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    os.replace(temporary, path)


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.linewidth": 1.4,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "grid.color": "#BDBDBD",
        "grid.alpha": 0.32,
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def _draw_panel(axis, summary: pd.DataFrame, panel, show_ci: bool = False) -> None:
    experiment, metric, xlabel, ylabel, legend_location = panel
    subset = summary[(summary["experiment"] == experiment) & (summary["metric"] == metric)]
    if subset.empty:
        axis.text(0.5, 0.5, f"No {experiment}/{metric} data", ha="center", va="center")
        axis.set_axis_off()
        return
    present = set(subset["algorithm"].astype(str))
    algorithms = [name for name in STYLE if name in present]
    algorithms.extend(sorted(present - set(algorithms)))
    for algorithm in algorithms:
        values = subset[subset["algorithm"] == algorithm].sort_values("x_value")
        style = STYLE.get(algorithm, {
            "label": algorithm.replace("_", " ").title(),
            "color": None,
            "marker": "o",
            "linestyle": "-",
        })
        x = values["x_value"].to_numpy(dtype=float)
        mean = values["mean"].to_numpy(dtype=float)
        axis.plot(
            x, mean,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markeredgewidth=0.7,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    x_values = sorted(subset["x_value"].astype(float).unique())
    axis.set_xticks(x_values)
    axis.set_xticklabels([
        str(int(value)) if float(value).is_integer() else f"{value:g}"
        for value in x_values
    ])
    axis.grid(True, which="major")
    axis.margins(x=0.03)
    _set_y_limits(axis, metric, subset)
    if show_ci:
        for algorithm in algorithms:
            rows = subset[subset["algorithm"] == algorithm].sort_values("x_value")
            style = STYLE.get(algorithm, {})
            axis.errorbar(rows["x_value"], rows["mean"],
                          yerr=[rows["mean"] - rows["ci95_low"], rows["ci95_high"] - rows["mean"]],
                          fmt="none", ecolor=style.get("color", "gray"), capsize=3, alpha=0.65)
        lo, hi = axis.get_ylim()
        axis.set_ylim(min(lo, float(subset["ci95_low"].min())),
                      max(hi, float(subset["ci95_high"].max())))

    legend = axis.legend(
        loc=legend_location,
        ncol=len(algorithms),
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=1.0,
        borderpad=0.45,
        handlelength=2.3,
        columnspacing=1.0,
    )
    legend.get_frame().set_edgecolor("#C7C7C7")
    legend.get_frame().set_linewidth(1.0)


def _set_y_limits(axis, metric: str, subset: pd.DataFrame) -> None:
    means = subset["mean"].to_numpy(dtype=float)
    finite_means = means[np.isfinite(means)]
    if not len(finite_means):
        return
    minimum = float(np.min(finite_means))
    maximum = float(np.max(finite_means))
    if metric.endswith("_pct"):
        span = max(5.0, maximum - minimum)
        lower = max(0.0, minimum - 0.18 * span)
        upper = min(102.0, max(100.5, maximum + 0.12 * span))
        axis.set_ylim(lower, upper)
    else:
        upper = maximum * 1.16 if maximum > 0 else 1.0
        axis.set_ylim(0.0, upper)


def validate_curve_data(
    summary: pd.DataFrame,
    expected_algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS,
    minimum_points: int = 4,
    allow_degenerate: bool = False,
) -> None:
    required = {
        "experiment", "x_value", "algorithm", "metric", "n",
        "mean", "ci95_low", "ci95_high",
    }
    missing_columns = required - set(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary is missing columns: {sorted(missing_columns)}")
    problems: list[str] = []
    for experiment, metric, *_ in PANELS:
        subset = summary[
            (summary["experiment"] == experiment)
            & (summary["metric"] == metric)
        ]
        if subset.empty:
            problems.append(f"missing {experiment}/{metric}")
            continue
        present = set(subset["algorithm"].astype(str))
        missing_algorithms = set(expected_algorithms) - present
        if missing_algorithms:
            problems.append(
                f"{experiment}/{metric} lacks {sorted(missing_algorithms)}"
            )
        x_sets = []
        for algorithm in expected_algorithms:
            values = subset[subset["algorithm"] == algorithm]
            if values.empty:
                continue
            x_values = set(values["x_value"].astype(float))
            if values.duplicated(["x_value"]).any():
                problems.append(f"{experiment}/{metric}/{algorithm} has duplicate x-values")
            x_sets.append(x_values)
            if len(x_values) < minimum_points:
                problems.append(
                    f"{experiment}/{metric}/{algorithm} has only "
                    f"{len(x_values)} x-values; need at least {minimum_points}"
                )
            if not allow_degenerate and int(values["n"].min()) < 2:
                problems.append(
                    f"{experiment}/{metric}/{algorithm} has fewer than two seeds"
                )
        if x_sets and any(values != x_sets[0] for values in x_sets[1:]):
            problems.append(f"{experiment}/{metric} has an incomplete algorithm grid")
        if not np.isfinite(subset[["mean", "ci95_low", "ci95_high"]].to_numpy()).all():
            problems.append(f"{experiment}/{metric} contains nonfinite plot values")
        if (subset["ci95_low"] > subset["mean"]).any() or (
            subset["mean"] > subset["ci95_high"]
        ).any():
            problems.append(f"{experiment}/{metric} has invalid confidence bounds")
        if not allow_degenerate:
            pivot = subset.pivot_table(
                index="x_value", columns="algorithm", values="mean", aggfunc="first"
            )
            within_curve = max(
                (float(pivot[column].max() - pivot[column].min())
                 for column in pivot.columns),
                default=0.0,
            )
            between_curves = max(
                (float(row.max() - row.min()) for _, row in pivot.iterrows()),
                default=0.0,
            )
            tolerance = 1e-6
            if max(within_curve, between_curves) <= tolerance:
                warnings.warn(f"{experiment}/{metric} is flat and fully overlapping; inspect inputs", stacklevel=2)
            if metric == "migration_mb_per_admitted_sfc" and (
                float(subset["mean"].max()) <= tolerance
            ):
                warnings.warn("No migrations measured; this can be a valid result", stacklevel=2)
    if problems:
        message = "\n - ".join(problems)
        raise ValueError(
            "Invalid figure data:\n - "
            + message
        )


if __name__ == "__main__":
    main()
