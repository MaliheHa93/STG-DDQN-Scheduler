from __future__ import annotations

import argparse

from .config import load_config
from .runner import run_single
from .stats import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired-seed STG-DDQN evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algorithms", default=None, help="Comma-separated override")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer override")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/evaluation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    algorithms = (
        [item.strip() for item in args.algorithms.split(",")]
        if args.algorithms else list(cfg["experiments"]["algorithms"])
    )
    seeds = (
        [int(item) for item in args.seeds.split(",")]
        if args.seeds else [int(seed) for seed in cfg["simulation"]["seeds"]]
    )
    checkpoint = args.checkpoint or cfg["experiments"].get("checkpoint")
    raw_rows, request_rows = [], []
    for algorithm in algorithms:
        for seed in seeds:
            selected_checkpoint = checkpoint if algorithm in {"stg_ddqn", "graph_ddqn"} else None
            row, requests, _ = run_single(
                cfg, algorithm, seed, checkpoint=selected_checkpoint, training=False
            )
            row.update({"experiment": "single", "x_name": "seed", "x_value": seed})
            raw_rows.append(row)
            request_rows.extend(requests)
            print(
                f"algorithm={algorithm} seed={seed} "
                f"acceptance={row['acceptance_ratio_pct']:.2f}% "
                f"continuity={row['continuity_satisfaction_ratio_pct']:.2f}%"
            )
    paths = write_results(raw_rows, request_rows, args.output)
    print("\n".join(str(path.resolve()) for path in paths))


if __name__ == "__main__":
    main()

