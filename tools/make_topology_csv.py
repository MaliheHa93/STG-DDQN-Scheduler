from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stg_ddqn.config import load_config
from stg_ddqn.topology import Topology


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the generated RSU topology for audit")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="data/topology")
    parser.add_argument("--resource-seed", type=int, default=101)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    topology = Topology(
        cfg["network"],
        np.random.default_rng(cfg["simulation"]["topology_structure_seed"]),
        np.random.default_rng(args.resource_seed),
    )
    with (output / "nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", "x_m", "y_m", "cpu_capacity",
                         "processing_rate_mi_ms", "coverage_radius_m"))
        for node in topology.nodes.values():
            writer.writerow((node.node_id, node.x_m, node.y_m, node.cpu_capacity,
                             node.processing_rate_mi_ms, node.coverage_radius_m))
    with (output / "links.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "destination", "capacity_mb_s", "delay_ms", "bidirectional"))
        for (u, v), link in topology.links.items():
            if u < v and (v, u) in topology.links:
                writer.writerow((u, v, link.capacity_mb_s, link.delay_ms, "true"))
    print(output.resolve())


if __name__ == "__main__":
    main()
