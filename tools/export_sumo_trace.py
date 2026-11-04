from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an epoch-sampled SUMO mobility CSV")
    parser.add_argument("--sumocfg", required=True)
    parser.add_argument("--output", default="data/mobility/sumo_trace.csv")
    parser.add_argument("--epoch", type=float, default=1.0)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    try:
        import traci
    except ImportError as exc:
        raise SystemExit(
            "SUMO TraCI is missing. Install SUMO and its Python tools, then retry."
        ) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.sumo_binary,
        "-c", str(Path(args.sumocfg).resolve()),
        "--seed", str(args.seed),
        "--step-length", str(args.epoch),
        "--no-step-log", "true",
    ]
    traci.start(command)
    try:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "time_s", "vehicle_id", "x_m", "y_m", "speed_m_s", "heading_rad"
            ))
            writer.writeheader()
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                time_s = float(traci.simulation.getTime())
                for vehicle_id in sorted(traci.vehicle.getIDList()):
                    x_m, y_m = traci.vehicle.getPosition(vehicle_id)
                    # SUMO reports navigation angle in degrees clockwise from north.
                    navigation_deg = float(traci.vehicle.getAngle(vehicle_id))
                    heading_rad = math.radians(90.0 - navigation_deg)
                    writer.writerow({
                        "time_s": time_s,
                        "vehicle_id": vehicle_id,
                        "x_m": x_m,
                        "y_m": y_m,
                        "speed_m_s": traci.vehicle.getSpeed(vehicle_id),
                        "heading_rad": heading_rad,
                    })
    finally:
        traci.close()
    print(output.resolve())


if __name__ == "__main__":
    main()

