from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
import atexit
from pathlib import Path
import shutil

import pandas as pd

from .config import load_config
from .runner import config_digest, run_single
from .mobility import resolve_sumo_run
from .stats import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic, mobility, and scalability sweeps")
    parser.add_argument("--config", default="configs/sumo_paper.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/paper")
    parser.add_argument("--experiments", default="traffic,mobility,scalability")
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithm override, useful for partitioned runs",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated paired-seed override, useful for partitioned runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted sweep from the existing raw CSV",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent algorithm/seed runs to execute concurrently",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    requested = {value.strip() for value in args.experiments.split(",")}
    algorithms = (
        [value.strip() for value in args.algorithms.split(",") if value.strip()]
        if args.algorithms else list(cfg["experiments"]["algorithms"])
    )
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds else [int(seed) for seed in cfg["simulation"]["seeds"]]
    )
    checkpoint = args.checkpoint or cfg["experiments"].get("checkpoint")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock,Timeout
    lock=FileLock(str(output / '.evaluation.lock'))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        if not args.resume:
            raise RuntimeError('This result directory already has an active evaluation writer')
        print('Waiting for the existing evaluation writer...',flush=True)
        lock.acquire()
    atexit.register(lock.release)
    frozen_checkpoint, checkpoint_hash = _freeze_checkpoint(
        checkpoint,
        output,
        resume=bool(args.resume),
        required=any(name in {"stg_ddqn", "graph_ddqn"} for name in algorithms),
    )
    if any(name in {"stg_ddqn", "graph_ddqn"} for name in algorithms):
        if frozen_checkpoint is None:
            raise FileNotFoundError(
                "A trained checkpoint is required for the learning schemes. "
                "Run python -m stg_ddqn.train first."
            )
    raw_path = output / "evaluation_raw.csv"
    request_path = output / "request_records.csv"
    if args.resume and raw_path.exists():
        raw_rows = pd.read_csv(raw_path).to_dict("records")
        if any(row.get("schema_version") != 3 for row in raw_rows):
            raise ValueError("Cannot resume legacy results: use a new schema-3 output directory")
        request_rows = (
            pd.read_csv(request_path).to_dict("records")
            if request_path.exists() else []
        )
    else:
        raw_rows, request_rows = [], []
    recorded_hashes = {
        str(row.get("checkpoint_sha256"))
        for row in raw_rows
        if pd.notna(row.get("checkpoint_sha256")) and row.get('checkpoint_sha256') != 'not_applicable'
    }
    if checkpoint_hash and recorded_hashes and recorded_hashes != {checkpoint_hash}:
        raise ValueError(
            "The resumed result directory was created with a different "
            "STG-DDQN checkpoint. Use a new output directory."
        )
    completed_keys = {
        (
            str(row["experiment"]),
            float(row["x_value"]),
            str(row["algorithm"]),
            int(row["seed"]),
        )
        for row in raw_rows
    }
    scenarios = list(_scenarios(cfg, requested))
    expected_digests = {(experiment, float(x_value), seed): config_digest(resolve_sumo_run(scenario, seed))
                        for experiment, _, x_value, scenario in scenarios for seed in seeds}
    for _, _, _, scenario in scenarios:
        for seed in seeds:
            resolved = resolve_sumo_run(scenario, seed)
            if resolved['mobility']['mode'] == 'sumo_csv':
                trace = Path(resolved['mobility']['sumo_csv'])
                if not trace.is_file():
                    raise FileNotFoundError(f'Generate the SUMO trace first: {trace}')
    for row in raw_rows:
        key = (str(row["experiment"]), float(row["x_value"]), int(row['seed']))
        if key in expected_digests and row.get("config_digest") != expected_digests[key]:
            raise ValueError("Cannot resume results from a different configuration")
        if key in expected_digests and cfg['mobility']['mode'] == 'sumo_csv':
            scenario = next(s for experiment, _, x, s in scenarios
                            if experiment == key[0] and float(x) == key[1])
            resolved = resolve_sumo_run(scenario, key[2])
            if row.get('sumo_trace_sha256') != _sha256(Path(resolved['mobility']['sumo_csv'])):
                raise ValueError('Cannot resume results after a SUMO trace changed')
    total = len(scenarios) * len(algorithms) * len(seeds)
    completed = 0
    jobs = []
    for experiment, x_name, x_value, scenario_cfg in scenarios:
        for algorithm in algorithms:
            for seed in seeds:
                key = (experiment, float(x_value), algorithm, int(seed))
                if key in completed_keys:
                    completed += 1
                    print(
                        f"[{completed}/{total}] resume-skip {experiment} "
                        f"{x_value} {algorithm} seed={seed}"
                    )
                    continue
                selected_checkpoint = (
                    str(frozen_checkpoint)
                    if algorithm in {"stg_ddqn", "graph_ddqn"} else None
                )
                jobs.append((
                    scenario_cfg,
                    algorithm,
                    seed,
                    selected_checkpoint,
                    str(Path(checkpoint)) if selected_checkpoint else None,
                    experiment,
                    x_name,
                    x_value,
                    checkpoint_hash if selected_checkpoint else "not_applicable",
                ))

    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            completed = _record_result(
                _run_job(job), raw_rows, request_rows, completed_keys,
                output, completed, total,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                completed = _record_result(
                    future.result(), raw_rows, request_rows, completed_keys,
                    output, completed, total,
                )
    paths = write_results(raw_rows, request_rows, output)
    (output / 'protocol.json').write_text(json.dumps({
        'status': 'complete', 'configuration': cfg, 'evaluation_seeds': seeds,
        'algorithm_names': algorithms, 'requested_experiments': sorted(requested),
        'expected_runs': total, 'stored_runs': len(raw_rows),
        'checkpoint_sha256': checkpoint_hash,
        'ci_unit': 'paired evaluation realization, conditional on one trained policy',
        'runtime_workers': workers,
        'source_sha256': {path.name: _sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
    }, indent=2) + '\n')
    (output / 'progress.json').write_text(json.dumps(dict(status='complete',
        completed_runs=len(raw_rows),expected_runs=total),indent=2)+'\n')
    print("\n".join(str(path.resolve()) for path in paths))


def _freeze_checkpoint(
    checkpoint: str | None,
    output: Path,
    resume: bool,
    required: bool,
) -> tuple[Path | None, str | None]:
    if not required:
        return None, None
    if not checkpoint:
        return None, None
    source = Path(checkpoint)
    frozen = output / "frozen_stg_ddqn.pt"
    if resume and frozen.exists():
        return frozen, _sha256(frozen)
    if not source.exists():
        return None, None
    temporary = frozen.with_name(frozen.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, frozen)
    return frozen, _sha256(frozen)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenarios(cfg: dict, requested: set[str]):
    experiments = cfg["experiments"]
    trace_pattern = cfg['mobility'].get('sumo_trace_pattern')
    if "mobility" in requested and cfg["mobility"]["mode"] == "sumo_csv" and not trace_pattern:
        fingerprints = set()
        for speed in experiments["mobility_speeds_m_s"]:
            path = _speed_trace(cfg, speed)
            fingerprint = _sha256(Path(path))
            if fingerprint in fingerprints:
                raise ValueError("Different SUMO speed conditions must use distinct trace contents")
            fingerprints.add(fingerprint)
    if "traffic" in requested:
        for arrival_rate in experiments["traffic_arrival_rates"]:
            scenario = deepcopy(cfg)
            scenario["traffic"]["arrival_rate_req_s"] = float(arrival_rate)
            scenario["mobility"]["maximum_speed_m_s"] = float(
                experiments["traffic_fixed_maximum_speed_m_s"]
            )
            if scenario["mobility"]["mode"] == "sumo_csv" and not trace_pattern:
                scenario["mobility"]["sumo_csv"] = _speed_trace(
                    cfg, experiments["traffic_fixed_maximum_speed_m_s"]
                )
            yield "traffic", "arrival_rate_req_s", float(arrival_rate), scenario
    if "mobility" in requested:
        for maximum_speed in experiments["mobility_speeds_m_s"]:
            scenario = deepcopy(cfg)
            scenario["mobility"]["maximum_speed_m_s"] = float(maximum_speed)
            if scenario["mobility"]["mode"] == "sumo_csv" and not trace_pattern:
                scenario["mobility"]["sumo_csv"] = _speed_trace(cfg, maximum_speed)
            scenario["traffic"]["arrival_rate_req_s"] = float(
                experiments["mobility_fixed_arrival_rate_req_s"]
            )
            yield "mobility", "maximum_speed_m_s", float(maximum_speed), scenario
    if "scalability" in requested:
        for nodes in experiments["scalability_nodes"]:
            scenario = deepcopy(cfg)
            scenario["network"]["nodes"] = int(nodes)
            scenario["network"]["physical_links"] = max(int(nodes) - 1, 2 * int(nodes) - 5)
            yield "scalability", "nodes", int(nodes), scenario


def _speed_trace(cfg: dict, speed: float) -> str:
    mapping = cfg["mobility"].get("sumo_csv_by_speed", {})
    for key, path in mapping.items():
        if float(key) == float(speed):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing SUMO trace for {speed} m/s: {path}")
            return str(path)
    raise ValueError(
        f"SUMO mobility sweep needs mobility.sumo_csv_by_speed[{speed}]. "
        "Changing maximum_speed_m_s does not modify a recorded SUMO trace."
    )


def _run_job(job):
    (
        scenario_cfg,
        algorithm,
        seed,
        checkpoint,
        checkpoint_source,
        experiment,
        x_name,
        x_value,
        checkpoint_hash,
    ) = job
    if checkpoint:
        selected = Path(checkpoint)
        if not selected.exists() and checkpoint_source:
            selected = Path(checkpoint_source)
        if not selected.exists():
            raise FileNotFoundError(
                "The frozen and source STG-DDQN checkpoints are both missing"
            )
        actual_hash = _sha256(selected)
        if actual_hash != checkpoint_hash:
            raise ValueError(
                "The STG-DDQN checkpoint changed during evaluation; "
                "start a new result directory"
            )
        checkpoint = str(selected)
    row, requests, _ = run_single(
        scenario_cfg,
        algorithm,
        seed,
        checkpoint=checkpoint,
        training=False,
    )
    row.update({
        "experiment": experiment,
        "x_name": x_name,
        "x_value": x_value,
        "checkpoint_sha256": checkpoint_hash,
    })
    for request_row in requests:
        request_row.update({
            "experiment": experiment,
            "x_name": x_name,
            "x_value": x_value,
        })
    return row, requests


def _record_result(
    result,
    raw_rows: list[dict],
    request_rows: list[dict],
    completed_keys: set[tuple],
    output: Path,
    completed: int,
    total: int,
) -> int:
    row, requests = result
    key = (
        str(row["experiment"]),
        float(row["x_value"]),
        str(row["algorithm"]),
        int(row["seed"]),
    )
    raw_rows.append(row)
    request_rows.extend(requests)
    completed_keys.add(key)
    completed += 1
    print(
        f"[{completed}/{total}] {row['experiment']} {row['x_value']} "
        f"{row['algorithm']} seed={row['seed']}",
        flush=True,
    )
    raw_rows.sort(key=lambda item: (
        str(item.get("experiment", "")),
        float(item.get("x_value", 0.0)),
        str(item.get("algorithm", "")),
        int(item.get("seed", 0)),
    ))
    request_rows.sort(key=lambda item: (
        str(item.get("experiment", "")),
        float(item.get("x_value", 0.0)),
        str(item.get("algorithm", "")),
        int(item.get("seed", 0)),
        int(item.get("request_id", 0)),
    ))
    # Save after every completed realization so an interrupted paper run loses
    # at most the still-running worker results.
    write_results(raw_rows, request_rows, output)
    progress=output / 'progress.json'
    temporary=progress.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(status='evaluating',completed_runs=len(raw_rows),
        expected_runs=total,last_experiment=row['experiment'],last_x_value=row['x_value'],
        last_algorithm=row['algorithm'],last_seed=row['seed']),indent=2)+'\n')
    os.replace(temporary,progress)
    return completed


if __name__ == "__main__":
    main()
