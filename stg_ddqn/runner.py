from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from copy import deepcopy

from .environment import STGEnvironment
from .mobility import resolve_sumo_run
from .policies import Policy, make_policy


def run_single(cfg: dict, algorithm: str, seed: int,
               checkpoint: str | None = None, training: bool = False,
               policy: Policy | None = None, environment: STGEnvironment | None = None,
               start_frame: int = 0, epoch_callback=None) -> tuple[dict, list[dict], Policy]:
    cfg = resolve_sumo_run(cfg, seed, training)
    selected_policy = policy or make_policy(
        algorithm, cfg, seed=seed + 10_000, checkpoint=checkpoint, training=training
    )
    environment_seed = seed + int(cfg.get('training_seed', 0)) * 1_000_000 if training else seed
    environment = environment or STGEnvironment(cfg, environment_seed)
    aggregate, request_rows = environment.run(selected_policy, training=training,
        start_frame=start_frame, epoch_callback=epoch_callback)
    aggregate.update({
        "algorithm": algorithm,
        "config_digest": config_digest(cfg),
        "mobility_mode": cfg["mobility"]["mode"],
        "sumo_trace_seed": cfg['mobility'].get('trace_seed', 'not_applicable'),
        "network_nodes": cfg["network"]["nodes"],
        "network_physical_links": cfg["network"]["physical_links"],
        "arrival_rate_req_s": cfg["traffic"]["arrival_rate_req_s"],
        "maximum_speed_m_s": cfg["mobility"]["maximum_speed_m_s"],
        "arrival_model": cfg["traffic"].get("arrival_model", "poisson"),
        "dwell_estimator": cfg["mobility"].get("dwell_estimator", "ray_circle"),
        "sumo_trace_sha256": (
            hashlib.sha256(Path(cfg["mobility"]["sumo_csv"]).read_bytes()).hexdigest()
            if cfg["mobility"]["mode"] == "sumo_csv" else "not_applicable"
        ),
        "runtime_platform": platform.platform(),
        "python_version": platform.python_version(),
        "runtime_context": __import__('os').environ.get('STG_RUNTIME_CONTEXT','unspecified'),
    })
    for row in request_rows:
        row.update({"algorithm": algorithm, "seed": seed})
    return aggregate, request_rows, selected_policy


def config_digest(cfg: dict) -> str:
    clean = deepcopy(cfg)
    clean.pop("_config_path", None)
    clean.pop("_config_dir", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

