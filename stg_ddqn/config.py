from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        cfg = deep_merge(load_config(parent_path), raw)
    else:
        cfg = raw
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    _resolve_paths(cfg, path.parent)
    validate_config(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any], base: Path) -> None:
    path_keys = (
        ("network", "topology_nodes_csv"),
        ("network", "topology_links_csv"),
        ("mobility", "sumo_csv"),
        ("mobility", "sumo_trace_pattern"),
        ("experiments", "checkpoint"),
    )
    for section, key in path_keys:
        value = cfg.get(section, {}).get(key)
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                cfg[section][key] = str((base / candidate).resolve())
    traces = cfg.get("mobility", {}).get("sumo_csv_by_speed", {})
    for speed, value in list(traces.items()):
        candidate = Path(value)
        if not candidate.is_absolute():
            traces[speed] = str((base / candidate).resolve())


def validate_config(cfg: dict[str, Any]) -> None:
    required = {
        "simulation", "network", "mobility", "traffic", "migration",
        "candidates", "state", "learning", "reward", "experiments",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    weights = cfg["reward"]
    weight_sum = sum(float(weights[name]) for name in (
        "latency", "lifetime_benefit", "migration_volume", "downtime"
    ))
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"Reward weights must sum to 1, got {weight_sum}")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("Reward weights must be nonnegative")
    if cfg["mobility"].get("dwell_estimator", "ray_circle") not in {"radial", "ray_circle"}:
        raise ValueError("Unknown dwell_estimator")
    if cfg["traffic"].get("arrival_model", "poisson") not in {"poisson", "mmpp"}:
        raise ValueError("Unknown arrival_model")
    if cfg["traffic"].get("arrival_model") == "mmpp":
        m = cfg["traffic"].get("mmpp", {})
        low, high = float(m.get("low_factor", 0.25)), float(m.get("high_factor", 2.5))
        if not 0 <= low < 1 < high:
            raise ValueError("MMPP factors must satisfy 0 <= low < 1 < high")
        if float(m.get("high_duration_s", 10)) <= 0:
            raise ValueError("MMPP high_duration_s must be positive")
    if cfg["state"]["maximum_sfc_length"] < cfg["traffic"]["sfc_length"][1]:
        raise ValueError("state.maximum_sfc_length must cover the SFC-length range")
    if cfg["candidates"]["maximum_actions"] < 1:
        raise ValueError("At least one candidate action is required")
    limit = cfg['candidates'].get('maximum_fallback_paths')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('maximum_fallback_paths must be a positive integer or null')
    assignment_limit = cfg['candidates'].get('maximum_assignment_nodes')
    if assignment_limit is not None and (not isinstance(assignment_limit, int) or assignment_limit < 1):
        raise ValueError('maximum_assignment_nodes must be a positive integer or null')


def set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def cloned_with(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    result = deepcopy(cfg)
    for key, value in overrides.items():
        set_nested(result, key.replace("__", "."), value)
    validate_config(result)
    return result

