from __future__ import annotations

from .types import PlacementPlan, SFCRequest


def initial_reward(cfg: dict, request: SFCRequest, plan: PlacementPlan) -> float:
    """Equation (23)."""
    weight = float(cfg["reward"]["latency"])
    return weight * (1.0 - plan.delay_ms / request.latency_requirement_ms)


def active_reward(cfg: dict, request: SFCRequest, plan: PlacementPlan,
                  pre_action_delay_ms: float, remaining_lifetime_s: float) -> float:
    """Equation (24), evaluated when a transition is finalized."""
    weights = cfg["reward"]
    delay_ratio_term = 1.0 - plan.delay_ms / request.latency_requirement_ms
    improvement = _clip(
        (pre_action_delay_ms - plan.delay_ms) / request.latency_requirement_ms,
        -1.0,
        1.0,
    )
    lifetime_ratio = _clip(remaining_lifetime_s / request.lifetime_s, 0.0, 1.0)
    migration_ratio = (
        plan.migration_volume_mb / float(cfg["migration"]["volume_normalizer_mb"])
    )
    downtime_ratio = (
        plan.downtime_ms / float(cfg["migration"]["downtime_normalizer_ms"])
    )
    return (
        float(weights["latency"]) * delay_ratio_term
        + float(weights["lifetime_benefit"]) * lifetime_ratio * improvement
        - float(weights["migration_volume"]) * migration_ratio
        - float(weights["downtime"]) * downtime_ratio
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))

