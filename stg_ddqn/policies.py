from __future__ import annotations

from pathlib import Path

import numpy as np

from .reward import active_reward
from .types import DecisionObservation, Transition


class Policy:
    name = "policy"

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        raise NotImplementedError

    def observe(self, transition: Transition) -> dict[str, float]:
        return {}

    def save(self, path: str | Path) -> None:
        del path


class DelayGreedyPolicy(Policy):
    name = "delay_greedy"

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        return min(
            range(len(observation.candidates)),
            key=lambda index: (
                observation.candidates[index].delay_ms,
                -observation.candidates[index].minimum_cpu_residual_ratio,
                index,
            ),
        )


class RetainFirstPolicy(Policy):
    name = "retain_first"

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        for index, plan in enumerate(observation.candidates):
            if plan.kind in {"retain", "route_update"}:
                return index
        return min(
            range(len(observation.candidates)),
            key=lambda index: (
                observation.candidates[index].migration_volume_mb,
                observation.candidates[index].delay_ms,
            ),
        )


class IterativeMigrationPolicy(Policy):
    """Reproducible IM-style selective-migration implementation for [3]."""

    name = "im"

    def __init__(self, trigger_ratio: float):
        self.trigger_ratio = float(trigger_ratio)

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        if observation.is_new:
            return _shared_initial_selection(observation)
        retain = _retain_index(observation)
        if retain is not None:
            plan = observation.candidates[retain]
            if plan.delay_ms <= self.trigger_ratio * float(
                observation.metadata["latency_requirement_ms"]
            ):
                return retain
        remaps = _remap_indices(observation)
        if not remaps:
            return retain if retain is not None else _shared_initial_selection(observation)
        return min(
            remaps,
            key=lambda index: (
                len(observation.candidates[index].changed_vnfs),
                observation.candidates[index].migration_volume_mb,
                observation.candidates[index].delay_ms,
                index,
            ),
        )


class DynamicLatencyAwarePartialMigrationPolicy(Policy):
    """Reproducible DLAPM-style partial-remapping implementation for [5]."""

    name = "dlapm"

    def __init__(self, improvement_threshold: float):
        self.improvement_threshold = float(improvement_threshold)

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        if observation.is_new:
            return _shared_initial_selection(observation)
        retain = _retain_index(observation)
        current_delay = float(observation.metadata["pre_action_delay_ms"])
        remaps = _remap_indices(observation)
        if not remaps:
            return retain if retain is not None else _shared_initial_selection(observation)
        best = min(
            remaps,
            key=lambda index: (
                len(observation.candidates[index].changed_vnfs),
                observation.candidates[index].delay_ms,
                observation.candidates[index].migration_volume_mb,
                index,
            ),
        )
        improvement = (
            current_delay - observation.candidates[best].delay_ms
        ) / float(observation.metadata["latency_requirement_ms"])
        if retain is not None and improvement < self.improvement_threshold:
            return retain
        return best


class LifetimeAwareGreedyPolicy(Policy):
    """Myopic equation-(24) oracle used for debugging, not a paper baseline."""

    name = "lifetime_greedy"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        if observation.is_new:
            return _shared_initial_selection(observation)
        request = observation.metadata["request"]
        return max(
            range(len(observation.candidates)),
            key=lambda index: active_reward(
                self.cfg,
                request,
                observation.candidates[index],
                float(observation.metadata["pre_action_delay_ms"]),
                float(observation.metadata["remaining_lifetime_s"]),
            ),
        )


class RandomFeasiblePolicy(Policy):
    name = "random_feasible"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def select(self, observation: DecisionObservation, training: bool = False) -> int:
        del training
        return int(self.rng.integers(0, len(observation.candidates)))


def _shared_initial_selection(observation: DecisionObservation) -> int:
    return min(
        range(len(observation.candidates)),
        key=lambda index: (
            observation.candidates[index].delay_ms,
            -observation.candidates[index].minimum_cpu_residual_ratio,
            -observation.candidates[index].minimum_bandwidth_residual_ratio,
            index,
        ),
    )


def _retain_index(observation: DecisionObservation) -> int | None:
    for index, plan in enumerate(observation.candidates):
        if plan.kind in {"retain", "route_update"}:
            return index
    return None


def _remap_indices(observation: DecisionObservation) -> list[int]:
    """Return only genuine host-changing actions.

    Both migration baselines previously ranked RETAIN together with remapping
    plans.  Because RETAIN changes zero VNFs, it always won the partial-
    migration ordering even after a migration trigger fired.  That silently
    reduced IM and DLAPM to identical retain-only policies.
    """

    return [
        index
        for index, plan in enumerate(observation.candidates)
        if plan.kind == "remap" and bool(plan.changed_vnfs)
    ]


def make_policy(name: str, cfg: dict, seed: int = 0,
                checkpoint: str | None = None, training: bool = False) -> Policy:
    normalized = name.lower().replace("-", "_")
    if normalized == "delay_greedy":
        return DelayGreedyPolicy()
    if normalized == "retain_first":
        return RetainFirstPolicy()
    if normalized == "im":
        return IterativeMigrationPolicy(cfg["baselines"]["im_latency_trigger_ratio"])
    if normalized == "dlapm":
        return DynamicLatencyAwarePartialMigrationPolicy(
            cfg["baselines"]["dlapm_improvement_threshold"]
        )
    if normalized == "lifetime_greedy":
        return LifetimeAwareGreedyPolicy(cfg)
    if normalized == "random_feasible":
        return RandomFeasiblePolicy(seed)
    if normalized in {"stg_ddqn", "graph_ddqn"}:
        from .neural import STGDDQNPolicy

        policy = STGDDQNPolicy(
            cfg,
            seed=seed,
            spatial_only=(normalized == "graph_ddqn"),
        )
        if checkpoint:
            policy.load(checkpoint)
        elif not training:
            raise ValueError(
                f"{normalized} evaluation requires a trained checkpoint; "
                "run python -m stg_ddqn.train first"
            )
        return policy
    raise ValueError(f"Unknown policy: {name}")
