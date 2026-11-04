from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VNFSpec:
    cpu: float
    workload_mi: float
    image_mb: float
    state_mb: float


@dataclass
class SFCRequest:
    request_id: int
    vehicle_id: int
    arrival_s: float
    lifetime_s: float
    bandwidth_mb_s: float
    latency_requirement_ms: float
    vnfs: tuple[VNFSpec, ...]
    tracked: bool = True

    @property
    def expiry_s(self) -> float:
        return self.arrival_s + self.lifetime_s


@dataclass(frozen=True)
class MigrationStep:
    vnf_index: int
    source: int
    destination: int
    path: tuple[int, ...]
    bandwidth_mb_s: float
    transfer_time_ms: float
    volume_mb: float


@dataclass(frozen=True)
class PlacementPlan:
    hosts: tuple[int, ...]
    route: tuple[int, ...]
    access_node: int
    delay_ms: float
    migration_volume_mb: float
    preparation_time_ms: float
    downtime_ms: float
    changed_vnfs: tuple[int, ...]
    kind: str
    minimum_cpu_residual_ratio: float
    minimum_bandwidth_residual_ratio: float
    migration_steps: tuple[MigrationStep, ...] = ()


@dataclass
class ActiveSFC:
    request: SFCRequest
    hosts: tuple[int, ...]
    route: tuple[int, ...]
    access_node: int
    admitted_s: float
    migration_volume_mb: float = 0.0
    downtime_ms: float = 0.0
    migrations: int = 0
    route_updates: int = 0
    handoffs: int = 0
    continuity_failed: bool = False
    resolved: bool = False
    last_delay_ms: float = 0.0
    pending: PendingMigration | None = None
    pending_decision: PendingDecision | None = None
    transferred_mb: float = 0.0
    canceled_transfer_mb: float = 0.0
    canceled_migrations: int = 0


@dataclass
class PendingMigration:
    plan: PlacementPlan
    old_hosts: tuple[int, ...]
    old_route: tuple[int, ...]
    observation: DecisionObservation
    action_index: int
    pre_action_delay_ms: float
    remaining_lifetime_s_at_decision: float
    started_s: float
    step_index: int = 0
    step_remaining_ms: float = 0.0
    reserved_step_edges: tuple[tuple[int, int], ...] = ()
    reserved_step_bandwidth_mb_s: float = 0.0
    step_transferred_mb: float = 0.0
    transferred_mb: float = 0.0


@dataclass
class DecisionObservation:
    request_id: int
    is_new: bool
    candidates: tuple[PlacementPlan, ...]
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_index: np.ndarray
    vehicle_features: np.ndarray
    request_features: np.ndarray
    vnf_features: np.ndarray
    current_hosts: np.ndarray
    candidate_hosts: np.ndarray
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def compact_copy(self) -> DecisionObservation:
        return DecisionObservation(
            request_id=self.request_id,
            is_new=self.is_new,
            candidates=(),
            node_features=self.node_features.astype(np.float16, copy=True),
            edge_features=self.edge_features.astype(np.float16, copy=True),
            edge_index=self.edge_index.astype(np.int32, copy=True),
            vehicle_features=self.vehicle_features.astype(np.float16, copy=True),
            request_features=self.request_features.astype(np.float16, copy=True),
            vnf_features=self.vnf_features.astype(np.float16, copy=True),
            current_hosts=self.current_hosts.astype(np.int16, copy=True),
            candidate_hosts=self.candidate_hosts.astype(np.int16, copy=True),
            candidate_features=self.candidate_features.astype(np.float16, copy=True),
            candidate_mask=self.candidate_mask.astype(bool, copy=True),
            metadata={},
        )


@dataclass
class Transition:
    observation: DecisionObservation
    action_index: int
    reward: float
    next_observation: DecisionObservation | None
    terminal: bool


@dataclass
class PendingDecision:
    """An instantaneous action awaiting the next real decision state.

    RETAIN, route-only, and initial-placement actions finish immediately in the
    infrastructure, but their reinforcement-learning transition must end at the
    next control/handoff/terminal event rather than at a duplicate same-time
    state.  Keeping this credit assignment explicit prevents a RETAIN self-loop
    from hiding a later mobility-induced failure.
    """

    observation: DecisionObservation
    action_index: int
    reward: float


@dataclass
class RequestRecord:
    request_id: int
    tracked: bool
    generated: bool = True
    admitted: bool = False
    blocked: bool = False
    continuity_resolved: bool = False
    continuity_satisfied: bool = False
    continuity_failed: bool = False
    still_active_at_end: bool = False
    migration_volume_mb: float = 0.0
    downtime_ms: float = 0.0
    migrations: int = 0
    handoffs: int = 0
    route_updates: int = 0
    mean_delay_ms: float = 0.0
    delay_samples: int = 0
    departure_reason: str = ""
    arrival_s: float = 0.0
    admitted_s: float = 0.0
    ended_s: float = 0.0
    observed_service_s: float = 0.0
    completed_migration_volume_mb: float = 0.0
    canceled_transfer_mb: float = 0.0
    canceled_migrations: int = 0

    def add_delay(self, delay_ms: float) -> None:
        total = self.mean_delay_ms * self.delay_samples + delay_ms
        self.delay_samples += 1
        self.mean_delay_ms = total / self.delay_samples
