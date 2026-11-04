from __future__ import annotations

import math
import time
from copy import deepcopy
from collections import defaultdict
from typing import Iterable

import numpy as np

from .candidates import CandidateGenerator
from .mobility import VehicleState, make_mobility
from .policies import Policy
from .requests import RequestGenerator
from .reward import active_reward, initial_reward
from .state import StateHistory
from .topology import Topology
from .types import (
    ActiveSFC,
    DecisionObservation,
    PendingDecision,
    PendingMigration,
    PlacementPlan,
    RequestRecord,
    SFCRequest,
    Transition,
)


class STGEnvironment:
    """Event-driven implementation of Algorithms 1 and 2."""

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        seed_sequence = np.random.SeedSequence(self.seed)
        resource_ss, mobility_ss, arrival_ss, request_ss = seed_sequence.spawn(4)
        structure_rng = np.random.default_rng(
            int(cfg["simulation"]["topology_structure_seed"])
        )
        self.topology = Topology(
            cfg["network"], structure_rng, np.random.default_rng(resource_ss)
        )
        self.mobility = make_mobility(
            cfg["mobility"],
            self.topology,
            np.random.default_rng(mobility_ss),
            float(cfg["simulation"]["epoch_s"]),
        )
        self.request_generator = RequestGenerator(
            cfg["traffic"],
            float(cfg["simulation"]["epoch_s"]),
            np.random.default_rng(arrival_ss),
            np.random.default_rng(request_ss),
        )
        self.candidate_generator = CandidateGenerator(cfg, self.topology)
        self.state_history = StateHistory(cfg, self.topology)
        self.active: dict[int, ActiveSFC] = {}
        self.records: dict[int, RequestRecord] = {}
        self.vehicles: dict[int, VehicleState] = {}
        self.previous_access: dict[int, int] = {}
        self.time_s = 0.0
        self.policy: Policy | None = None
        self.training = False

        self.decision_times_ms: list[float] = []
        self.candidate_times_ms: list[float] = []
        self.reference_route_times_ms: list[float] = []
        self.fallback_times_ms: list[float] = []
        self.policy_times_ms: list[float] = []
        self.delay_samples_ms: list[float] = []
        self.cpu_utilization: list[float] = []
        self.bandwidth_utilization: list[float] = []
        self.rewards: list[float] = []
        self.losses: list[float] = []
        self.fallback_decisions = 0
        self.paths_examined = 0
        self.assignment_nodes_examined = 0
        self.total_handoffs = 0
        self.total_decisions = 0
        self.temporary_cpu_seconds = 0.0
        self.temporary_service_bandwidth_mb = 0.0
        self.migration_bandwidth_reservation_mb = 0.0
        self.peak_temporary_cpu = 0.0
        self.peak_temporary_service_bandwidth_mb_s = 0.0
        self.peak_migration_bandwidth_mb_s = 0.0
        self.peak_pending_migrations = 0
        self.blocked_without_temporary_reservations = 0
        self.search_budget_exhausted_decisions = 0

    @property
    def warmup_s(self) -> float:
        return float(self.cfg["simulation"]["warmup_s"])

    @property
    def measurement_end_s(self) -> float:
        return self.warmup_s + float(self.cfg["simulation"]["measurement_s"])

    def run(self, policy: Policy, training: bool = False, start_frame: int = 0,
            epoch_callback=None) -> tuple[dict, list[dict]]:
        self.policy = policy
        self.training = bool(training)
        epoch_s = float(self.cfg["simulation"]["epoch_s"])
        frame_count = int(math.ceil(self.measurement_end_s / epoch_s))
        if not 0 <= start_frame <= frame_count:
            raise ValueError('Resume frame is outside the experiment horizon')
        for frame in range(start_frame, frame_count):
            self.time_s = frame * epoch_s
            self._run_epoch(epoch_s if frame > 0 else 0.0)
            self.topology.assert_consistent()
            if epoch_callback is not None:
                epoch_callback(self, frame + 1, frame_count)
        self._mark_active_at_end()
        return self._aggregate_results(), self._request_rows()

    def _run_epoch(self, elapsed_s: float) -> None:
        self._integrate_reservations(elapsed_s)
        self.vehicles = self.mobility.state_at(self.time_s)
        self._release_expired_or_departed()
        handoff_ids = self._update_access_and_routes()
        self.state_history.record(self.time_s, self.vehicles)
        reconsider_ids = self._advance_pending(elapsed_s, handoff_ids)
        self.state_history.record(self.time_s, self.vehicles)

        tracked = self.time_s >= self.warmup_s
        arrivals = self.request_generator.generate(
            self.time_s, sorted(self.vehicles), tracked=tracked
        )
        for request in arrivals:
            self.records[request.request_id] = RequestRecord(
                request_id=request.request_id, tracked=request.tracked,
                arrival_s=request.arrival_s,
            )

        control_period = float(self.cfg["simulation"]["control_period_s"])
        periodic = abs((self.time_s / control_period) - round(self.time_s / control_period)) < 1e-9
        active_events = set(handoff_ids) | set(reconsider_ids)
        if periodic:
            active_events.update(self.active)
        for request_id in sorted(active_events):
            active = self.active.get(request_id)
            if active is None or active.pending is not None:
                continue
            self._decide(active.request, active)
        for request in arrivals:
            if request.vehicle_id in self.vehicles:
                self._decide(request, None)
            else:
                record = self.records[request.request_id]
                record.blocked = True
                record.departure_reason = "vehicle_absent_at_arrival"

        self.state_history.record(self.time_s, self.vehicles)
        if tracked:
            cpu, bandwidth = self.topology.utilization()
            self.cpu_utilization.append(cpu)
            self.bandwidth_utilization.append(bandwidth)
            self._sample_active_delays()
            self._capture_reservation_peaks()

    def _release_expired_or_departed(self) -> None:
        for request_id in list(self.active):
            active = self.active[request_id]
            expired = self.time_s >= active.request.expiry_s - 1e-9
            departed = active.request.vehicle_id not in self.vehicles
            if expired or departed:
                reason = "lifetime_expiry" if expired else "vehicle_departure"
                self._resolve_normal(active, reason)

    def _update_access_and_routes(self) -> set[int]:
        handoffs: set[int] = set()
        current_access: dict[int, int] = {}
        for vehicle_id, vehicle in self.vehicles.items():
            access = self.topology.access_node(vehicle.x_m, vehicle.y_m)
            if access is not None:
                current_access[vehicle_id] = access
        for request_id in list(self.active):
            active = self.active.get(request_id)
            if active is None:
                continue
            new_access = current_access.get(active.request.vehicle_id)
            if new_access is None:
                # Vehicle absence is handled before this method. A present
                # vehicle losing coverage is an outage, not successful expiry.
                self._continuity_failure(active, "coverage_gap")
                continue
            if new_access == active.access_node:
                continue
            handoffs.add(request_id)
            active.handoffs += 1
            if active.request.tracked:
                self.total_handoffs += 1
            if active.pending is not None:
                self._cancel_pending(active, "handoff_during_preparation")
            # A handoff is a decision event, not an automatic route rewrite.
            # Candidate generation below will compare a route-only update of
            # the current hosts with feasible selective remappings.  The old
            # implementation refreshed the route here, which erased migration
            # pressure, or failed the SFC before any policy could remap it.
            active.access_node = new_access
        self.previous_access = current_access
        return {request_id for request_id in handoffs if request_id in self.active}

    def _refresh_current_route(self, active: ActiveSFC, new_access: int) -> bool:
        old_edges = set(self.topology.route_edges(active.route))
        route = self.topology.shortest_ordered_route(
            new_access,
            active.hosts,
            required_bandwidth=active.request.bandwidth_mb_s,
            credit_edges=old_edges,
        )
        if route is None:
            return False
        refreshed_delay = self._delay_for(active.request, active.hosts, route)
        if refreshed_delay > active.request.latency_requirement_ms + 1e-9:
            return False
        new_edges = set(self.topology.route_edges(route))
        new_only = new_edges - old_edges
        old_only = old_edges - new_edges
        try:
            self.topology.reserve_service(new_only, active.request.bandwidth_mb_s)
        except RuntimeError:
            return False
        self.topology.release_service(old_only, active.request.bandwidth_mb_s)
        if route != active.route:
            active.route_updates += 1
        active.route = route
        active.access_node = new_access
        active.last_delay_ms = refreshed_delay
        return True

    def _advance_pending(self, elapsed_s: float, handoff_ids: set[int]) -> set[int]:
        reconsider: set[int] = set()
        for request_id in list(self.active):
            active = self.active.get(request_id)
            if active is None or active.pending is None:
                continue
            if request_id in handoff_ids:
                reconsider.add(request_id)
                continue
            pending = active.pending
            remaining_ms = max(0.0, active.request.expiry_s - self.time_s) * 1000.0
            required_ms = pending.step_remaining_ms + sum(
                step.transfer_time_ms
                for step in pending.plan.migration_steps[pending.step_index + 1:]
            ) + pending.plan.downtime_ms
            if required_ms >= remaining_ms:
                self._cancel_pending(active, "insufficient_remaining_lifetime")
                reconsider.add(request_id)
                continue
            budget_ms = elapsed_s * 1000.0
            while budget_ms > 1e-9 and active.pending is not None:
                pending = active.pending
                if pending.step_remaining_ms <= 1e-9:
                    if not self._start_next_migration_step(active):
                        self._cancel_pending(active, "migration_bandwidth_unavailable")
                        reconsider.add(request_id)
                        break
                    pending = active.pending
                    if pending is None:
                        break
                advance = min(budget_ms, pending.step_remaining_ms)
                step = pending.plan.migration_steps[pending.step_index]
                transferred = min(
                    max(0.0, step.volume_mb - pending.step_transferred_mb),
                    advance * pending.reserved_step_bandwidth_mb_s / 1000.0,
                )
                pending.step_transferred_mb += transferred
                pending.transferred_mb += transferred
                active.transferred_mb += transferred
                pending.step_remaining_ms -= advance
                budget_ms -= advance
                if pending.step_remaining_ms <= 1e-9:
                    self.topology.release_migration(
                        pending.reserved_step_edges,
                        pending.reserved_step_bandwidth_mb_s,
                    )
                    pending.reserved_step_edges = ()
                    pending.reserved_step_bandwidth_mb_s = 0.0
                    pending.step_index += 1
                    if pending.step_index >= len(pending.plan.migration_steps):
                        self._finalize_migration(active)
                        break
        return reconsider

    def _decide(self, request: SFCRequest, active: ActiveSFC | None) -> None:
        assert self.policy is not None
        vehicle = self.vehicles.get(request.vehicle_id)
        if vehicle is None:
            if active is None:
                self.records[request.request_id].blocked = True
            else:
                self._resolve_normal(active, "vehicle_departure")
            return
        access = self.topology.access_node(vehicle.x_m, vehicle.y_m)
        if access is None:
            if active is None:
                self.records[request.request_id].blocked = True
            else:
                self._continuity_failure(active, "coverage_gap")
            return
        start = time.perf_counter_ns()
        pre_delay = self._pre_action_delay(request, active, access)
        candidate_start = time.perf_counter_ns()
        reference_elapsed = (candidate_start - start) / 1e6
        candidates = self.candidate_generator.build(
            request, active, access, self.time_s
        )
        candidate_elapsed = (time.perf_counter_ns() - candidate_start) / 1e6
        if self._in_measurement():
            self.candidate_times_ms.append(candidate_elapsed)
            self.reference_route_times_ms.append(reference_elapsed)
            self.fallback_times_ms.append(self.candidate_generator.last_fallback_runtime_ms)
            self.paths_examined += self.candidate_generator.last_paths_examined
            self.assignment_nodes_examined += self.candidate_generator.last_assignment_nodes
            self.search_budget_exhausted_decisions += int(self.candidate_generator.last_search_budget_exhausted)
            if self.candidate_generator.last_fallback_used:
                self.fallback_decisions += 1
        if not candidates:
            if self._in_measurement():
                # A rejected decision is still an online decision attempt.
                # Candidate enumeration/fallback time must therefore be part
                # of end-to-end decision runtime rather than disappearing
                # from the scalability metric merely because no action was
                # feasible.  Policy time is zero because masking terminates
                # the attempt before policy selection.
                self.policy_times_ms.append(0.0)
                self.decision_times_ms.append((time.perf_counter_ns() - start) / 1e6)
                self.total_decisions += 1
            if active is None:
                record = self.records[request.request_id]
                record.blocked = True
                record.departure_reason = (
                    'search_budget_exhausted' if self.candidate_generator.last_search_budget_exhausted
                    else 'no_feasible_placement'
                )
                if request.tracked and self.cfg["simulation"].get("audit_temporary_blocking", False):
                    self._audit_temporary_blocking(request, access)
            else:
                self._continuity_failure(active,
                    'search_budget_exhausted' if self.candidate_generator.last_search_budget_exhausted
                    else 'no_feasible_action')
            if self._in_measurement():
                self.rewards.append(-1.0)
            return
        observation = self.state_history.build(
            request,
            active,
            candidates,
            self.time_s,
            pre_delay,
            self.candidate_generator.last_fallback_used,
            self.candidate_generator.last_paths_examined,
        )
        # Complete the preceding instantaneous action at this genuinely later
        # control/handoff state.  The former implementation created a
        # same-time self-transition here, which prevented RETAIN from receiving
        # credit for a subsequent mobility-induced success or failure.
        if active is not None and active.pending_decision is not None:
            self._finish_pending_decision(
                active,
                next_observation=observation,
                terminal=False,
            )
        policy_start = time.perf_counter_ns()
        action_index = int(self.policy.select(observation, training=self.training))
        policy_elapsed = (time.perf_counter_ns() - policy_start) / 1e6
        total_elapsed = (time.perf_counter_ns() - start) / 1e6
        if action_index < 0 or action_index >= len(candidates):
            raise RuntimeError(f"Policy returned invalid action {action_index}")
        if self._in_measurement():
            self.policy_times_ms.append(policy_elapsed)
            self.decision_times_ms.append(total_elapsed)
            self.total_decisions += 1
        plan = candidates[action_index]
        if active is None:
            self._apply_initial(request, plan, observation, action_index)
        elif plan.hosts == active.hosts:
            self._apply_same_hosts(active, plan, observation, action_index, pre_delay)
        else:
            self._initiate_migration(active, plan, observation, action_index, pre_delay)

    def _apply_initial(self, request: SFCRequest, plan: PlacementPlan,
                       observation: DecisionObservation, action_index: int) -> None:
        cpu = self.topology.node_cpu_amounts(
            plan.hosts, (vnf.cpu for vnf in request.vnfs)
        )
        self.topology.reserve_cpu(cpu)
        self.topology.reserve_service(
            self.topology.route_edges(plan.route), request.bandwidth_mb_s
        )
        active = ActiveSFC(
            request=request,
            hosts=plan.hosts,
            route=plan.route,
            access_node=plan.access_node,
            admitted_s=self.time_s,
            last_delay_ms=plan.delay_ms,
        )
        self.active[request.request_id] = active
        self.records[request.request_id].admitted = True
        self.records[request.request_id].admitted_s = self.time_s
        reward = initial_reward(self.cfg, request, plan)
        self._queue_immediate_transition(
            observation, action_index, reward, active, terminal=False
        )

    def _apply_same_hosts(self, active: ActiveSFC, plan: PlacementPlan,
                          observation: DecisionObservation, action_index: int,
                          pre_delay: float) -> None:
        old_edges = set(self.topology.route_edges(active.route))
        new_edges = set(self.topology.route_edges(plan.route))
        self.topology.reserve_service(
            new_edges - old_edges, active.request.bandwidth_mb_s
        )
        self.topology.release_service(
            old_edges - new_edges, active.request.bandwidth_mb_s
        )
        if plan.route != active.route:
            active.route_updates += 1
        active.route = plan.route
        active.access_node = plan.access_node
        active.last_delay_ms = plan.delay_ms
        reward = active_reward(
            self.cfg,
            active.request,
            plan,
            pre_delay,
            max(0.0, active.request.expiry_s - self.time_s),
        )
        self._queue_immediate_transition(
            observation, action_index, reward, active, terminal=False
        )

    def _initiate_migration(self, active: ActiveSFC, plan: PlacementPlan,
                            observation: DecisionObservation, action_index: int,
                            pre_delay: float) -> None:
        cpu_additions: dict[int, float] = defaultdict(float)
        for index in plan.changed_vnfs:
            cpu_additions[plan.hosts[index]] += active.request.vnfs[index].cpu
        old_edges = set(self.topology.route_edges(active.route))
        new_edges = set(self.topology.route_edges(plan.route))
        new_only = new_edges - old_edges
        self.topology.reserve_cpu(dict(cpu_additions))
        try:
            self.topology.reserve_service(new_only, active.request.bandwidth_mb_s)
        except Exception:
            self.topology.release_cpu(dict(cpu_additions))
            raise
        active.pending = PendingMigration(
            plan=plan,
            old_hosts=active.hosts,
            old_route=active.route,
            observation=observation,
            action_index=action_index,
            pre_action_delay_ms=pre_delay,
            remaining_lifetime_s_at_decision=max(
                0.0, active.request.expiry_s - self.time_s
            ),
            started_s=self.time_s,
        )
        if not self._start_next_migration_step(active):
            self._cancel_pending(active, "migration_bandwidth_unavailable")
        self._capture_reservation_peaks()

    def _start_next_migration_step(self, active: ActiveSFC) -> bool:
        pending = active.pending
        if pending is None:
            return False
        if pending.step_index >= len(pending.plan.migration_steps):
            self._finalize_migration(active)
            return True
        step = pending.plan.migration_steps[pending.step_index]
        edges = self.topology.route_edges(step.path)
        if not edges:
            pending.step_remaining_ms = 0.0
            return True
        available = min(self.topology.links[edge].residual_mb_s for edge in edges)
        bandwidth = min(
            float(self.cfg["migration"]["maximum_bandwidth_mb_s"]), available
        )
        if bandwidth <= 1e-12:
            return False
        self.topology.reserve_migration(edges, bandwidth)
        pending.reserved_step_edges = edges
        pending.reserved_step_bandwidth_mb_s = bandwidth
        pending.step_transferred_mb = 0.0
        pending.step_remaining_ms = (
            1000.0 * step.volume_mb / bandwidth
            + self.topology.path_delay_ms(step.path)
        )
        return True

    def _finalize_migration(self, active: ActiveSFC) -> None:
        pending = active.pending
        if pending is None:
            return
        plan = pending.plan
        release_cpu: dict[int, float] = defaultdict(float)
        for index in plan.changed_vnfs:
            release_cpu[pending.old_hosts[index]] += active.request.vnfs[index].cpu
        self.topology.release_cpu(dict(release_cpu))
        old_edges = set(self.topology.route_edges(pending.old_route))
        new_edges = set(self.topology.route_edges(plan.route))
        self.topology.release_service(
            old_edges - new_edges, active.request.bandwidth_mb_s
        )
        active.hosts = plan.hosts
        active.route = plan.route
        active.access_node = plan.access_node
        active.last_delay_ms = plan.delay_ms
        active.migration_volume_mb += plan.migration_volume_mb
        active.downtime_ms += plan.downtime_ms
        active.migrations += 1
        active.pending = None
        record = self.records[active.request.request_id]
        record.migration_volume_mb = active.migration_volume_mb
        record.downtime_ms = active.downtime_ms
        record.migrations = active.migrations
        reward = active_reward(
            self.cfg,
            active.request,
            plan,
            pending.pre_action_delay_ms,
            pending.remaining_lifetime_s_at_decision,
        )
        self.state_history.record(self.time_s, self.vehicles)
        next_observation = self._next_observation(active)
        self._observe(Transition(
            observation=pending.observation,
            action_index=pending.action_index,
            reward=reward,
            next_observation=next_observation,
            terminal=next_observation is None,
        ))

    def _cancel_pending(self, active: ActiveSFC, reason: str,
                        terminal: bool = False) -> None:
        pending = active.pending
        if pending is None:
            return
        active.canceled_transfer_mb += pending.transferred_mb
        active.canceled_migrations += 1
        if pending.reserved_step_edges:
            self.topology.release_migration(
                pending.reserved_step_edges, pending.reserved_step_bandwidth_mb_s
            )
        cpu_release: dict[int, float] = defaultdict(float)
        for index in pending.plan.changed_vnfs:
            cpu_release[pending.plan.hosts[index]] += active.request.vnfs[index].cpu
        self.topology.release_cpu(dict(cpu_release))
        old_edges = set(self.topology.route_edges(pending.old_route))
        new_edges = set(self.topology.route_edges(pending.plan.route))
        self.topology.release_service(
            new_edges - old_edges, active.request.bandwidth_mb_s
        )
        active.pending = None
        self.state_history.record(self.time_s, self.vehicles)
        next_observation = None if terminal else self._next_observation(active)
        self._observe(Transition(
            observation=pending.observation,
            action_index=pending.action_index,
            reward=-1.0,
            next_observation=next_observation,
            terminal=terminal or next_observation is None,
        ))
        self.records[active.request.request_id].departure_reason = reason

    def _queue_immediate_transition(self, observation: DecisionObservation,
                                    action_index: int, reward: float,
                                    active: ActiveSFC, terminal: bool) -> None:
        if active.pending_decision is not None:
            raise RuntimeError("An SFC already has an unfinished immediate transition")
        active.pending_decision = PendingDecision(
            observation=observation,
            action_index=int(action_index),
            reward=float(reward),
        )
        if terminal:
            self._finish_pending_decision(active, None, terminal=True)

    def _finish_pending_decision(
        self,
        active: ActiveSFC,
        next_observation: DecisionObservation | None,
        terminal: bool,
        reward_override: float | None = None,
    ) -> bool:
        pending = active.pending_decision
        if pending is None:
            return False
        active.pending_decision = None
        self._observe(Transition(
            observation=pending.observation,
            action_index=pending.action_index,
            reward=(
                float(reward_override)
                if reward_override is not None else pending.reward
            ),
            next_observation=None if terminal else next_observation,
            terminal=bool(terminal or next_observation is None),
        ))
        return True

    def _next_observation(self, active: ActiveSFC) -> DecisionObservation | None:
        if active.pending is not None or active.request.vehicle_id not in self.vehicles:
            return None
        access = self.topology.access_node(
            self.vehicles[active.request.vehicle_id].x_m,
            self.vehicles[active.request.vehicle_id].y_m,
        )
        if access is None:
            return None
        candidates = self.candidate_generator.build(
            active.request, active, access, self.time_s
        )
        if not candidates:
            return None
        pre_delay = self._delay_for(active.request, active.hosts, active.route)
        return self.state_history.build(
            active.request,
            active,
            candidates,
            self.time_s,
            pre_delay,
            self.candidate_generator.last_fallback_used,
            self.candidate_generator.last_paths_examined,
        )

    def _observe(self, transition: Transition) -> None:
        assert self.policy is not None
        metrics = self.policy.observe(transition)
        if self._in_measurement():
            self.rewards.append(float(transition.reward))
            if "loss" in metrics:
                self.losses.append(float(metrics["loss"]))

    def _resolve_normal(self, active: ActiveSFC, reason: str) -> None:
        request_id = active.request.request_id
        if active.pending is not None:
            self._cancel_pending(active, f"{reason}_during_preparation", terminal=True)
        self._finish_pending_decision(active, None, terminal=True)
        self._release_active_resources(active)
        active.resolved = True
        record = self.records[request_id]
        record.continuity_resolved = True
        record.continuity_satisfied = True
        record.departure_reason = reason
        self._sync_record(active)
        self.active.pop(request_id, None)

    def _continuity_failure(self, active: ActiveSFC, reason: str) -> None:
        request_id = active.request.request_id
        observed_failure = False
        if active.pending is not None:
            self._cancel_pending(active, reason, terminal=True)
            observed_failure = True
        if self._finish_pending_decision(
            active,
            None,
            terminal=True,
            reward_override=-1.0,
        ):
            observed_failure = True
        self._release_active_resources(active)
        active.resolved = True
        active.continuity_failed = True
        record = self.records[request_id]
        record.continuity_resolved = True
        record.continuity_failed = True
        record.continuity_satisfied = False
        record.departure_reason = reason
        self._sync_record(active)
        self.active.pop(request_id, None)
        if self._in_measurement() and not observed_failure:
            self.rewards.append(-1.0)

    def _release_active_resources(self, active: ActiveSFC) -> None:
        if active.pending is not None:
            self._release_pending_without_transition(active)
        cpu = self.topology.node_cpu_amounts(
            active.hosts, (vnf.cpu for vnf in active.request.vnfs)
        )
        self.topology.release_cpu(cpu)
        self.topology.release_service(
            self.topology.route_edges(active.route), active.request.bandwidth_mb_s
        )

    def _release_pending_without_transition(self, active: ActiveSFC) -> None:
        pending = active.pending
        if pending is None:
            return
        if pending.reserved_step_edges:
            self.topology.release_migration(
                pending.reserved_step_edges, pending.reserved_step_bandwidth_mb_s
            )
        cpu_release: dict[int, float] = defaultdict(float)
        for index in pending.plan.changed_vnfs:
            cpu_release[pending.plan.hosts[index]] += active.request.vnfs[index].cpu
        self.topology.release_cpu(dict(cpu_release))
        old_edges = set(self.topology.route_edges(pending.old_route))
        new_edges = set(self.topology.route_edges(pending.plan.route))
        self.topology.release_service(
            new_edges - old_edges, active.request.bandwidth_mb_s
        )
        active.pending = None

    def _sync_record(self, active: ActiveSFC) -> None:
        record = self.records[active.request.request_id]
        record.migration_volume_mb = active.transferred_mb
        record.completed_migration_volume_mb = active.migration_volume_mb
        record.canceled_transfer_mb = active.canceled_transfer_mb
        record.canceled_migrations = active.canceled_migrations
        record.ended_s = self.time_s
        record.observed_service_s = max(0.0, self.time_s - active.admitted_s)
        record.downtime_ms = active.downtime_ms
        record.migrations = active.migrations
        record.handoffs = active.handoffs
        record.route_updates = active.route_updates

    def _sample_active_delays(self) -> None:
        for active in self.active.values():
            delay = self._delay_for(active.request, active.hosts, active.route)
            active.last_delay_ms = delay
            self.delay_samples_ms.append(delay)
            record = self.records[active.request.request_id]
            if record.tracked:
                record.add_delay(delay)

    def _delay_for(self, request: SFCRequest, hosts: Iterable[int],
                   route: Iterable[int]) -> float:
        processing = sum(
            vnf.workload_mi / self.topology.nodes[host].processing_rate_mi_ms
            for vnf, host in zip(request.vnfs, hosts)
        )
        return (
            float(self.cfg["network"]["wireless_delay_ms"])
            + processing
            + self.topology.path_delay_ms(route)
        )

    def _pre_action_delay(
        self,
        request: SFCRequest,
        active: ActiveSFC | None,
        access_node: int,
    ) -> float:
        if active is None:
            return 0.0
        current_route = self.topology.shortest_ordered_route(
            access_node,
            active.hosts,
            required_bandwidth=request.bandwidth_mb_s,
            credit_edges=set(self.topology.route_edges(active.route)),
        )
        if current_route is None:
            # The placement is presently infeasible; use the threshold as the
            # bounded reference in Eq. (24) while candidate masking determines
            # whether a route update or remapping can restore service.
            return float(request.latency_requirement_ms)
        return self._delay_for(request, active.hosts, current_route)

    def _mark_active_at_end(self) -> None:
        for active in self.active.values():
            # Episode truncation is terminal for learning, although these
            # requests remain excluded from the paper's continuity denominator.
            self._finish_pending_decision(active, None, terminal=True)
            record = self.records[active.request.request_id]
            if record.tracked:
                record.still_active_at_end = True
                record.departure_reason = "active_at_measurement_end"
            self._sync_record(active)

    def _in_measurement(self) -> bool:
        return self.warmup_s <= self.time_s < self.measurement_end_s

    def _aggregate_results(self) -> dict:
        tracked = [record for record in self.records.values() if record.tracked]
        generated = len(tracked)
        admitted_records = [record for record in tracked if record.admitted]
        admitted = len(admitted_records)
        resolved = [record for record in admitted_records if record.continuity_resolved]
        satisfied = [record for record in resolved if record.continuity_satisfied]
        migration_total = sum(record.migration_volume_mb for record in admitted_records)
        downtime_total = sum(record.downtime_ms for record in admitted_records)
        migrations = sum(record.migrations for record in admitted_records)
        route_updates = sum(record.route_updates for record in admitted_records)
        return {
            "schema_version": 3,
            "seed": self.seed,
            "generated_requests": generated,
            "admitted_requests": admitted,
            "blocked_requests": generated - admitted,
            "acceptance_ratio_pct": _percentage(admitted, generated),
            "continuity_resolved_requests": len(resolved),
            "continuity_satisfied_requests": len(satisfied),
            "continuity_failed_requests": len(resolved) - len(satisfied),
            "active_excluded_requests": sum(record.still_active_at_end for record in admitted_records),
            "continuity_satisfaction_ratio_pct": _percentage(len(satisfied), len(resolved)),
            "migration_volume_total_mb": migration_total,
            "migration_mb_per_admitted_sfc": migration_total / admitted if admitted else math.nan,
            "downtime_ms_per_admitted_sfc": downtime_total / admitted if admitted else math.nan,
            "migration_count": migrations,
            "route_update_count": route_updates,
            "handoff_count": sum(record.handoffs for record in admitted_records),
            "mean_e2e_delay_ms": _mean(self.delay_samples_ms),
            "p95_e2e_delay_ms": _percentile(self.delay_samples_ms, 95),
            "mean_decision_runtime_ms": _mean(self.decision_times_ms),
            "p95_decision_runtime_ms": _percentile(self.decision_times_ms, 95),
            "mean_candidate_runtime_ms": _mean(self.candidate_times_ms),
            "mean_reference_route_runtime_ms": _mean(self.reference_route_times_ms),
            "mean_fallback_runtime_ms": _mean(self.fallback_times_ms),
            "mean_policy_runtime_ms": _mean(self.policy_times_ms),
            "mean_cpu_utilization_pct": 100.0 * _mean(self.cpu_utilization),
            "mean_bandwidth_utilization_pct": 100.0 * _mean(self.bandwidth_utilization),
            "mean_reward": _mean(self.rewards),
            "mean_training_loss": _mean(self.losses),
            "decision_count": self.total_decisions,
            "complete_fallback_count": self.fallback_decisions,
            "candidate_paths_examined": self.paths_examined,
            "assignment_nodes_examined": self.assignment_nodes_examined,
            "search_budget_exhausted_decisions": self.search_budget_exhausted_decisions,
            "search_budget_blocked_requests": sum(r.blocked and r.departure_reason == 'search_budget_exhausted' for r in tracked),
            "search_budget_continuity_failures": sum(r.continuity_failed and r.departure_reason == 'search_budget_exhausted' for r in tracked),
            "controller_time_model": "instantaneous_simulation_external_wall_clock",
            "vehicle_departure_requests": sum(r.departure_reason == "vehicle_departure" for r in resolved),
            "lifetime_expiry_requests": sum(r.departure_reason == "lifetime_expiry" for r in resolved),
            "coverage_gap_failures": sum(r.departure_reason == "coverage_gap" for r in resolved),
            "active_service_seconds": sum(r.observed_service_s for r in admitted_records),
            "failures_per_service_hour": (
                3600.0 * (len(resolved) - len(satisfied))
                / sum(r.observed_service_s for r in admitted_records)
                if sum(r.observed_service_s for r in admitted_records) > 0 else math.nan
            ),
            "completed_migration_volume_total_mb": sum(r.completed_migration_volume_mb for r in admitted_records),
            "canceled_transfer_total_mb": sum(r.canceled_transfer_mb for r in admitted_records),
            "canceled_migration_count": sum(r.canceled_migrations for r in admitted_records),
            "peak_temporary_cpu": self.peak_temporary_cpu,
            "temporary_cpu_seconds": self.temporary_cpu_seconds,
            "peak_temporary_service_bandwidth_mb_s": self.peak_temporary_service_bandwidth_mb_s,
            "temporary_service_bandwidth_mb": self.temporary_service_bandwidth_mb,
            "peak_migration_bandwidth_mb_s": self.peak_migration_bandwidth_mb_s,
            "migration_bandwidth_reservation_mb": self.migration_bandwidth_reservation_mb,
            "peak_pending_migrations": self.peak_pending_migrations,
            "blocked_recoverable_without_temporary_reservations": self.blocked_without_temporary_reservations,
        }

    def _temporary_allocations(self):
        cpu, service, migration = defaultdict(float), defaultdict(float), defaultdict(float)
        pending_count = 0
        for active in self.active.values():
            pending = active.pending
            if pending is None:
                continue
            pending_count += 1
            for index in pending.plan.changed_vnfs:
                cpu[pending.plan.hosts[index]] += active.request.vnfs[index].cpu
            old = set(self.topology.route_edges(pending.old_route))
            for edge in set(self.topology.route_edges(pending.plan.route)) - old:
                service[edge] += active.request.bandwidth_mb_s
            for edge in pending.reserved_step_edges:
                migration[edge] += pending.reserved_step_bandwidth_mb_s
        return cpu, service, migration, pending_count

    def _capture_reservation_peaks(self):
        if not self._in_measurement():
            return
        cpu, service, migration, count = self._temporary_allocations()
        self.peak_temporary_cpu = max(self.peak_temporary_cpu, sum(cpu.values()))
        self.peak_temporary_service_bandwidth_mb_s = max(self.peak_temporary_service_bandwidth_mb_s, sum(service.values()))
        self.peak_migration_bandwidth_mb_s = max(self.peak_migration_bandwidth_mb_s, sum(migration.values()))
        self.peak_pending_migrations = max(self.peak_pending_migrations, count)

    def _integrate_reservations(self, elapsed_s):
        duration = max(0.0, min(self.time_s, self.measurement_end_s) - max(self.warmup_s, self.time_s - elapsed_s))
        cpu, service, migration, _ = self._temporary_allocations()
        self.temporary_cpu_seconds += sum(cpu.values()) * duration
        self.temporary_service_bandwidth_mb += sum(service.values()) * duration
        self.migration_bandwidth_reservation_mb += sum(migration.values()) * duration

    def _audit_temporary_blocking(self, request, access):
        cpu, service, migration, count = self._temporary_allocations()
        if not count:
            return
        # Offline counterfactual only. Never releases the live source or
        # destination allocations and is excluded from decision-runtime data.
        topology = deepcopy(self.topology)
        topology.release_cpu(dict(cpu))
        for edge, amount in service.items():
            topology.release_service([edge], amount)
        for edge, amount in migration.items():
            topology.release_migration([edge], amount)
        if CandidateGenerator(self.cfg, topology).build(request, None, access, self.time_s):
            self.blocked_without_temporary_reservations += 1

    def _request_rows(self) -> list[dict]:
        rows = []
        for record in sorted(self.records.values(), key=lambda item: item.request_id):
            if not bool(self.cfg["simulation"].get("record_requests", True)):
                break
            rows.append(vars(record).copy())
        return rows


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else math.nan


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan
