from __future__ import annotations

import itertools
import time
from dataclasses import replace

from .topology import Topology
from .types import ActiveSFC, MigrationStep, PlacementPlan, SFCRequest


class CandidateGenerator:
    """Ordered candidates, full feasibility screening and explicit work budgets."""

    def __init__(self, cfg: dict, topology: Topology):
        self.cfg = cfg
        self.topology = topology
        self.candidate_cfg = cfg["candidates"]
        self.network_cfg = cfg["network"]
        self.migration_cfg = cfg["migration"]
        self.last_fallback_used = False
        self.last_paths_examined = 0
        self.last_fallback_runtime_ms = 0.0
        self.last_search_budget_exhausted = False
        self.last_assignment_nodes = 0

    def build(self, request: SFCRequest, active: ActiveSFC | None,
              access_node: int, time_s: float) -> list[PlacementPlan]:
        self.last_fallback_used = False
        self.last_paths_examined = 0
        self.last_fallback_runtime_ms = 0.0
        self.last_search_budget_exhausted = False
        self.last_assignment_nodes = 0
        maximum_actions = int(self.candidate_cfg["maximum_actions"])
        feasible: list[PlacementPlan] = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

        if active is not None:
            retain_route = self.topology.shortest_ordered_route(
                access_node,
                active.hosts,
                required_bandwidth=request.bandwidth_mb_s,
                credit_edges=set(self.topology.route_edges(active.route)),
            )
            if retain_route is not None:
                plan = self._screen(request, active, access_node, retain_route,
                                    active.hosts, time_s)
                if plan is not None:
                    feasible.append(plan)
                    seen.add((plan.hosts, plan.route))

        if active is None and sum(v.cpu for v in request.vnfs) > sum(
            node.residual_cpu for node in self.topology.nodes.values()
        ) + 1e-9:
            return []
        if active is not None and not feasible and self._all_remaps_expire(request, time_s):
            # Shortest ordered RETAIN already failed. If even the smallest
            # single-VNF transfer cannot finish, no remap can rescue it.
            # This proves infeasibility without enumerating every simple path.
            return []

        endpoints = self.topology.nodes_within_hops(
            access_node, int(self.candidate_cfg["hop_cutoff"])
        )
        credit_edges = (
            set(self.topology.route_edges(active.route)) if active is not None else set()
        )
        for endpoint in endpoints:
            paths = self.topology.k_shortest_simple_paths(
                access_node,
                endpoint,
                k=int(self.candidate_cfg["paths_per_endpoint"]),
                max_hops=int(self.candidate_cfg["hop_cutoff"]),
                required_bandwidth=request.bandwidth_mb_s,
                credit_edges=credit_edges,
                propagation_cutoff_ms=request.latency_requirement_ms,
            )
            for route in paths:
                self.last_paths_examined += 1
                self._screen_assignments(
                    request, active, access_node, route, time_s, feasible, seen
                )
                if self.last_search_budget_exhausted:
                    break
            if self.last_search_budget_exhausted:
                break

        materialized = self._materialize(feasible, request, active, access_node, time_s)
        if (not materialized and not self.last_search_budget_exhausted and
                bool(self.candidate_cfg.get("complete_fallback", True))):
            fallback_start = time.perf_counter_ns()
            self.last_fallback_used = True
            processing_lower_bound = sum(
                vnf.workload_mi
                / max(node.processing_rate_mi_ms for node in self.topology.nodes.values())
                for vnf in request.vnfs
            )
            propagation_cutoff = (
                request.latency_requirement_ms
                - float(self.network_cfg["wireless_delay_ms"])
                - processing_lower_bound
            )
            if propagation_cutoff >= 0:
                fallback_paths = 0
                limit = self.candidate_cfg.get('maximum_fallback_paths')
                for route in self.topology.all_simple_paths_by_delay(
                    access_node, propagation_cutoff_ms=propagation_cutoff,
                    required_bandwidth=request.bandwidth_mb_s, credit_edges=credit_edges,
                ):
                    if limit is not None and fallback_paths >= int(limit):
                        self.last_search_budget_exhausted = True
                        break
                    fallback_paths += 1
                    self.last_paths_examined += 1
                    route_plans = []
                    self._screen_assignments(
                        request, active, access_node, route, time_s, route_plans, seen
                    )
                    # Preliminary service feasibility is insufficient: all
                    # remaps may fail transfer-bandwidth or lifetime checks.
                    materialized.extend(self._materialize(
                        route_plans, request, active, access_node, time_s
                    ))
                    if len(materialized) >= maximum_actions or self.last_search_budget_exhausted:
                        break
            self.last_fallback_runtime_ms = (time.perf_counter_ns() - fallback_start) / 1e6

        return self._rank_and_bound(materialized, active, maximum_actions)

    def _materialize(self, feasible, request, active, access_node, time_s):
        maximum_actions = int(self.candidate_cfg["maximum_actions"])

        # Rank cheap service-feasible plans first.  Migration paths are the
        # expensive part of candidate construction, so materialize them only
        # for ranked plans until Mmax feasible actions have been obtained.
        ranked = self._rank_and_bound(feasible, active, len(feasible))
        if active is None:
            return ranked[:maximum_actions]
        materialized: list[PlacementPlan] = []
        for plan in ranked:
            if plan.changed_vnfs:
                plan = self._screen(
                    request,
                    active,
                    access_node,
                    plan.route,
                    plan.hosts,
                    time_s,
                    include_migration=True,
                )
                if plan is None:
                    continue
            materialized.append(plan)
            if len(materialized) >= maximum_actions:
                break
        return materialized

    def _screen_assignments(
        self,
        request: SFCRequest,
        active: ActiveSFC | None,
        access_node: int,
        route: tuple[int, ...],
        time_s: float,
        feasible: list[PlacementPlan],
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        for hosts in self._pruned_placements(request, active, route, time_s):
            key = (hosts, route)
            if key in seen:
                continue
            seen.add(key)
            plan = self._screen(
                request,
                active,
                access_node,
                route,
                hosts,
                time_s,
                include_migration=False,
            )
            if plan is not None:
                feasible.append(plan)

    def _pruned_placements(self, request, active, route, time_s):
        """Same path-order enumeration, pruning only necessary conditions.

        Nonnegative partial CPU, processing and minimum transfer costs cannot
        decrease after assigning additional VNFs. The optional shared work
        budget counts partial placement trials and reports truncation explicitly.
        """
        if not route:
            return
        old_edges = set(self.topology.route_edges(active.route)) if active else set()
        if any(self.topology.links[edge].residual_mb_s + 1e-9 < request.bandwidth_mb_s
               for edge in self.topology.route_edges(route) if edge not in old_edges):
            return
        delay_budget = (request.latency_requirement_ms - self.network_cfg['wireless_delay_ms']
                        - self.topology.path_delay_ms(route))
        migration_impossible = active is not None and self._all_remaps_expire(request, time_s)
        allocations = {}
        hosts = []

        def visit(index, start, processing_ms):
            if index == len(request.vnfs):
                yield tuple(hosts)
                return
            vnf = request.vnfs[index]
            positions = [len(route)-1] if index == len(request.vnfs)-1 else range(start, len(route))
            for position in positions:
                limit = self.candidate_cfg.get('maximum_assignment_nodes')
                if limit is not None and self.last_assignment_nodes >= int(limit):
                    self.last_search_budget_exhausted = True
                    return
                self.last_assignment_nodes += 1
                host = route[position]
                node = self.topology.nodes[host]
                moving = active is not None and host != active.hosts[index]
                if moving and migration_impossible:
                    continue
                addition = vnf.cpu if active is None or moving else 0.0
                allocated = allocations.get(host, 0.0)
                if allocated + addition > node.residual_cpu + 1e-9:
                    continue
                processing = processing_ms + vnf.workload_mi / node.processing_rate_mi_ms
                if processing > delay_budget + 1e-9:
                    continue
                allocations[host] = allocated + addition
                hosts.append(host)
                yield from visit(index+1, position, processing)
                hosts.pop()
                allocations[host] = allocated
                if self.last_search_budget_exhausted:
                    return
        yield from visit(0, 0, 0.0)

    def _all_remaps_expire(self, request, time_s):
        bandwidth = float(self.migration_cfg['maximum_bandwidth_mb_s'])
        if bandwidth <= 0:
            return True
        minimum_transfer_ms = 1000.0 * min(
            v.image_mb + v.state_mb for v in request.vnfs
        ) / bandwidth
        return (minimum_transfer_ms + self.migration_cfg['cutover_downtime_ms']
                >= max(0.0, request.expiry_s - time_s) * 1000.0)

    @staticmethod
    def _placements_on_route(route: tuple[int, ...], length: int):
        if not route:
            return
        endpoint_index = len(route) - 1
        for prefix in itertools.combinations_with_replacement(
            range(len(route)), max(0, length - 1)
        ):
            positions = prefix + (endpoint_index,)
            yield tuple(route[index] for index in positions)

    def _screen(
        self,
        request: SFCRequest,
        active: ActiveSFC | None,
        access_node: int,
        route: tuple[int, ...],
        hosts: tuple[int, ...],
        time_s: float,
        include_migration: bool = True,
    ) -> PlacementPlan | None:
        if len(route) != len(set(route)) or not self._hosts_follow_route(hosts, route):
            return None
        route_edges = set(self.topology.route_edges(route))
        old_edges = set(self.topology.route_edges(active.route)) if active else set()
        new_only_edges = route_edges - old_edges
        service_edges_to_check = route_edges if active is None else new_only_edges
        for edge in service_edges_to_check:
            if self.topology.links[edge].residual_mb_s + 1e-9 < request.bandwidth_mb_s:
                return None

        changed = (
            () if active is None
            else tuple(
                index for index, host in enumerate(hosts)
                if host != active.hosts[index]
            )
        )
        cpu_additions: dict[int, float] = {}
        for index, host in enumerate(hosts):
            if active is None or index in changed:
                cpu_additions[host] = cpu_additions.get(host, 0.0) + request.vnfs[index].cpu
        for node_id, amount in cpu_additions.items():
            if self.topology.nodes[node_id].residual_cpu + 1e-9 < amount:
                return None

        processing_delay = sum(
            request.vnfs[index].workload_mi
            / self.topology.nodes[host].processing_rate_mi_ms
            for index, host in enumerate(hosts)
        )
        delay_ms = (
            float(self.network_cfg["wireless_delay_ms"])
            + processing_delay
            + self.topology.path_delay_ms(route)
        )
        if delay_ms > request.latency_requirement_ms + 1e-9:
            return None

        service_adjustment = {
            edge: request.bandwidth_mb_s for edge in new_only_edges
        }
        migration_steps: list[MigrationStep] = []
        if active is not None and include_migration:
            for index in changed:
                source, destination = active.hosts[index], hosts[index]
                migration_path = self.topology.shortest_migration_path(
                    source, destination, service_adjustment
                )
                if migration_path is None:
                    return None
                migration_edges = self.topology.route_edges(migration_path)
                residuals = [
                    self.topology.links[edge].residual_mb_s
                    - service_adjustment.get(edge, 0.0)
                    for edge in migration_edges
                ]
                if not residuals or min(residuals) <= 1e-12:
                    return None
                bandwidth = min(
                    float(self.migration_cfg["maximum_bandwidth_mb_s"]),
                    min(residuals),
                )
                vnf = request.vnfs[index]
                volume = vnf.image_mb + vnf.state_mb
                transfer_ms = (
                    1000.0 * volume / bandwidth
                    + self.topology.path_delay_ms(migration_path)
                )
                migration_steps.append(MigrationStep(
                    vnf_index=index,
                    source=source,
                    destination=destination,
                    path=migration_path,
                    bandwidth_mb_s=bandwidth,
                    transfer_time_ms=transfer_ms,
                    volume_mb=volume,
                ))

        migration_volume = (
            sum(step.volume_mb for step in migration_steps)
            if include_migration
            else sum(
                request.vnfs[index].image_mb + request.vnfs[index].state_mb
                for index in changed
            )
        )
        preparation_time = sum(step.transfer_time_ms for step in migration_steps)
        downtime = (
            float(self.migration_cfg["cutover_downtime_ms"])
            if migration_steps else 0.0
        )
        remaining_lifetime_ms = max(0.0, request.expiry_s - time_s) * 1000.0
        if migration_steps and preparation_time + downtime >= remaining_lifetime_ms:
            return None

        cpu_ratios = []
        for host in set(hosts):
            node = self.topology.nodes[host]
            cpu_ratios.append(
                max(0.0, node.residual_cpu - cpu_additions.get(host, 0.0))
                / node.cpu_capacity
            )
        bandwidth_ratios = []
        for edge in route_edges:
            link = self.topology.links[edge]
            extra = request.bandwidth_mb_s if edge in service_edges_to_check else 0.0
            bandwidth_ratios.append(max(0.0, link.residual_mb_s - extra) / link.capacity_mb_s)

        if active is None:
            kind = "initial"
        elif changed:
            kind = "remap"
        elif route == active.route:
            kind = "retain"
        else:
            kind = "route_update"
        return PlacementPlan(
            hosts=hosts,
            route=route,
            access_node=access_node,
            delay_ms=float(delay_ms),
            migration_volume_mb=float(migration_volume),
            preparation_time_ms=float(preparation_time),
            downtime_ms=float(downtime),
            changed_vnfs=changed,
            kind=kind,
            minimum_cpu_residual_ratio=min(cpu_ratios) if cpu_ratios else 1.0,
            minimum_bandwidth_residual_ratio=(
                min(bandwidth_ratios) if bandwidth_ratios else 1.0
            ),
            migration_steps=tuple(migration_steps),
        )

    @staticmethod
    def _hosts_follow_route(hosts: tuple[int, ...], route: tuple[int, ...]) -> bool:
        position = 0
        for host in hosts:
            try:
                index = route.index(host, position)
            except ValueError:
                return False
            position = index
        return True

    def _rank_and_bound(self, plans: list[PlacementPlan], active: ActiveSFC | None,
                        maximum_actions: int) -> list[PlacementPlan]:
        if not plans:
            return []
        # Final de-duplication protects RETAIN from an equivalent path candidate.
        unique: dict[tuple[tuple[int, ...], tuple[int, ...]], PlacementPlan] = {}
        for plan in plans:
            key = (plan.hosts, plan.route)
            existing = unique.get(key)
            if existing is None or plan.delay_ms < existing.delay_ms:
                unique[key] = plan
        plans = list(unique.values())
        if active is None:
            plans.sort(key=lambda plan: (
                plan.delay_ms,
                -plan.minimum_cpu_residual_ratio,
                -plan.minimum_bandwidth_residual_ratio,
                plan.hosts,
            ))
            return plans[:maximum_actions]

        retained = [plan for plan in plans if plan.kind in {"retain", "route_update"}]
        others = [plan for plan in plans if plan not in retained]
        # Two-objective Pareto partition in O(P log P), where P is the number
        # of cheap service-feasible plans.  The former all-pairs dominance test
        # was O(P^2) and dominated the 48/96-RSU scalability experiments.
        nondominated: list[PlacementPlan] = []
        dominated: list[PlacementPlan] = []
        pareto_order = sorted(
            others,
            key=lambda plan: (
                plan.delay_ms,
                plan.migration_volume_mb,
                plan.hosts,
                plan.route,
            ),
        )
        best_volume_at_lower_delay = float("inf")
        group_start = 0
        while group_start < len(pareto_order):
            delay = pareto_order[group_start].delay_ms
            group_end = group_start + 1
            while (
                group_end < len(pareto_order)
                and pareto_order[group_end].delay_ms == delay
            ):
                group_end += 1
            group = pareto_order[group_start:group_end]
            group_minimum_volume = min(plan.migration_volume_mb for plan in group)
            for candidate in group:
                dominated_by_lower_delay = (
                    best_volume_at_lower_delay
                    <= candidate.migration_volume_mb
                )
                dominated_at_equal_delay = (
                    group_minimum_volume
                    < candidate.migration_volume_mb
                )
                (
                    dominated
                    if dominated_by_lower_delay or dominated_at_equal_delay
                    else nondominated
                ).append(candidate)
            best_volume_at_lower_delay = min(
                best_volume_at_lower_delay,
                group_minimum_volume,
            )
            group_start = group_end
        ordering = lambda plan: (plan.delay_ms, plan.migration_volume_mb, plan.hosts)
        retained.sort(key=ordering)
        nondominated.sort(key=ordering)
        dominated.sort(key=ordering)
        ranked: list[PlacementPlan] = []
        if bool(self.candidate_cfg.get("preserve_retain", True)) and retained:
            ranked.append(retained[0])
            retained = retained[1:]
        ranked.extend(nondominated)
        ranked.extend(retained)
        ranked.extend(dominated)
        return ranked[:maximum_actions]


def plan_with_kind(plan: PlacementPlan, kind: str) -> PlacementPlan:
    """Small test/helper hook used by ablation policies."""
    return replace(plan, kind=kind)
