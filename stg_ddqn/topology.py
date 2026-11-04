from __future__ import annotations

import csv
import heapq
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class FogNode:
    node_id: int
    x_m: float
    y_m: float
    cpu_capacity: float
    processing_rate_mi_ms: float
    coverage_radius_m: float
    cpu_used: float = 0.0

    @property
    def residual_cpu(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used)

    @property
    def residual_cpu_ratio(self) -> float:
        return self.residual_cpu / self.cpu_capacity


@dataclass
class DirectedLink:
    source: int
    destination: int
    capacity_mb_s: float
    delay_ms: float
    service_used_mb_s: float = 0.0
    migration_used_mb_s: float = 0.0

    @property
    def residual_mb_s(self) -> float:
        return max(
            0.0,
            self.capacity_mb_s - self.service_used_mb_s - self.migration_used_mb_s,
        )

    @property
    def residual_ratio(self) -> float:
        return self.residual_mb_s / self.capacity_mb_s


class Topology:
    """Directed fog graph with explicit service and migration reservations."""

    def __init__(self, cfg: dict, structure_rng: np.random.Generator,
                 resource_rng: np.random.Generator):
        self.cfg = cfg
        self.area_m = tuple(float(x) for x in cfg["area_m"])
        self.nodes: dict[int, FogNode] = {}
        self.links: dict[tuple[int, int], DirectedLink] = {}
        self.adjacency: dict[int, list[int]] = {}
        if cfg.get("topology_nodes_csv") and cfg.get("topology_links_csv"):
            self._load_csv(
                Path(cfg["topology_nodes_csv"]),
                Path(cfg["topology_links_csv"]),
            )
        else:
            self._generate(structure_rng, resource_rng)
        self._finalize()

    def _generate(self, structure_rng: np.random.Generator,
                  resource_rng: np.random.Generator) -> None:
        count = int(self.cfg["nodes"])
        physical_links = int(self.cfg["physical_links"])
        if physical_links < count - 1:
            raise ValueError("A connected topology needs at least nodes-1 physical links")
        width, height = self.area_m
        columns = math.ceil(math.sqrt(count * width / height))
        rows = math.ceil(count / columns)
        coords: list[tuple[float, float]] = []
        for row in range(rows):
            for column in range(columns):
                if len(coords) == count:
                    break
                x = width * (column + 0.5) / columns
                y = height * (row + 0.5) / rows
                coords.append((x, y))

        cpu_low, cpu_high = self.cfg["cpu_capacity"]
        rate_low, rate_high = self.cfg["processing_rate_mi_per_ms"]
        radius = float(self.cfg["coverage_radius_m"])
        for node_id, (x_m, y_m) in enumerate(coords):
            self.nodes[node_id] = FogNode(
                node_id=node_id,
                x_m=x_m,
                y_m=y_m,
                cpu_capacity=float(resource_rng.uniform(cpu_low, cpu_high)),
                processing_rate_mi_ms=float(resource_rng.uniform(rate_low, rate_high)),
                coverage_radius_m=radius,
            )

        pairs = sorted(
            (math.dist(coords[u], coords[v]), float(structure_rng.random()), u, v)
            for u in range(count)
            for v in range(u + 1, count)
        )
        parent = list(range(count))

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        selected: list[tuple[int, int]] = []
        selected_set: set[tuple[int, int]] = set()
        for _, _, u, v in pairs:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v
                selected.append((u, v))
                selected_set.add((u, v))
        for _, _, u, v in pairs:
            if len(selected) >= physical_links:
                break
            if (u, v) not in selected_set:
                selected.append((u, v))
                selected_set.add((u, v))

        bw_low, bw_high = self.cfg["link_bandwidth_mb_s"]
        delay_low, delay_high = self.cfg["propagation_delay_ms"]
        for u, v in selected:
            capacity = float(resource_rng.uniform(bw_low, bw_high))
            delay = float(resource_rng.uniform(delay_low, delay_high))
            self.links[(u, v)] = DirectedLink(u, v, capacity, delay)
            self.links[(v, u)] = DirectedLink(v, u, capacity, delay)

    def _load_csv(self, nodes_path: Path, links_path: Path) -> None:
        with nodes_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                node_id = int(row["node_id"])
                self.nodes[node_id] = FogNode(
                    node_id=node_id,
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    cpu_capacity=float(row["cpu_capacity"]),
                    processing_rate_mi_ms=float(row["processing_rate_mi_ms"]),
                    coverage_radius_m=float(
                        row.get("coverage_radius_m") or self.cfg["coverage_radius_m"]
                    ),
                )
        with links_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                u, v = int(row["source"]), int(row["destination"])
                capacity = float(row["capacity_mb_s"])
                delay = float(row["delay_ms"])
                self.links[(u, v)] = DirectedLink(u, v, capacity, delay)
                bidirectional = str(row.get("bidirectional", "true")).lower()
                if bidirectional in {"1", "true", "yes"} and (v, u) not in self.links:
                    self.links[(v, u)] = DirectedLink(v, u, capacity, delay)

    def _finalize(self) -> None:
        if sorted(self.nodes) != list(range(len(self.nodes))):
            raise ValueError("Node IDs must be contiguous and start at zero")
        self.adjacency = {node_id: [] for node_id in self.nodes}
        for u, v in sorted(self.links):
            if u not in self.nodes or v not in self.nodes:
                raise ValueError(f"Link {(u, v)} refers to an unknown node")
            self.adjacency[u].append(v)
        for values in self.adjacency.values():
            values.sort()
        if not self._all_reachable_from(0):
            raise ValueError("Generated/loaded directed topology is not reachable from node 0")
        self.directed_edges = tuple(sorted(self.links))
        model_edges = list(self.directed_edges) + [(n, n) for n in sorted(self.nodes)]
        self.model_edges = tuple(model_edges)
        self.edge_index = np.asarray(model_edges, dtype=np.int64).T

    def _all_reachable_from(self, source: int) -> bool:
        seen = {source}
        stack = [source]
        while stack:
            current = stack.pop()
            for neighbor in self.adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(self.nodes)

    @staticmethod
    def route_edges(route: Iterable[int]) -> tuple[tuple[int, int], ...]:
        nodes = tuple(route)
        return tuple(zip(nodes[:-1], nodes[1:]))

    def path_delay_ms(self, route: Iterable[int]) -> float:
        return sum(self.links[edge].delay_ms for edge in self.route_edges(route))

    def access_node(self, x_m: float, y_m: float) -> int | None:
        covered = [
            (math.hypot(x_m - node.x_m, y_m - node.y_m), node.node_id)
            for node in self.nodes.values()
            if math.hypot(x_m - node.x_m, y_m - node.y_m) <= node.coverage_radius_m
        ]
        return min(covered)[1] if covered else None

    def nodes_within_hops(self, source: int, hop_cutoff: int) -> list[int]:
        distance = {source: 0}
        queue = [source]
        for current in queue:
            if distance[current] >= hop_cutoff:
                continue
            for neighbor in self.adjacency[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        return sorted(distance, key=lambda node: (distance[node], node))

    def k_shortest_simple_paths(self, source: int, target: int, k: int,
                                max_hops: int | None = None,
                                required_bandwidth: float = 0.0,
                                credit_edges: set[tuple[int, int]] | None = None,
                                residual_adjustment: dict[tuple[int, int], float] | None = None,
                                propagation_cutoff_ms: float | None = None) -> list[tuple[int, ...]]:
        if source == target:
            return [(source,)]
        credit_edges = credit_edges or set()
        residual_adjustment = residual_adjustment or {}
        counter = itertools.count()
        heap: list[tuple[float, int, tuple[int, ...]]] = [(0.0, next(counter), (source,))]
        results: list[tuple[int, ...]] = []
        while heap and len(results) < k:
            cost, _, path = heapq.heappop(heap)
            current = path[-1]
            if current == target:
                results.append(path)
                continue
            if max_hops is not None and len(path) - 1 >= max_hops:
                continue
            for neighbor in self.adjacency[current]:
                if neighbor in path:
                    continue
                edge = (current, neighbor)
                if edge not in credit_edges:
                    residual = self.links[edge].residual_mb_s - residual_adjustment.get(edge, 0.0)
                    if residual + 1e-9 < required_bandwidth:
                        continue
                next_cost = cost + self.links[edge].delay_ms
                if propagation_cutoff_ms is not None and next_cost > propagation_cutoff_ms:
                    continue
                heapq.heappush(heap, (next_cost, next(counter), path + (neighbor,)))
        return results

    def all_simple_paths_by_delay(self, source: int,
                                  propagation_cutoff_ms: float | None = None,
                                  required_bandwidth: float = 0.0,
                                  credit_edges: set[tuple[int, int]] | None = None,
                                  ) -> Iterator[tuple[int, ...]]:
        credit_edges = credit_edges or set()
        counter = itertools.count()
        heap: list[tuple[float, int, tuple[int, ...]]] = [(0.0, next(counter), (source,))]
        while heap:
            cost, _, path = heapq.heappop(heap)
            yield path
            current = path[-1]
            for neighbor in self.adjacency[current]:
                if neighbor in path:
                    continue
                edge = (current, neighbor)
                if edge not in credit_edges and self.links[edge].residual_mb_s + 1e-9 < required_bandwidth:
                    continue
                next_cost = cost + self.links[(current, neighbor)].delay_ms
                if propagation_cutoff_ms is not None and next_cost > propagation_cutoff_ms:
                    continue
                heapq.heappush(heap, (next_cost, next(counter), path + (neighbor,)))

    def shortest_ordered_route(
        self,
        access_node: int,
        hosts: Iterable[int],
        required_bandwidth: float = 0.0,
        credit_edges: set[tuple[int, int]] | None = None,
    ) -> tuple[int, ...] | None:
        targets = tuple(hosts)
        # Consecutive VNFs may share one host. Returning to a previously
        # consumed host, however, would violate the loop-free service route.
        distinct_targets = tuple(key for key, _ in itertools.groupby(targets))
        if len(distinct_targets) != len(set(distinct_targets)):
            return None
        if access_node in distinct_targets and distinct_targets[0] != access_node:
            # At a downstream host already: visiting earlier hosts and then
            # returning to source cannot be a simple route. This exact proof
            # avoids enumerating every simple prefix in a large fog graph.
            return None
        targets = distinct_targets
        credit_edges = credit_edges or set()
        reverse = {node: [] for node in self.nodes}
        for edge, link in self.links.items():
            if edge in credit_edges or link.residual_mb_s + 1e-9 >= required_bandwidth:
                reverse[edge[1]].append((edge[0], link.delay_ms))

        def remaining_cost(path, index):
            """Admissible relaxed segment costs; unreachable segments prove failure.

            Known visited nodes and out-of-order target hosts are forbidden.
            Segment interiors are relaxed to overlap, so the sum never exceeds
            the cost of a feasible loop-free completion.
            """
            current = path[-1]
            while index < len(targets) and targets[index] == current:
                index += 1
            value = 0.0
            for step in range(index, len(targets)):
                goal = targets[step]
                origin = current if step == index else targets[step - 1]
                forbidden = ((set(path) - {origin}) | set(targets[:step]) |
                             set(targets[step + 1:])) - {origin, goal}
                distances = {goal: 0.0}
                queue = [(0.0, goal)]
                found = False
                while queue:
                    distance, node = heapq.heappop(queue)
                    if distance > distances[node]:
                        continue
                    if node == origin:
                        value += distance
                        found = True
                        break
                    for predecessor, delay in reverse[node]:
                        if predecessor in forbidden:
                            continue
                        candidate = distance + delay
                        if candidate < distances.get(predecessor, float('inf')):
                            distances[predecessor] = candidate
                            heapq.heappush(queue, (candidate, predecessor))
                if not found:
                    return float('inf')
            return value

        counter = itertools.count()
        initial = remaining_cost((access_node,), 0)
        if not math.isfinite(initial):
            return None
        heap = [(initial, next(counter), 0.0, (access_node,), 0)]
        while heap:
            _, _, cost, path, target_index = heapq.heappop(heap)
            current = path[-1]
            while target_index < len(targets) and targets[target_index] == current:
                target_index += 1
            if target_index == len(targets):
                return path
            current_target = targets[target_index]
            future_targets = set(targets[target_index + 1:])
            for neighbor in self.adjacency[current]:
                if neighbor in path:
                    continue
                if neighbor in future_targets and neighbor != current_target:
                    continue
                edge = (current, neighbor)
                if edge not in credit_edges and self.links[edge].residual_mb_s + 1e-9 < required_bandwidth:
                    continue
                next_cost = cost + self.links[edge].delay_ms
                next_path = path + (neighbor,)
                bound = remaining_cost(next_path, target_index)
                if math.isfinite(bound):
                    heapq.heappush(heap, (next_cost + bound, next(counter),
                                         next_cost, next_path, target_index))
        return None

    def shortest_migration_path(
        self,
        source: int,
        destination: int,
        service_adjustment: dict[tuple[int, int], float] | None = None,
    ) -> tuple[int, ...] | None:
        if source == destination:
            return (source,)
        adjustment = service_adjustment or {}
        # Only one shortest path is needed. Enumerating simple paths can
        # exhaust exponentially many routes when the destination is cut off.
        # Dijkstra is complete for this nonnegative, capacity-filtered graph.
        counter = itertools.count()
        best = {source: 0.0}
        heap = [(0.0, next(counter), source, (source,))]
        while heap:
            cost, _, current, path = heapq.heappop(heap)
            if cost > best[current]:
                continue
            if current == destination:
                return path
            for neighbor in self.adjacency[current]:
                edge = (current, neighbor)
                link = self.links[edge]
                if link.residual_mb_s - adjustment.get(edge, 0.0) <= 1e-12:
                    continue
                next_cost = cost + link.delay_ms
                if next_cost < best.get(neighbor, float('inf')):
                    best[neighbor] = next_cost
                    heapq.heappush(heap, (next_cost, next(counter), neighbor, path + (neighbor,)))
        return None

    def reserve_cpu(self, amounts: dict[int, float]) -> None:
        for node_id, amount in amounts.items():
            if self.nodes[node_id].residual_cpu + 1e-9 < amount:
                raise RuntimeError(f"CPU over-reservation on node {node_id}")
        for node_id, amount in amounts.items():
            self.nodes[node_id].cpu_used += amount

    def release_cpu(self, amounts: dict[int, float]) -> None:
        for node_id, amount in amounts.items():
            node = self.nodes[node_id]
            node.cpu_used = max(0.0, node.cpu_used - amount)

    def reserve_service(self, edges: Iterable[tuple[int, int]], bandwidth_mb_s: float) -> None:
        edge_tuple = tuple(edges)
        for edge in edge_tuple:
            if self.links[edge].residual_mb_s + 1e-9 < bandwidth_mb_s:
                raise RuntimeError(f"Service-bandwidth over-reservation on edge {edge}")
        for edge in edge_tuple:
            self.links[edge].service_used_mb_s += bandwidth_mb_s

    def release_service(self, edges: Iterable[tuple[int, int]], bandwidth_mb_s: float) -> None:
        for edge in tuple(edges):
            link = self.links[edge]
            link.service_used_mb_s = max(0.0, link.service_used_mb_s - bandwidth_mb_s)

    def reserve_migration(self, edges: Iterable[tuple[int, int]], bandwidth_mb_s: float) -> None:
        edge_tuple = tuple(edges)
        for edge in edge_tuple:
            if self.links[edge].residual_mb_s + 1e-9 < bandwidth_mb_s:
                raise RuntimeError(f"Migration-bandwidth over-reservation on edge {edge}")
        for edge in edge_tuple:
            self.links[edge].migration_used_mb_s += bandwidth_mb_s

    def release_migration(self, edges: Iterable[tuple[int, int]], bandwidth_mb_s: float) -> None:
        for edge in tuple(edges):
            link = self.links[edge]
            link.migration_used_mb_s = max(0.0, link.migration_used_mb_s - bandwidth_mb_s)

    def node_cpu_amounts(self, hosts: Iterable[int], cpu_values: Iterable[float]) -> dict[int, float]:
        amounts: dict[int, float] = {}
        for host, cpu in zip(hosts, cpu_values):
            amounts[host] = amounts.get(host, 0.0) + float(cpu)
        return amounts

    def resource_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        node_ratios = np.asarray(
            [self.nodes[node].residual_cpu_ratio for node in sorted(self.nodes)],
            dtype=np.float32,
        )
        edge_features = []
        max_delay = max(link.delay_ms for link in self.links.values())
        for edge in self.model_edges:
            if edge[0] == edge[1]:
                edge_features.append((1.0, 0.0))
            else:
                link = self.links[edge]
                edge_features.append((link.residual_ratio, link.delay_ms / max_delay))
        return node_ratios, np.asarray(edge_features, dtype=np.float32)

    def utilization(self) -> tuple[float, float]:
        cpu = np.mean([node.cpu_used / node.cpu_capacity for node in self.nodes.values()])
        bandwidth = np.mean([
            (link.service_used_mb_s + link.migration_used_mb_s) / link.capacity_mb_s
            for link in self.links.values()
        ])
        return float(cpu), float(bandwidth)

    def assert_consistent(self) -> None:
        for node in self.nodes.values():
            if node.cpu_used < -1e-8 or node.cpu_used > node.cpu_capacity + 1e-8:
                raise AssertionError(f"Invalid CPU reservation on node {node.node_id}")
        for edge, link in self.links.items():
            used = link.service_used_mb_s + link.migration_used_mb_s
            if used < -1e-8 or used > link.capacity_mb_s + 1e-8:
                raise AssertionError(f"Invalid bandwidth reservation on edge {edge}")

