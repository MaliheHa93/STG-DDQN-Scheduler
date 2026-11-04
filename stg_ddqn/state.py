from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .mobility import VehicleState, dwell_time_s
from .topology import Topology
from .types import ActiveSFC, DecisionObservation, PlacementPlan, SFCRequest


@dataclass
class Snapshot:
    time_s: float
    node_cpu_ratios: np.ndarray
    edge_features: np.ndarray
    vehicles: dict[int, VehicleState]


class StateHistory:
    def __init__(self, cfg: dict, topology: Topology):
        self.cfg = cfg
        self.topology = topology
        self.window = int(cfg["state"]["history_window"])
        self.snapshots: deque[Snapshot] = deque(maxlen=self.window)

    def record(self, time_s: float, vehicles: dict[int, VehicleState]) -> None:
        node_ratios, edge_features = self.topology.resource_snapshot()
        snapshot = Snapshot(
            time_s=float(time_s),
            node_cpu_ratios=node_ratios.copy(),
            edge_features=edge_features.copy(),
            vehicles=dict(vehicles),
        )
        if self.snapshots and abs(self.snapshots[-1].time_s - time_s) < 1e-9:
            self.snapshots[-1] = snapshot
        else:
            self.snapshots.append(snapshot)

    def build(
        self,
        request: SFCRequest,
        active: ActiveSFC | None,
        candidates: list[PlacementPlan],
        time_s: float,
        pre_action_delay_ms: float,
        fallback_used: bool,
        paths_examined: int,
    ) -> DecisionObservation:
        if not self.snapshots:
            raise RuntimeError("At least one state snapshot is required")
        snapshots = list(self.snapshots)
        while len(snapshots) < self.window:
            snapshots.insert(0, snapshots[0])
        snapshots = snapshots[-self.window:]
        node_features = np.stack([
            self._node_features(snapshot, request.vehicle_id) for snapshot in snapshots
        ])
        edge_features = np.stack([snapshot.edge_features for snapshot in snapshots])
        vehicle_features = np.stack([
            self._vehicle_features(snapshot, request.vehicle_id) for snapshot in snapshots
        ])
        remaining_lifetime = max(0.0, request.expiry_s - time_s)
        request_features = self._request_features(
            request, active, time_s, remaining_lifetime
        )
        vnf_features = self._vnf_features(request)
        max_sfc = int(self.cfg["state"]["maximum_sfc_length"])
        current_hosts = np.full(max_sfc, -1, dtype=np.int64)
        if active is not None:
            current_hosts[:len(active.hosts)] = active.hosts
        maximum_actions = int(self.cfg["candidates"]["maximum_actions"])
        candidate_hosts = np.full((maximum_actions, max_sfc), -1, dtype=np.int64)
        candidate_features = np.zeros((maximum_actions, 11), dtype=np.float32)
        candidate_mask = np.zeros(maximum_actions, dtype=bool)
        for index, plan in enumerate(candidates[:maximum_actions]):
            candidate_hosts[index, :len(plan.hosts)] = plan.hosts
            candidate_features[index] = self._candidate_features(
                request, plan, remaining_lifetime
            )
            candidate_mask[index] = True
        return DecisionObservation(
            request_id=request.request_id,
            is_new=active is None,
            candidates=tuple(candidates[:maximum_actions]),
            node_features=node_features,
            edge_features=edge_features,
            edge_index=self.topology.edge_index.copy(),
            vehicle_features=vehicle_features,
            request_features=request_features,
            vnf_features=vnf_features,
            current_hosts=current_hosts,
            candidate_hosts=candidate_hosts,
            candidate_features=candidate_features,
            candidate_mask=candidate_mask,
            metadata={
                "request": request,
                "latency_requirement_ms": request.latency_requirement_ms,
                "remaining_lifetime_s": remaining_lifetime,
                "pre_action_delay_ms": pre_action_delay_ms,
                "fallback_used": fallback_used,
                "paths_examined": paths_examined,
            },
        )

    def _node_features(self, snapshot: Snapshot, vehicle_id: int) -> np.ndarray:
        width, height = self.topology.area_m
        count = len(self.topology.nodes)
        features = np.zeros((count, 4), dtype=np.float32)
        vehicle = snapshot.vehicles.get(vehicle_id)
        access = (
            self.topology.access_node(vehicle.x_m, vehicle.y_m)
            if vehicle is not None else None
        )
        for node_id, node in self.topology.nodes.items():
            features[node_id] = (
                node.x_m / width,
                node.y_m / height,
                snapshot.node_cpu_ratios[node_id],
                1.0 if node_id == access else 0.0,
            )
        return features

    def _vehicle_features(self, snapshot: Snapshot, vehicle_id: int) -> np.ndarray:
        width, height = self.topology.area_m
        vehicle = snapshot.vehicles.get(vehicle_id)
        if vehicle is None:
            return np.zeros(6, dtype=np.float32)
        access = self.topology.access_node(vehicle.x_m, vehicle.y_m)
        dwell_clip = float(self.cfg["mobility"]["dwell_clip_s"])
        dwell = (
            dwell_time_s(vehicle, access, self.topology, dwell_clip,
                         self.cfg["mobility"].get("dwell_estimator", "ray_circle"))
            if access is not None else 0.0
        )
        maximum_speed = max(1e-9, float(self.cfg["mobility"]["maximum_speed_m_s"]))
        return np.asarray((
            vehicle.x_m / width,
            vehicle.y_m / height,
            min(1.0, vehicle.speed_m_s / maximum_speed),
            math.sin(vehicle.heading_rad),
            math.cos(vehicle.heading_rad),
            min(1.0, dwell / dwell_clip),
        ), dtype=np.float32)

    def _request_features(self, request: SFCRequest, active: ActiveSFC | None,
                          time_s: float, remaining_lifetime: float) -> np.ndarray:
        traffic = self.cfg["traffic"]
        max_bandwidth = float(traffic["bandwidth_mb_s"][1])
        max_latency = float(traffic["latency_requirement_ms"][1])
        max_sfc = float(self.cfg["state"]["maximum_sfc_length"])
        elapsed_ratio = min(1.0, max(0.0, time_s - request.arrival_s) / request.lifetime_s)
        return np.asarray((
            request.bandwidth_mb_s / max_bandwidth,
            request.latency_requirement_ms / max_latency,
            remaining_lifetime / request.lifetime_s,
            len(request.vnfs) / max_sfc,
            1.0 if active is None else 0.0,
            elapsed_ratio,
        ), dtype=np.float32)

    def _vnf_features(self, request: SFCRequest) -> np.ndarray:
        traffic = self.cfg["traffic"]
        max_sfc = int(self.cfg["state"]["maximum_sfc_length"])
        result = np.zeros((max_sfc, 4), dtype=np.float32)
        for index, vnf in enumerate(request.vnfs):
            result[index] = (
                vnf.cpu / float(traffic["cpu_requirement"][1]),
                vnf.workload_mi / float(traffic["workload_mi"][1]),
                vnf.image_mb / float(traffic["vnf_image_mb"][1]),
                vnf.state_mb / max(1e-9, float(traffic["runtime_state_mb"])),
            )
        return result

    def _candidate_features(self, request: SFCRequest, plan: PlacementPlan,
                            remaining_lifetime: float) -> np.ndarray:
        type_vector = {
            "initial": (1.0, 0.0, 0.0, 0.0),
            "retain": (0.0, 1.0, 0.0, 0.0),
            "route_update": (0.0, 0.0, 1.0, 0.0),
            "remap": (0.0, 0.0, 0.0, 1.0),
        }[plan.kind]
        return np.asarray((
            plan.delay_ms / request.latency_requirement_ms,
            plan.migration_volume_mb / float(self.cfg["migration"]["volume_normalizer_mb"]),
            len(plan.changed_vnfs) / len(request.vnfs),
            plan.preparation_time_ms / max(1.0, remaining_lifetime * 1000.0),
            plan.downtime_ms / float(self.cfg["migration"]["downtime_normalizer_ms"]),
            plan.minimum_cpu_residual_ratio,
            plan.minimum_bandwidth_residual_ratio,
            *type_vector,
        ), dtype=np.float32)

