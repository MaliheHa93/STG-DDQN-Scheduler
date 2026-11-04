from __future__ import annotations

import csv
import math
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .topology import Topology


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    x_m: float
    y_m: float
    speed_m_s: float
    heading_rad: float


class MobilityProvider:
    def state_at(self, time_s: float) -> dict[int, VehicleState]:
        raise NotImplementedError


class SyntheticMobility(MobilityProvider):
    """Deterministic smoke/testing mobility; final paper runs must use SUMO CSV."""

    def __init__(self, cfg: dict, topology: Topology, rng: np.random.Generator):
        self.cfg = cfg
        self.topology = topology
        self.rng = rng
        self.last_time_s = 0.0
        self.states: dict[int, VehicleState] = {}
        minimum_speed = float(cfg["minimum_speed_m_s"])
        maximum_speed = float(cfg["maximum_speed_m_s"])
        nodes = list(topology.nodes.values())
        for vehicle_id in range(int(cfg["maximum_vehicles"])):
            anchor = nodes[int(rng.integers(0, len(nodes)))]
            radius = anchor.coverage_radius_m * math.sqrt(float(rng.uniform(0.0, 0.65)))
            angle = float(rng.uniform(-math.pi, math.pi))
            x_m = anchor.x_m + radius * math.cos(angle)
            y_m = anchor.y_m + radius * math.sin(angle)
            speed = float(rng.uniform(minimum_speed, maximum_speed))
            heading = float(rng.uniform(-math.pi, math.pi))
            self.states[vehicle_id] = VehicleState(vehicle_id, x_m, y_m, speed, heading)

    def state_at(self, time_s: float) -> dict[int, VehicleState]:
        delta = float(time_s - self.last_time_s)
        if delta < -1e-9:
            raise ValueError("Synthetic mobility cannot move backward in time")
        if delta <= 0:
            return dict(self.states)
        width, height = self.topology.area_m
        updated: dict[int, VehicleState] = {}
        turn_probability = float(self.cfg.get("synthetic_turn_probability", 0.0))
        for vehicle_id, state in self.states.items():
            heading = state.heading_rad
            if self.rng.random() < turn_probability:
                heading += float(self.rng.normal(0.0, math.pi / 8.0))
            x_m = state.x_m + math.cos(heading) * state.speed_m_s * delta
            y_m = state.y_m + math.sin(heading) * state.speed_m_s * delta
            if x_m < 0 or x_m > width:
                heading = math.pi - heading
                x_m = min(width, max(0.0, x_m))
            if y_m < 0 or y_m > height:
                heading = -heading
                y_m = min(height, max(0.0, y_m))
            if self.topology.access_node(x_m, y_m) is None:
                # Preserve the analytical assumption of continuous fog coverage.
                heading += math.pi
                x_m, y_m = state.x_m, state.y_m
            heading = (heading + math.pi) % (2.0 * math.pi) - math.pi
            updated[vehicle_id] = VehicleState(
                vehicle_id, x_m, y_m, state.speed_m_s, heading
            )
        self.states = updated
        self.last_time_s = float(time_s)
        return dict(self.states)


class SUMOCSVMobility(MobilityProvider):
    """Reads epoch-sampled SUMO vehicle states from a reproducible CSV trace."""

    REQUIRED_COLUMNS = {"time_s", "vehicle_id", "x_m", "y_m", "speed_m_s", "heading_rad"}

    def __getstate__(self):
        # Resume states reference the immutable, checksummed trace rather than
        # serializing millions of vehicle objects at every training checkpoint.
        return {'path': self.path, 'epoch_s': self.epoch_s,
                'required_duration_s': self.required_duration_s}

    def __setstate__(self, state):
        self.__init__(state['path'], state['epoch_s'], state['required_duration_s'])

    def __init__(self, path: str | Path, epoch_s: float, required_duration_s: float | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"SUMO trace not found: {self.path}. Use tools/export_sumo_trace.py first."
            )
        self.epoch_s = float(epoch_s)
        self.required_duration_s = required_duration_s
        if self.epoch_s <= 0:
            raise ValueError('SUMO epoch must be positive')
        metadata_path = self.path.with_suffix('.json')
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if required_duration_s is not None:
            if not self.metadata:
                raise ValueError('SUMO trace needs a metadata sidecar to verify empty frames and duration')
            if abs(self.metadata['epoch_s'] - self.epoch_s) > 1e-9:
                raise ValueError('SUMO trace epoch differs from simulation epoch')
            if self.metadata['start_time_s'] > 0 or self.metadata['end_time_exclusive_s'] < required_duration_s:
                raise ValueError('SUMO trace is too short for the warmup and measurement window')
            import hashlib
            if self.metadata.get('csv_sha256') != hashlib.sha256(self.path.read_bytes()).hexdigest():
                raise ValueError('SUMO trace checksum differs from its metadata')
        self.frames: dict[int, dict[int, VehicleState]] = {}
        id_names = {}
        with self.path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"SUMO CSV is missing columns: {sorted(missing)}")
            for row in reader:
                values = [float(row[key]) for key in ('time_s', 'x_m', 'y_m', 'speed_m_s', 'heading_rad')]
                if not all(math.isfinite(value) for value in values) or values[0] < 0 or values[3] < 0:
                    raise ValueError('SUMO trace contains invalid numeric values')
                frame = int(round(values[0] / self.epoch_s))
                if abs(frame * self.epoch_s - values[0]) > 1e-6:
                    raise ValueError('SUMO trace contains a time not aligned to the epoch')
                vehicle_id = _stable_vehicle_id(row["vehicle_id"])
                if vehicle_id in id_names and id_names[vehicle_id] != row['vehicle_id']:
                    raise ValueError('SUMO vehicle ID hash collision')
                id_names[vehicle_id] = row['vehicle_id']
                if vehicle_id in self.frames.get(frame, {}):
                    raise ValueError('Duplicate SUMO vehicle in one epoch')
                self.frames.setdefault(frame, {})[vehicle_id] = VehicleState(
                    vehicle_id=vehicle_id,
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    speed_m_s=float(row["speed_m_s"]),
                    heading_rad=float(row["heading_rad"]),
                )
        if not self.frames:
            raise ValueError(f"SUMO trace is empty: {self.path}")

    def state_at(self, time_s: float) -> dict[int, VehicleState]:
        if self.metadata and not self.metadata['start_time_s'] <= time_s < self.metadata['end_time_exclusive_s']:
            raise ValueError('Requested time is outside the SUMO trace horizon')
        frame = int(round(float(time_s) / self.epoch_s))
        return dict(self.frames.get(frame, {}))


def _stable_vehicle_id(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        # FNV-1a gives a stable identifier without Python's randomized hash salt.
        result = 2166136261
        for byte in value.encode("utf-8"):
            result ^= byte
            result = (result * 16777619) & 0xFFFFFFFF
        return result


def make_mobility(cfg: dict, topology: Topology, rng: np.random.Generator,
                  epoch_s: float) -> MobilityProvider:
    mode = str(cfg["mode"]).lower()
    if mode == "synthetic":
        return SyntheticMobility(cfg, topology, rng)
    if mode == "sumo_csv":
        if not cfg.get("sumo_csv"):
            raise ValueError("mobility.sumo_csv is required for sumo_csv mode")
        return SUMOCSVMobility(cfg["sumo_csv"], epoch_s, cfg.get('required_duration_s'))
    raise ValueError(f"Unknown mobility mode: {mode}")


def resolve_sumo_run(cfg: dict, seed: int, training: bool = False) -> dict:
    """Resolve a paired seed/speed trace before environment and provenance creation."""
    if cfg['mobility']['mode'] != 'sumo_csv' or not cfg['mobility'].get('sumo_trace_pattern'):
        return cfg
    result = deepcopy(cfg)
    mobility = result['mobility']
    allowed = mobility['training_trace_seeds' if training else 'evaluation_trace_seeds']
    # train.py offsets repeated episode seeds by 100000; reuse the designated
    # training mobility realization, never a held-out evaluation realization.
    trace_seed = seed % 100000 if training else seed
    if trace_seed not in allowed:
        raise ValueError(f'No authorized SUMO trace seed {trace_seed}; generate/configure it first')
    speed = float(mobility['maximum_speed_m_s'])
    mobility['sumo_csv'] = mobility['sumo_trace_pattern'].format(speed=f'{speed:g}', seed=trace_seed)
    mobility['trace_seed'] = trace_seed
    mobility['required_duration_s'] = (float(result['simulation']['warmup_s']) +
                                       float(result['simulation']['measurement_s']))
    return result


def dwell_time_s(vehicle: VehicleState, access_node: int, topology: Topology,
                 clip_s: float, method: str = "ray_circle") -> float:
    node = topology.nodes[access_node]
    dx, dy = vehicle.x_m - node.x_m, vehicle.y_m - node.y_m
    distance = math.hypot(dx, dy)
    if method == "ray_circle":
        # Constant-velocity coverage exit, not a prediction of association
        # change in overlapping coverage or of future road turns.
        speed = vehicle.speed_m_s
        if speed <= 1e-12:
            return float(clip_s)
        dot = speed * (dx * math.cos(vehicle.heading_rad) + dy * math.sin(vehicle.heading_rad))
        if distance > node.coverage_radius_m + 1e-9:
            return 0.0
        discriminant = dot * dot + speed * speed * (node.coverage_radius_m ** 2 - distance ** 2)
        exit_s = (-dot + math.sqrt(max(0.0, discriminant))) / (speed * speed)
        return float(min(clip_s, max(0.0, exit_s)))
    if method != "radial":
        raise ValueError(f"Unknown dwell estimator: {method}")
    exit_distance = max(0.0, node.coverage_radius_m - distance)
    if distance <= 1e-12:
        return float(clip_s)
    velocity_x = vehicle.speed_m_s * math.cos(vehicle.heading_rad)
    velocity_y = vehicle.speed_m_s * math.sin(vehicle.heading_rad)
    radial_velocity = (velocity_x * dx + velocity_y * dy) / distance
    if radial_velocity <= 0:
        return float(clip_s)
    return float(min(clip_s, exit_distance / radial_velocity))

