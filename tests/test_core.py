from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from stg_ddqn.candidates import CandidateGenerator
from stg_ddqn.config import load_config
from stg_ddqn.environment import STGEnvironment
from stg_ddqn.policies import (
    DelayGreedyPolicy,
    DynamicLatencyAwarePartialMigrationPolicy,
    IterativeMigrationPolicy,
)
from stg_ddqn.reward import active_reward, initial_reward
from stg_ddqn.topology import Topology
from stg_ddqn.types import (
    ActiveSFC,
    PlacementPlan,
    RequestRecord,
    SFCRequest,
    VNFSpec,
)


ROOT = Path(__file__).resolve().parents[1]


def smoke_config():
    return load_config(ROOT / "configs" / "smoke.yaml")


def topology_and_generator(cfg=None):
    cfg = cfg or smoke_config()
    topology = Topology(
        cfg["network"], np.random.default_rng(47), np.random.default_rng(11)
    )
    return topology, CandidateGenerator(cfg, topology)


def request(length=3, lifetime=200.0, bandwidth=5.0, latency=250.0):
    return SFCRequest(
        request_id=1,
        vehicle_id=0,
        arrival_s=0.0,
        lifetime_s=lifetime,
        bandwidth_mb_s=bandwidth,
        latency_requirement_ms=latency,
        vnfs=tuple(VNFSpec(5.0, 300.0, 50.0, 5.0) for _ in range(length)),
    )


class CandidateTests(unittest.TestCase):
    def test_candidates_preserve_order_and_loop_free_route(self):
        topology, generator = topology_and_generator()
        plans = generator.build(request(), None, access_node=0, time_s=0.0)
        self.assertTrue(plans)
        for plan in plans:
            self.assertEqual(len(plan.route), len(set(plan.route)))
            positions = [plan.route.index(host) for host in plan.hosts]
            self.assertEqual(positions, sorted(positions))
            self.assertLessEqual(plan.delay_ms, 250.0)

    def test_selective_migration_counts_only_changed_vnfs(self):
        topology, generator = topology_and_generator()
        req = request()
        neighbor = topology.adjacency[0][0]
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0.0)
        plan = generator._screen(
            req, active, 0, (0, neighbor), (0, neighbor, neighbor), 0.0
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.changed_vnfs, (1, 2))
        self.assertAlmostEqual(plan.migration_volume_mb, 110.0, places=8)

    def test_migration_uses_mb_per_second_and_milliseconds(self):
        topology, generator = topology_and_generator()
        req = request()
        neighbor = topology.adjacency[0][0]
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0.0)
        plan = generator._screen(
            req, active, 0, (0, neighbor), (neighbor, neighbor, neighbor), 0.0
        )
        self.assertIsNotNone(plan)
        per_step = 1000.0 * 55.0 / 20.0 + topology.links[(0, neighbor)].delay_ms
        self.assertAlmostEqual(plan.preparation_time_ms, 3.0 * per_step, places=7)

    def test_bandwidth_is_a_hard_mask(self):
        topology, generator = topology_and_generator()
        req = request(bandwidth=20.0)
        neighbor = topology.adjacency[0][0]
        link = topology.links[(0, neighbor)]
        link.service_used_mb_s = link.capacity_mb_s - 1.0
        plan = generator._screen(
            req, None, 0, (0, neighbor), (0, neighbor, neighbor), 0.0
        )
        self.assertIsNone(plan)

    def test_complete_fallback_prevents_false_blocking(self):
        cfg = smoke_config()
        cfg = deepcopy(cfg)
        cfg["candidates"]["hop_cutoff"] = 0
        topology, generator = topology_and_generator(cfg)
        topology.nodes[0].cpu_used = topology.nodes[0].cpu_capacity
        plans = generator.build(request(), None, access_node=0, time_s=0.0)
        self.assertTrue(generator.last_fallback_used)
        self.assertTrue(plans)
        self.assertTrue(any(any(host != 0 for host in plan.hosts) for plan in plans))

    def test_pareto_ranking_keeps_nondominated_plans_before_dominated(self):
        _, generator = topology_and_generator()
        req = request()
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0.0)

        def plan(kind, delay, volume, hosts):
            return PlacementPlan(
                hosts=hosts,
                route=(0,),
                access_node=0,
                delay_ms=delay,
                migration_volume_mb=volume,
                preparation_time_ms=0.0,
                downtime_ms=0.0,
                changed_vnfs=() if kind == "retain" else (0,),
                kind=kind,
                minimum_cpu_residual_ratio=0.5,
                minimum_bandwidth_residual_ratio=1.0,
            )

        retain = plan("retain", 20.0, 0.0, (0, 0, 0))
        a = plan("remap", 10.0, 100.0, (1, 1, 1))
        b = plan("remap", 12.0, 120.0, (2, 2, 2))
        c = plan("remap", 15.0, 80.0, (3, 3, 3))
        d = plan("remap", 10.0, 110.0, (4, 4, 4))
        ranked = generator._rank_and_bound([b, retain, d, c, a], active, 5)
        self.assertEqual(ranked, [retain, a, c, d, b])

    def test_fast_pareto_order_matches_brute_force_definition(self):
        _, generator = topology_and_generator()
        req = request()
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0.0)
        rng = np.random.default_rng(29)
        plans = []
        for index in range(80):
            delay = float(rng.integers(1, 16))
            volume = float(rng.integers(1, 16))
            plans.append(PlacementPlan(
                hosts=(index + 1, index + 1, index + 1),
                route=(0,),
                access_node=0,
                delay_ms=delay,
                migration_volume_mb=volume,
                preparation_time_ms=0.0,
                downtime_ms=0.0,
                changed_vnfs=(0,),
                kind="remap",
                minimum_cpu_residual_ratio=0.5,
                minimum_bandwidth_residual_ratio=1.0,
            ))
        nondominated, dominated = [], []
        for candidate in plans:
            is_dominated = any(
                other.delay_ms <= candidate.delay_ms
                and other.migration_volume_mb <= candidate.migration_volume_mb
                and (
                    other.delay_ms < candidate.delay_ms
                    or other.migration_volume_mb < candidate.migration_volume_mb
                )
                for other in plans
                if other is not candidate
            )
            (dominated if is_dominated else nondominated).append(candidate)
        ordering = lambda item: (item.delay_ms, item.migration_volume_mb, item.hosts)
        expected = sorted(nondominated, key=ordering) + sorted(dominated, key=ordering)
        actual = generator._rank_and_bound(plans, active, len(plans))
        self.assertEqual(actual, expected)


class RewardTests(unittest.TestCase):
    def test_equations_23_and_24(self):
        cfg = smoke_config()
        topology, generator = topology_and_generator(cfg)
        req = request()
        plan = generator.build(req, None, 0, 0.0)[0]
        expected_initial = cfg["reward"]["latency"] * (
            1.0 - plan.delay_ms / req.latency_requirement_ms
        )
        self.assertAlmostEqual(initial_reward(cfg, req, plan), expected_initial)
        reward = active_reward(
            cfg, req, plan, pre_action_delay_ms=plan.delay_ms + 10.0,
            remaining_lifetime_s=req.lifetime_s / 2.0,
        )
        expected = (
            cfg["reward"]["latency"] * (1.0 - plan.delay_ms / req.latency_requirement_ms)
            + cfg["reward"]["lifetime_benefit"] * 0.5 * (10.0 / req.latency_requirement_ms)
        )
        self.assertAlmostEqual(reward, expected)


class BaselinePolicyTests(unittest.TestCase):
    @staticmethod
    def _plan(kind, delay, changed=(), volume=0.0):
        return PlacementPlan(
            hosts=(0, 0, 0) if not changed else (0, 1, 1),
            route=(0,) if not changed else (0, 1),
            access_node=0,
            delay_ms=float(delay),
            migration_volume_mb=float(volume),
            preparation_time_ms=0.0,
            downtime_ms=0.0,
            changed_vnfs=tuple(changed),
            kind=kind,
            minimum_cpu_residual_ratio=0.5,
            minimum_bandwidth_residual_ratio=0.5,
        )

    def test_im_trigger_cannot_select_retain_again(self):
        observation = SimpleNamespace(
            is_new=False,
            candidates=(
                self._plan("retain", 95.0),
                self._plan("remap", 70.0, changed=(1, 2), volume=110.0),
            ),
            metadata={"latency_requirement_ms": 100.0},
        )
        self.assertEqual(IterativeMigrationPolicy(0.9).select(observation), 1)

    def test_dlapm_trigger_cannot_select_retain_again(self):
        observation = SimpleNamespace(
            is_new=False,
            candidates=(
                self._plan("route_update", 94.0),
                self._plan("remap", 70.0, changed=(1, 2), volume=110.0),
            ),
            metadata={
                "pre_action_delay_ms": 100.0,
                "latency_requirement_ms": 100.0,
            },
        )
        self.assertEqual(
            DynamicLatencyAwarePartialMigrationPolicy(0.05).select(observation),
            1,
        )


class SimulationTests(unittest.TestCase):
    def test_rejected_attempt_is_counted_in_decision_runtime(self):
        cfg = smoke_config()
        cfg["network"]["cpu_capacity"] = [1.0, 1.0]
        cfg["simulation"]["warmup_s"] = 0.0
        cfg["simulation"]["measurement_s"] = 10.0
        cfg["traffic"]["arrival_rate_req_s"] = 2.0
        environment = STGEnvironment(cfg, 11)
        row, _ = environment.run(DelayGreedyPolicy(), training=False)
        self.assertGreater(row["decision_count"], 0)
        self.assertTrue(np.isfinite(row["mean_decision_runtime_ms"]))
        self.assertEqual(row["mean_policy_runtime_ms"], 0.0)

    def test_instantaneous_action_waits_for_next_real_decision_state(self):
        class RecordingPolicy(DelayGreedyPolicy):
            def __init__(self):
                self.transitions = []

            def observe(self, transition):
                self.transitions.append(transition)
                return {}

        cfg = smoke_config()
        environment = STGEnvironment(cfg, 11)
        environment.policy = RecordingPolicy()
        environment.vehicles = environment.mobility.state_at(0.0)
        environment.state_history.record(0.0, environment.vehicles)
        vehicle_id = next(iter(environment.vehicles))
        req = request()
        req.vehicle_id = vehicle_id
        access = environment.topology.access_node(
            environment.vehicles[vehicle_id].x_m,
            environment.vehicles[vehicle_id].y_m,
        )
        self.assertIsNotNone(access)
        plans = environment.candidate_generator.build(req, None, access, 0.0)
        observation = environment.state_history.build(
            req, None, plans, 0.0, 0.0, False,
            environment.candidate_generator.last_paths_examined,
        )
        environment.records[req.request_id] = RequestRecord(req.request_id, True)
        environment._apply_initial(req, plans[0], observation, 0)
        active = environment.active[req.request_id]
        self.assertEqual(environment.policy.transitions, [])
        self.assertIsNotNone(active.pending_decision)
        next_observation = environment._next_observation(active)
        self.assertIsNotNone(next_observation)
        environment._finish_pending_decision(active, next_observation, terminal=False)
        self.assertEqual(len(environment.policy.transitions), 1)
        self.assertIs(
            environment.policy.transitions[0].next_observation,
            next_observation,
        )
        environment._release_active_resources(active)
        environment.topology.assert_consistent()

    def test_make_before_break_keeps_source_and_destination_cpu_reserved(self):
        cfg = smoke_config()
        environment = STGEnvironment(cfg, 11)
        environment.vehicles = environment.mobility.state_at(0.0)
        req = request()
        neighbor = environment.topology.adjacency[0][0]
        source_cpu = sum(vnf.cpu for vnf in req.vnfs)
        environment.topology.reserve_cpu({0: source_cpu})
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0.0)
        environment.active[req.request_id] = active
        environment.records[req.request_id] = RequestRecord(
            req.request_id, True, admitted=True
        )
        plan = environment.candidate_generator._screen(
            req, active, 0, (0, neighbor), (0, neighbor, neighbor), 0.0
        )
        self.assertIsNotNone(plan)
        environment.state_history.record(0.0, environment.vehicles)
        observation = environment.state_history.build(
            req, active, [plan], 0.0, 0.0, False, 1
        )
        environment._initiate_migration(active, plan, observation, 0, 0.0)
        self.assertAlmostEqual(environment.topology.nodes[0].cpu_used, source_cpu)
        self.assertAlmostEqual(environment.topology.nodes[neighbor].cpu_used, 10.0)
        environment._release_pending_without_transition(active)
        environment.topology.release_cpu({0: source_cpu})
        environment.topology.assert_consistent()

    def test_paired_seed_reproduces_scientific_metrics(self):
        cfg = smoke_config()
        first, _ = STGEnvironment(cfg, 11).run(DelayGreedyPolicy())
        second, _ = STGEnvironment(cfg, 11).run(DelayGreedyPolicy())
        ignored = {
            "mean_decision_runtime_ms", "p95_decision_runtime_ms",
            "mean_candidate_runtime_ms", "mean_policy_runtime_ms",
            "mean_fallback_runtime_ms",
            "mean_reference_route_runtime_ms",
        }
        for key in first:
            if key not in ignored:
                if isinstance(first[key], float) and np.isnan(first[key]):
                    self.assertTrue(np.isnan(second[key]))
                else:
                    self.assertEqual(first[key], second[key], key)

    def test_active_requests_are_excluded_from_continuity_denominator(self):
        cfg = smoke_config()
        cfg = deepcopy(cfg)
        cfg["traffic"]["service_lifetime_s"] = [100, 100]
        result, _ = STGEnvironment(cfg, 11).run(DelayGreedyPolicy())
        self.assertEqual(result["continuity_resolved_requests"], 0)
        self.assertGreaterEqual(result["active_excluded_requests"], 1)


if __name__ == "__main__":
    unittest.main()
