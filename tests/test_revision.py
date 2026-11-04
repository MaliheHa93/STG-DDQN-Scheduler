from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import math
import unittest
from dataclasses import replace

import numpy as np

from test_core import smoke_config, request, topology_and_generator
from stg_ddqn.environment import STGEnvironment
from stg_ddqn.experiments import _scenarios
from stg_ddqn.mobility import VehicleState, dwell_time_s
from stg_ddqn.policies import DelayGreedyPolicy
from stg_ddqn.requests import RequestGenerator
from stg_ddqn.types import ActiveSFC, RequestRecord


class RevisionTests(unittest.TestCase):
    def test_fallback_budget_is_distinct_from_proven_infeasibility(self):
        topology, generator = topology_and_generator()
        generator.candidate_cfg['maximum_fallback_paths'] = 1
        req = request()
        active = ActiveSFC(req, (0,) * 3, (0,), 0, 0)
        neighbor = topology.adjacency[0][0]
        with patch.object(topology, 'shortest_ordered_route', return_value=None), \
             patch.object(topology, 'nodes_within_hops', return_value=[]), \
             patch.object(topology, 'all_simple_paths_by_delay', return_value=iter([(0,), (0, neighbor)])), \
             patch.object(generator, '_screen_assignments'):
            self.assertEqual(generator.build(req, active, 0, 0), [])
        self.assertTrue(generator.last_search_budget_exhausted)
        self.assertEqual(generator.last_paths_examined, 1)

    def test_migration_path_avoids_saturated_shortcut_and_handles_disconnection(self):
        from stg_ddqn.topology import DirectedLink, Topology
        topology = SimpleNamespace(
            adjacency={0: [1, 2], 1: [], 2: [1]},
            links={(0, 1): DirectedLink(0, 1, 10, 1, service_used_mb_s=10),
                   (0, 2): DirectedLink(0, 2, 10, 2),
                   (2, 1): DirectedLink(2, 1, 10, 2)},
        )
        self.assertEqual(Topology.shortest_migration_path(topology, 0, 1), (0, 2, 1))
        self.assertIsNone(Topology.shortest_migration_path(topology, 0, 1, {(2, 1): 10}))

    def test_infeasibility_proof_avoids_exhaustive_expiry_search(self):
        topology, generator = topology_and_generator()
        req = request(lifetime=0.1)
        active = ActiveSFC(req, (0,) * 3, (0,), 0, 0)
        with patch.object(topology, 'shortest_ordered_route', return_value=None), \
             patch.object(topology, 'nodes_within_hops', side_effect=AssertionError('unnecessary search')):
            self.assertEqual(generator.build(req, active, 0, 0), [])

    def test_pruning_preserves_simulated_outcomes(self):
        from stg_ddqn.candidates import CandidateGenerator
        cfg = smoke_config()
        cfg['traffic']['arrival_rate_req_s'] = 1.0
        first, records_first = STGEnvironment(cfg, 11).run(DelayGreedyPolicy())
        def exhaustive(self, req, active, route, time_s):
            return self._placements_on_route(route, len(req.vnfs))
        with patch.object(CandidateGenerator, '_pruned_placements', exhaustive):
            second, records_second = STGEnvironment(cfg, 11).run(DelayGreedyPolicy())
        self.assertEqual(records_first, records_second)
        for key in first:
            if 'runtime' in key or key in {'candidate_paths_examined', 'assignment_nodes_examined'}:
                continue
            if isinstance(first[key], float) and np.isnan(first[key]):
                self.assertTrue(np.isnan(second[key]))
            else:
                self.assertEqual(first[key], second[key], key)

    def test_pruned_enumeration_matches_fully_screened_brute_force(self):
        topology, generator = topology_and_generator()
        neighbor = topology.adjacency[0][0]
        route = (0, neighbor)
        for lifetime in (0.5, 5, 200):
            req = request(length=4, lifetime=lifetime)
            for active in (None, ActiveSFC(req, (0,) * 4, (0,), 0, 0)):
                expected = [hosts for hosts in generator._placements_on_route(route, 4)
                            if generator._screen(req, active, 0, route, hosts, 0) is not None]
                actual = [hosts for hosts in generator._pruned_placements(req, active, route, 0)
                          if generator._screen(req, active, 0, route, hosts, 0) is not None]
                self.assertEqual(actual, expected)

    def test_ray_circle_center_tangent_and_inward_motion(self):
        node = SimpleNamespace(x_m=0, y_m=0, coverage_radius_m=100)
        topology = SimpleNamespace(nodes={0: node})
        for x, y, heading, expected in [
            (0, 0, 0, 10), (60, 0, math.pi / 2, 8),
            (60, 0, math.pi, 16), (60, 0, 0, 4),
        ]:
            self.assertAlmostEqual(dwell_time_s(VehicleState(0, x, y, 10, heading), 0, topology, 60), expected)

    def test_fallback_after_all_preliminary_remaps_fail(self):
        topology, generator = topology_and_generator()
        req = request()
        neighbor = topology.adjacency[0][0]
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0)
        good = generator._screen(req, active, 0, (0, neighbor), (neighbor,) * 3, 0)
        self.assertIsNotNone(good)
        cheap = good
        calls = []
        def screen_assignments(req, act, access, route, now, feasible, seen):
            feasible.append(cheap)
        def materialize(feasible, req, act, access, now):
            calls.append(len(feasible))
            return [good] if generator.last_fallback_used else []
        with patch.object(topology, 'shortest_ordered_route', return_value=None), \
             patch.object(generator, '_screen_assignments', side_effect=screen_assignments), \
             patch.object(generator, '_materialize', side_effect=materialize), \
             patch.object(topology, 'all_simple_paths_by_delay', return_value=iter([(0, neighbor)])):
            plans = generator.build(req, active, 0, 0)
        self.assertTrue(generator.last_fallback_used)
        self.assertEqual(plans, [good])

    def test_sumo_sweep_rejects_missing_or_identical_traces(self):
        cfg = smoke_config()
        cfg['mobility']['mode'] = 'sumo_csv'
        with self.assertRaisesRegex(ValueError, 'sumo_csv_by_speed'):
            list(_scenarios(cfg, {'mobility'}))
        with TemporaryDirectory() as directory:
            first, second = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            first.write_text('identical'); second.write_text('identical')
            cfg['mobility']['sumo_csv_by_speed'] = {10: str(first), 25: str(second)}
            with self.assertRaisesRegex(ValueError, 'distinct trace contents'):
                list(_scenarios(cfg, {'mobility'}))
            second.write_text('different')
            scenarios = list(_scenarios(cfg, {'mobility'}))
            self.assertEqual(scenarios[1][3]['mobility']['sumo_csv'], str(second))

    def test_fallback_does_not_stop_at_preliminary_action_limit(self):
        topology, generator = topology_and_generator()
        generator.candidate_cfg['maximum_actions'] = 1
        req = request()
        neighbor = topology.adjacency[0][0]
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0)
        good = generator._screen(req, active, 0, (0, neighbor), (neighbor,) * 3, 0)
        bad = replace(good, hosts=(0, 0, 0), route=(0,))
        def assignments(req, act, access, route, now, feasible, seen):
            feasible.append(bad if route == (0,) else good)
        def screen(req, act, access, route, hosts, now, **kwargs):
            return good if hosts == good.hosts else None
        with patch.object(topology, 'shortest_ordered_route', return_value=None), \
             patch.object(topology, 'nodes_within_hops', return_value=[0]), \
             patch.object(topology, 'k_shortest_simple_paths', return_value=[(0,)]), \
             patch.object(topology, 'all_simple_paths_by_delay', return_value=iter([(0,), (0, neighbor)])), \
             patch.object(generator, '_screen_assignments', side_effect=assignments), \
             patch.object(generator, '_screen', side_effect=screen):
            plans = generator.build(req, active, 0, 0)
        self.assertEqual(plans, [good])
        self.assertEqual(generator.last_paths_examined, 3)

    def test_present_vehicle_outside_coverage_is_failure(self):
        env = STGEnvironment(smoke_config(), 11)
        env.policy = DelayGreedyPolicy()
        req = request()
        env.topology.reserve_cpu({0: 15})
        env.active[1] = ActiveSFC(req, (0, 0, 0), (0,), 0, 0)
        env.records[1] = RequestRecord(1, True, admitted=True)
        env.vehicles = {0: VehicleState(0, 1e9, 1e9, 10, 0)}
        env._update_access_and_routes()
        self.assertTrue(env.records[1].continuity_failed)
        self.assertEqual(env.records[1].departure_reason, 'coverage_gap')
        env.topology.assert_consistent()

    def test_canceled_preparation_retains_transferred_bytes(self):
        cfg = smoke_config()
        env = STGEnvironment(cfg, 11)
        env.policy = DelayGreedyPolicy()
        env.vehicles = env.mobility.state_at(0)
        req = request()
        neighbor = env.topology.adjacency[0][0]
        env.topology.reserve_cpu({0: 15})
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0)
        env.active[1] = active
        env.records[1] = RequestRecord(1, True, admitted=True)
        plan = env.candidate_generator._screen(req, active, 0, (0, neighbor), (neighbor,) * 3, 0)
        env.state_history.record(0, env.vehicles)
        obs = env.state_history.build(req, active, [plan], 0, 0, False, 1)
        env._initiate_migration(active, plan, obs, 0, 0)
        env.time_s = 0.25
        env._advance_pending(0.25, set())
        self.assertAlmostEqual(active.transferred_mb, 5)
        env._cancel_pending(active, 'test', terminal=True)
        env._sync_record(active)
        self.assertAlmostEqual(env.records[1].migration_volume_mb, 5)
        self.assertAlmostEqual(env.records[1].canceled_transfer_mb, 5)
        self.assertEqual(env.records[1].completed_migration_volume_mb, 0)
        env._release_active_resources(active)
        env.topology.assert_consistent()

    def test_mmpp_mean_and_burstiness(self):
        cfg = smoke_config()['traffic']
        cfg.update(arrival_model='mmpp', arrival_rate_req_s=2)
        rng = np.random.default_rng(192)
        generator = RequestGenerator(cfg, 1, rng, np.random.default_rng(1))
        values = np.array([generator.arrival_count() for _ in range(30000)])
        self.assertAlmostEqual(values.mean(), 2, delta=0.15)
        self.assertGreater(values.var(), values.mean() * 1.5)

    def test_temporary_reservation_audit_preserves_live_allocations(self):
        cfg = smoke_config()
        env = STGEnvironment(cfg, 11)
        env.policy = DelayGreedyPolicy()
        env.vehicles = env.mobility.state_at(0)
        req = request()
        neighbor = env.topology.adjacency[0][0]
        env.topology.reserve_cpu({0: 15})
        active = ActiveSFC(req, (0, 0, 0), (0,), 0, 0)
        env.active[1] = active
        env.records[1] = RequestRecord(1, True, admitted=True)
        plan = env.candidate_generator._screen(req, active, 0, (0, neighbor), (0, neighbor, neighbor), 0)
        env.state_history.record(0, env.vehicles)
        obs = env.state_history.build(req, active, [plan], 0, 0, False, 1)
        env._initiate_migration(active, plan, obs, 0, 0)
        env.time_s = 6
        env._capture_reservation_peaks()
        env._integrate_reservations(1)
        self.assertAlmostEqual(env.temporary_cpu_seconds, 10)
        self.assertAlmostEqual(env.peak_temporary_cpu, 10)
        with patch('stg_ddqn.environment.CandidateGenerator') as constructor:
            constructor.return_value.build.return_value = [plan]
            env._audit_temporary_blocking(req, 0)
            copied = constructor.call_args.args[1]
            self.assertAlmostEqual(copied.nodes[neighbor].cpu_used, 0)
        self.assertEqual(env.blocked_without_temporary_reservations, 1)
        self.assertAlmostEqual(env.topology.nodes[neighbor].cpu_used, 10)
        self.assertAlmostEqual(env.topology.nodes[0].cpu_used, 15)
        env._release_active_resources(active)
        env.topology.assert_consistent()


if __name__ == '__main__':
    unittest.main()
