from pathlib import Path
import json
import math
import tempfile
import unittest

from stg_ddqn.config import load_config
from stg_ddqn.mobility import SUMOCSVMobility, resolve_sumo_run
from stg_ddqn.sumo_scenario import fcd_to_csv
from stg_ddqn.topology import Topology
from stg_ddqn.candidates import CandidateGenerator
import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SUMOTests(unittest.TestCase):
    def trace(self, root):
        fcd = root / 'fcd.xml'
        fcd.write_text('<fcd-export><timestep time="60">'
                       '<vehicle id="car.0" x="2" y="3" angle="90" speed="10"/>'
                       '</timestep><timestep time="61"/>'
                       '<timestep time="62"><vehicle id="car.1" x="4" y="5" '
                       'angle="0" speed="5"/></timestep></fcd-export>')
        csv = root / 'trace.csv'
        metadata = fcd_to_csv(fcd, csv, 60, 3, 1, 10)
        csv.with_suffix('.json').write_text(json.dumps(metadata))
        return csv

    def test_heading_empty_frame_and_vehicle_departure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = self.trace(Path(directory))
            mobility = SUMOCSVMobility(trace, 1, 3)
            first = next(iter(mobility.state_at(0).values()))
            self.assertAlmostEqual(first.heading_rad, 0)
            self.assertEqual(mobility.state_at(1), {})
            last = next(iter(mobility.state_at(2).values()))
            self.assertAlmostEqual(last.heading_rad, math.pi / 2)
            self.assertNotEqual(first.vehicle_id, last.vehicle_id)
            with self.assertRaisesRegex(ValueError, 'outside'):
                mobility.state_at(3)

    def test_incomplete_trace_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = self.trace(Path(directory))
            with self.assertRaisesRegex(ValueError, 'too short'):
                SUMOCSVMobility(trace, 1, 4)
            trace.write_text(trace.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                SUMOCSVMobility(trace, 1, 3)

    def test_missing_fcd_epoch_is_not_vehicle_departure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.trace(root)
            with self.assertRaisesRegex(ValueError, 'every requested epoch'):
                fcd_to_csv(root / 'fcd.xml', trace, 60, 4, 1, 10)

    def test_speed_limit_violation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.trace(root)
            with self.assertRaisesRegex(ValueError, 'speed cap'):
                fcd_to_csv(root / 'fcd.xml', trace, 60, 3, 1, 9)

    def test_held_out_seed_selection_and_speed_specific_files(self):
        cfg = load_config(ROOT / 'configs/sumo_paper.yaml')
        first = resolve_sumo_run(cfg, 101)
        self.assertIn('speed_30/seed_101', first['mobility']['sumo_csv'])
        training = resolve_sumo_run(cfg, 110001, True)
        self.assertIn('seed_10001', training['mobility']['sumo_csv'])
        with self.assertRaisesRegex(ValueError, 'authorized SUMO trace'):
            resolve_sumo_run(cfg, 101, True)
        with self.assertRaisesRegex(ValueError, 'authorized SUMO trace'):
            resolve_sumo_run(cfg, 10001, False)
        self.assertIsNone(cfg['mobility']['sumo_csv'])

    def test_paper_profile_keeps_table_and_baseline_parameters(self):
        cfg = load_config(ROOT / 'configs/sumo_paper.yaml')
        self.assertEqual(cfg['baselines']['im_latency_trigger_ratio'], .9)
        self.assertEqual(cfg['baselines']['dlapm_improvement_threshold'], .05)
        self.assertEqual(cfg['network']['nodes'], 24)
        self.assertEqual(cfg['learning']['embedding_dimension'], 128)
        self.assertEqual(cfg['simulation']['measurement_s'], 3600)
        self.assertEqual(cfg['simulation']['warmup_s'], 600)
        self.assertFalse(set(cfg['mobility']['training_trace_seeds']) &
                         set(cfg['mobility']['evaluation_trace_seeds']))

    def test_downstream_access_proves_retain_loop_infeasible_without_search(self):
        cfg = load_config(ROOT / 'configs/sumo_paper.yaml')
        topology = Topology(cfg['network'], np.random.default_rng(1), np.random.default_rng(2))
        with patch('stg_ddqn.topology.heapq.heappop', side_effect=AssertionError('unnecessary search')):
            self.assertIsNone(topology.shortest_ordered_route(2, (1, 1, 2)))
            self.assertIsNone(topology.shortest_ordered_route(0, (1, 2, 1)))
        self.assertEqual(topology.shortest_ordered_route(1, (1, 1)), (1,))

    def test_ordered_astar_matches_exhaustive_simple_routes(self):
        cfg = load_config(ROOT / 'configs/default.yaml')
        cfg['network'].update(nodes=6, physical_links=8)
        rng = np.random.default_rng(42)
        for realization in range(3):
            topology = Topology(cfg['network'], np.random.default_rng(realization), rng)
            # Include saturated links and retained reservations in the comparison.
            for link in topology.links.values():
                if rng.random() < .15:
                    link.service_used_mb_s = link.capacity_mb_s
            for _ in range(40):
                source = int(rng.integers(6))
                hosts = tuple(int(x) for x in rng.integers(0, 6, 3))
                routes = []
                for route in topology.all_simple_paths_by_delay(source, required_bandwidth=5):
                    if route[-1] != hosts[-1] or not set(hosts).issubset(route):
                        continue
                    positions = [route.index(host) for host in hosts]
                    if positions == sorted(positions):
                        routes.append(route)
                actual = topology.shortest_ordered_route(source, hosts, required_bandwidth=5)
                if not routes:
                    self.assertIsNone(actual)
                else:
                    self.assertIsNotNone(actual, (source, hosts, routes))
                    self.assertAlmostEqual(topology.path_delay_ms(actual),
                                           min(topology.path_delay_ms(route) for route in routes))

    def test_assignment_budget_preserves_only_valid_candidates_and_reports_exhaustion(self):
        from test_core import request
        cfg = load_config(ROOT / 'configs/default.yaml')
        topology = Topology(cfg['network'], np.random.default_rng(1), np.random.default_rng(2))
        cfg['candidates']['maximum_assignment_nodes'] = 1
        generator = CandidateGenerator(cfg, topology)
        self.assertEqual(generator.build(request(), None, 0, 0), [])
        self.assertEqual(generator.last_assignment_nodes, 1)
        self.assertTrue(generator.last_search_budget_exhausted)


if __name__ == '__main__':
    unittest.main()
