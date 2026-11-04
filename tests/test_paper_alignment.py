from __future__ import annotations

from pathlib import Path
import unittest

from stg_ddqn.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class PaperTableTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs" / "paper.yaml")

    def test_table_i_network_and_workload_values(self):
        self.assertEqual(self.cfg["network"]["nodes"], 24)
        self.assertEqual(self.cfg["network"]["physical_links"], 43)
        self.assertEqual(self.cfg["network"]["coverage_radius_m"], 350)
        self.assertEqual(self.cfg["traffic"]["sfc_length"], [3, 6])
        self.assertEqual(self.cfg["traffic"]["latency_requirement_ms"], [80, 250])
        self.assertEqual(self.cfg["traffic"]["bandwidth_mb_s"], [5, 20])
        self.assertEqual(self.cfg["traffic"]["service_lifetime_s"], [30, 120])

    def test_table_i_candidate_and_learning_values(self):
        self.assertEqual(self.cfg["candidates"]["hop_cutoff"], 4)
        self.assertEqual(self.cfg["candidates"]["paths_per_endpoint"], 3)
        self.assertEqual(self.cfg["candidates"]["maximum_actions"], 32)
        self.assertEqual(self.cfg["state"]["history_window"], 5)
        self.assertEqual(self.cfg["learning"]["attention_heads"], 8)
        self.assertEqual(self.cfg["learning"]["embedding_dimension"], 128)
        self.assertEqual(self.cfg["learning"]["training_episodes"], 15)

    def test_reward_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.cfg["reward"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()

