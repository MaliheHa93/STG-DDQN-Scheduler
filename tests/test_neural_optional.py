from __future__ import annotations

from pathlib import Path
import unittest

from stg_ddqn.config import load_config
from stg_ddqn.environment import STGEnvironment
from stg_ddqn.neural import TORCH_AVAILABLE, STGDDQNPolicy
from stg_ddqn.requests import RequestGenerator


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the minimal runtime")
class NeuralShapeTests(unittest.TestCase):
    def test_portable_attention_matches_standard_attention_values_and_gradients(self):
        import torch
        from stg_ddqn.neural import PortableTransformerEncoderLayer
        torch.manual_seed(6)
        layer = PortableTransformerEncoderLayer(16, 4, dropout=0, batch_first=True)
        first = torch.randn(3, 5, 16, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        expected = layer.self_attn(first, first, first, need_weights=False)[0]
        actual = layer.explicit_self_attention(second)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        expected.square().sum().backward()
        expected_gradients = {name: p.grad.clone() for name,p in layer.self_attn.named_parameters()}
        layer.zero_grad()
        actual.square().sum().backward()
        torch.testing.assert_close(second.grad, first.grad, atol=2e-6, rtol=1e-5)
        for name,p in layer.self_attn.named_parameters():
            torch.testing.assert_close(p.grad, expected_gradients[name], atol=1e-5, rtol=1e-5)

    def test_masked_network_returns_a_valid_action(self):
        cfg = load_config(ROOT / "configs" / "smoke.yaml")
        environment = STGEnvironment(cfg, 11)
        environment.vehicles = environment.mobility.state_at(0.0)
        environment.state_history.record(0.0, environment.vehicles)
        request = environment.request_generator._sample(
            0.0, sorted(environment.vehicles)[0], tracked=True
        )
        vehicle = environment.vehicles[request.vehicle_id]
        access = environment.topology.access_node(vehicle.x_m, vehicle.y_m)
        candidates = environment.candidate_generator.build(request, None, access, 0.0)
        observation = environment.state_history.build(
            request, None, candidates, 0.0, 0.0, False, 1
        )
        policy = STGDDQNPolicy(cfg, seed=11)
        action = policy.select(observation, training=False)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, len(candidates))


if __name__ == "__main__":
    unittest.main()

