from __future__ import annotations

from collections import deque
from pathlib import Path
import math
from typing import Sequence

import numpy as np

from .policies import Policy
from .types import DecisionObservation, Transition

try:
    import torch
    from torch import nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal simulator installs
    torch = None
    nn = None
    functional = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PortableTransformerEncoderLayer(nn.TransformerEncoderLayer):
        """Identical attention parameters/equations with MPS dropout support.

        The fused MPS SDPA operator rejects nonzero dropout in training. Use
        explicit scaled dot-product attention on that backend, preserving the
        configured dropout and checkpoint parameter names. CPU/CUDA keep the
        standard PyTorch implementation.
        """

        def explicit_self_attention(self, x):
            batch, length, dimension = x.shape
            heads = self.self_attn.num_heads
            width = dimension // heads
            projected = functional.linear(x, self.self_attn.in_proj_weight,
                                          self.self_attn.in_proj_bias)
            q, k, v = [part.reshape(batch, length, heads, width).transpose(1, 2)
                       for part in projected.chunk(3, dim=-1)]
            scores = (q * (width ** -0.5)) @ k.transpose(-2, -1)
            weights = functional.softmax(scores, dim=-1)
            weights = functional.dropout(weights, p=self.self_attn.dropout,
                                         training=self.training)
            values = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, dimension)
            return functional.linear(values, self.self_attn.out_proj.weight,
                                     self.self_attn.out_proj.bias)

        def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
            if x.device.type == 'mps' and attn_mask is None and key_padding_mask is None and not is_causal:
                return self.dropout1(self.explicit_self_attention(x))
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal=is_causal)

    class GraphAttentionLayer(nn.Module):
        def __init__(self, dimension: int, heads: int):
            super().__init__()
            if dimension % heads:
                raise ValueError("Embedding dimension must be divisible by attention heads")
            self.dimension = dimension
            self.heads = heads
            self.head_dimension = dimension // heads
            self.node_projection = nn.Linear(dimension, dimension, bias=False)
            self.edge_projection = nn.Linear(2, heads, bias=False)
            self.attention_source = nn.Parameter(torch.empty(heads, self.head_dimension))
            self.attention_destination = nn.Parameter(torch.empty(heads, self.head_dimension))
            self.output_projection = nn.Linear(dimension, dimension)
            self.normalization = nn.LayerNorm(dimension)
            nn.init.xavier_uniform_(self.attention_source)
            nn.init.xavier_uniform_(self.attention_destination)

        def forward(self, nodes, edge_index, edge_features):
            batch, node_count, _ = nodes.shape
            source, destination = edge_index[0], edge_index[1]
            edge_count = source.numel()
            projected = self.node_projection(nodes).view(
                batch, node_count, self.heads, self.head_dimension
            )
            source_values = projected[:, source]
            destination_values = projected[:, destination]
            scores = (
                (source_values * self.attention_source).sum(-1)
                + (destination_values * self.attention_destination).sum(-1)
                + self.edge_projection(edge_features)
            )
            scores = functional.leaky_relu(scores, negative_slope=0.2)
            destination_index = destination.view(1, edge_count, 1).expand(
                batch, edge_count, self.heads
            )
            maxima = torch.full(
                (batch, node_count, self.heads),
                -torch.inf,
                device=nodes.device,
                dtype=nodes.dtype,
            )
            maxima.scatter_reduce_(
                1, destination_index, scores, reduce="amax", include_self=True
            )
            stabilized = scores - maxima.gather(1, destination_index)
            exponentials = torch.exp(stabilized)
            denominators = torch.zeros_like(maxima)
            denominators.index_add_(1, destination, exponentials)
            attention = exponentials / (
                denominators.gather(1, destination_index) + 1e-12
            )
            messages = source_values * attention.unsqueeze(-1)
            aggregated = torch.zeros(
                (batch, node_count, self.heads, self.head_dimension),
                device=nodes.device,
                dtype=nodes.dtype,
            )
            aggregated.index_add_(1, destination, messages)
            output = self.output_projection(aggregated.reshape(batch, node_count, -1))
            return self.normalization(nodes + functional.relu(output))


    class GraphEncoder(nn.Module):
        def __init__(self, dimension: int, heads: int, layers: int):
            super().__init__()
            self.input_projection = nn.Linear(4, dimension)
            self.layers = nn.ModuleList(
                GraphAttentionLayer(dimension, heads) for _ in range(layers)
            )
            self.pool_projection = nn.Linear(2 * dimension, dimension)

        def forward(self, node_features, edge_index, edge_features):
            nodes = functional.relu(self.input_projection(node_features))
            for layer in self.layers:
                nodes = layer(nodes, edge_index, edge_features)
            global_pool = nodes.mean(dim=1)
            access_weights = node_features[..., 3:4]
            access_pool = (nodes * access_weights).sum(dim=1) / (
                access_weights.sum(dim=1) + 1e-6
            )
            graph = functional.relu(
                self.pool_projection(torch.cat((global_pool, access_pool), dim=-1))
            )
            return graph, nodes


    class STGQNetwork(nn.Module):
        def __init__(self, cfg: dict, spatial_only: bool = False):
            super().__init__()
            learning = cfg["learning"]
            dimension = int(learning["embedding_dimension"])
            heads = int(learning["attention_heads"])
            self.dimension = dimension
            self.spatial_only = spatial_only
            self.graph_encoder = GraphEncoder(
                dimension,
                heads,
                int(learning["graph_layers"]),
            )
            self.vehicle_projection = nn.Linear(6, dimension)
            temporal_layer = PortableTransformerEncoderLayer(
                d_model=dimension,
                nhead=heads,
                dim_feedforward=int(learning["feedforward_dimension"]),
                activation="relu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                temporal_layer,
                num_layers=int(learning["temporal_layers"]),
            )
            self.request_projection = nn.Linear(6, dimension)
            self.vnf_projection = nn.Linear(4, dimension // 2)
            self.request_gru = nn.GRU(
                dimension + dimension // 2,
                dimension,
                batch_first=True,
            )
            self.candidate_gru = nn.GRU(
                dimension + dimension // 2,
                dimension,
                batch_first=True,
            )
            self.candidate_scalar_projection = nn.Linear(11, dimension)
            self.null_host = nn.Parameter(torch.zeros(1, 1, dimension))
            layers = []
            input_dimension = 5 * dimension
            for _ in range(int(learning["q_hidden_layers"])):
                layers.extend((nn.Linear(input_dimension, int(learning["q_hidden_dimension"])), nn.ReLU()))
                input_dimension = int(learning["q_hidden_dimension"])
            layers.append(nn.Linear(input_dimension, 1))
            self.q_network = nn.Sequential(*layers)

        def forward(self, batch):
            node_features = batch["node_features"]
            edge_features = batch["edge_features"]
            edge_index = batch["edge_index"]
            vehicle_features = batch["vehicle_features"]
            batch_size, window, node_count, _ = node_features.shape
            edge_count = edge_features.shape[2]
            flat_nodes = node_features.reshape(batch_size * window, node_count, 4)
            flat_edges = edge_features.reshape(batch_size * window, edge_count, 2)
            graph, node_embeddings = self.graph_encoder(
                flat_nodes, edge_index, flat_edges
            )
            graph = graph.reshape(batch_size, window, self.dimension)
            node_embeddings = node_embeddings.reshape(
                batch_size, window, node_count, self.dimension
            )
            temporal_input = graph + functional.relu(
                self.vehicle_projection(vehicle_features)
            )
            temporal_input = temporal_input + _positional_encoding(
                window, self.dimension, temporal_input.device, temporal_input.dtype
            ).unsqueeze(0)
            if self.spatial_only:
                temporal_state = temporal_input[:, -1]
            else:
                temporal_state = self.temporal_encoder(temporal_input)[:, -1]
            latest_nodes = node_embeddings[:, -1]
            host_table = torch.cat((
                latest_nodes,
                self.null_host.expand(batch_size, -1, -1),
            ), dim=1)

            vnf_features = batch["vnf_features"]
            current_hosts = batch["current_hosts"].clone()
            current_hosts[current_hosts < 0] = node_count
            current_embedding = host_table.gather(
                1,
                current_hosts.unsqueeze(-1).expand(-1, -1, self.dimension),
            )
            vnf_embedding = functional.relu(self.vnf_projection(vnf_features))
            request_sequence = torch.cat((vnf_embedding, current_embedding), dim=-1)
            request_output, _ = self.request_gru(request_sequence)
            lengths = (vnf_features.abs().sum(dim=-1) > 0).sum(dim=-1).clamp(min=1)
            request_sequence_state = request_output[
                torch.arange(batch_size, device=request_output.device), lengths - 1
            ]
            request_state = request_sequence_state + functional.relu(
                self.request_projection(batch["request_features"])
            )

            candidate_hosts = batch["candidate_hosts"].clone()
            candidate_hosts[candidate_hosts < 0] = node_count
            action_count, chain_length = candidate_hosts.shape[1:3]
            expanded_table = host_table[:, None].expand(
                batch_size, action_count, node_count + 1, self.dimension
            )
            candidate_embedding = expanded_table.gather(
                2,
                candidate_hosts.unsqueeze(-1).expand(
                    -1, -1, -1, self.dimension
                ),
            )
            candidate_vnf = vnf_embedding[:, None].expand(
                batch_size, action_count, chain_length, vnf_embedding.shape[-1]
            )
            candidate_sequence = torch.cat((candidate_vnf, candidate_embedding), dim=-1)
            candidate_output, _ = self.candidate_gru(
                candidate_sequence.reshape(
                    batch_size * action_count, chain_length, -1
                )
            )
            candidate_output = candidate_output.reshape(
                batch_size, action_count, chain_length, self.dimension
            )
            gather_index = (lengths - 1)[:, None, None, None].expand(
                batch_size, action_count, 1, self.dimension
            )
            candidate_sequence_state = candidate_output.gather(
                2, gather_index
            ).squeeze(2)
            candidate_scalar = functional.relu(
                self.candidate_scalar_projection(batch["candidate_features"])
            )
            combined = torch.cat((
                temporal_state[:, None].expand(-1, action_count, -1),
                request_state[:, None].expand(-1, action_count, -1),
                candidate_sequence_state,
                candidate_scalar,
                latest_nodes.mean(dim=1)[:, None].expand(-1, action_count, -1),
            ), dim=-1)
            q_values = self.q_network(combined).squeeze(-1)
            return q_values.masked_fill(~batch["candidate_mask"], -1e9)


    def _positional_encoding(length, dimension, device, dtype):
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        divisions = torch.exp(
            torch.arange(0, dimension, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros((length, dimension), device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * divisions)
        encoding[:, 1::2] = torch.cos(positions * divisions[:encoding[:, 1::2].shape[1]])
        return encoding


    class STGDDQNPolicy(Policy):
        name = "stg_ddqn"

        def __init__(self, cfg: dict, seed: int = 0, spatial_only: bool = False):
            self.cfg = cfg
            self.spatial_only = bool(spatial_only)
            self.name = "graph_ddqn" if spatial_only else "stg_ddqn"
            self.rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            device_name = str(cfg["learning"].get("device", "auto"))
            if device_name == "auto":
                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_name)
            self.online = STGQNetwork(cfg, spatial_only=spatial_only).to(self.device)
            self.target = STGQNetwork(cfg, spatial_only=spatial_only).to(self.device)
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
            self.optimizer = torch.optim.Adam(
                self.online.parameters(), lr=float(cfg["learning"]["learning_rate"])
            )
            self.replay: deque[Transition] = deque(
                maxlen=int(cfg["learning"]["replay_capacity"])
            )
            self.epsilon = float(cfg["learning"]["epsilon_start"])
            self.update_count = 0
            self.training_enabled = False

        def select(self, observation: DecisionObservation, training: bool = False) -> int:
            self.training_enabled = bool(training)
            valid = np.flatnonzero(observation.candidate_mask)
            if not len(valid):
                raise ValueError("STG-DDQN received an empty candidate mask")
            epsilon = (
                self.epsilon if training
                else float(self.cfg["learning"]["evaluation_epsilon"])
            )
            if self.rng.random() < epsilon:
                return int(self.rng.choice(valid))
            self.online.eval()
            with torch.no_grad():
                q_values = self.online(_batch_observations([observation], self.device))[0]
            return int(torch.argmax(q_values).item())

        def observe(self, transition: Transition) -> dict[str, float]:
            if not self.training_enabled:
                return {}
            compact = Transition(
                observation=transition.observation.compact_copy(),
                action_index=int(transition.action_index),
                reward=float(transition.reward),
                next_observation=(
                    transition.next_observation.compact_copy()
                    if transition.next_observation is not None else None
                ),
                terminal=bool(transition.terminal),
            )
            self.replay.append(compact)
            if len(self.replay) < int(self.cfg["learning"]["replay_warmup"]):
                return {}
            losses = []
            for _ in range(int(self.cfg["learning"]["updates_per_decision"])):
                losses.append(self._update())
            return {"loss": float(np.mean(losses))}

        def _update(self) -> float:
            learning = self.cfg["learning"]
            batch_size = min(int(learning["batch_size"]), len(self.replay))
            indices = self.rng.choice(len(self.replay), size=batch_size, replace=False)
            items = [self.replay[int(index)] for index in indices]
            observations = _batch_observations(
                [item.observation for item in items], self.device
            )
            placeholder_next = [
                item.next_observation if item.next_observation is not None else item.observation
                for item in items
            ]
            next_observations = _batch_observations(placeholder_next, self.device)
            actions = torch.as_tensor(
                [item.action_index for item in items], dtype=torch.long, device=self.device
            )
            rewards = torch.as_tensor(
                [item.reward for item in items], dtype=torch.float32, device=self.device
            )
            terminals = torch.as_tensor(
                [item.terminal for item in items], dtype=torch.float32, device=self.device
            )
            self.online.train()
            q_values = self.online(observations).gather(1, actions[:, None]).squeeze(1)
            with torch.no_grad():
                next_actions = self.online(next_observations).argmax(dim=1)
                next_values = self.target(next_observations).gather(
                    1, next_actions[:, None]
                ).squeeze(1)
                targets = rewards + float(learning["gamma"]) * (1.0 - terminals) * next_values
            loss = functional.huber_loss(
                q_values,
                targets,
                delta=float(learning["huber_delta"]),
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.online.parameters(), float(learning["gradient_clip_norm"])
            )
            self.optimizer.step()
            tau = float(learning["soft_update_factor"])
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.target.parameters(), self.online.parameters()
                ):
                    target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)
            self.update_count += 1
            self.epsilon = max(
                float(learning["epsilon_minimum"]),
                self.epsilon * float(learning["epsilon_decay"]),
            )
            return float(loss.item())

        def save(self, path: str | Path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema_version": 1,
                "spatial_only": self.spatial_only,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
            }, path)

        def load(self, path: str | Path) -> None:
            checkpoint = torch.load(Path(path), map_location=self.device)
            if bool(checkpoint.get("spatial_only", False)) != self.spatial_only:
                raise ValueError("Checkpoint spatial_only setting does not match the policy")
            self.online.load_state_dict(checkpoint["online"])
            self.target.load_state_dict(checkpoint["target"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
            self.update_count = int(checkpoint.get("update_count", 0))


    def _batch_observations(observations: Sequence[DecisionObservation], device):
        first_edges = observations[0].edge_index
        if any(
            observation.edge_index.shape != first_edges.shape
            or not np.array_equal(observation.edge_index, first_edges)
            for observation in observations[1:]
        ):
            raise ValueError("A replay mini-batch must use one fixed fog topology")

        def tensor(name, dtype=torch.float32):
            array = np.stack([getattr(observation, name) for observation in observations])
            return torch.as_tensor(array, dtype=dtype, device=device)

        return {
            "node_features": tensor("node_features"),
            "edge_features": tensor("edge_features"),
            "edge_index": torch.as_tensor(first_edges, dtype=torch.long, device=device),
            "vehicle_features": tensor("vehicle_features"),
            "request_features": tensor("request_features"),
            "vnf_features": tensor("vnf_features"),
            "current_hosts": tensor("current_hosts", dtype=torch.long),
            "candidate_hosts": tensor("candidate_hosts", dtype=torch.long),
            "candidate_features": tensor("candidate_features"),
            "candidate_mask": tensor("candidate_mask", dtype=torch.bool),
        }


else:

    class STGDDQNPolicy(Policy):
        name = "stg_ddqn"

        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ImportError(
                "PyTorch is required for STG-DDQN. Install requirements.txt, "
                "then rerun training or evaluation."
            )

