from __future__ import annotations

from math import prod
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..config import ModelConfig


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]


def _dtype(name: str):
    return jnp.float64 if str(name).lower() in {"float64", "double"} else jnp.float32


def _activation(name: str) -> ActivationFn:
    activations = {
        "relu": nn.relu,
        "silu": nn.silu,
        "swish": nn.silu,
        "tanh": jnp.tanh,
        "gelu": nn.gelu,
    }
    return activations.get(str(name).lower(), nn.gelu)


def _flatten_nodes(nodes: jnp.ndarray) -> jnp.ndarray:
    return nodes.reshape((nodes.shape[0], -1))


def _nearest_message(nodes: jnp.ndarray) -> jnp.ndarray:
    message_sum = sum(
        jnp.roll(nodes, shift, axis=axis)
        for axis in range(1, nodes.ndim - 1)
        for shift in (-1, 1)
    )
    neighbor_count = 2 * max(0, nodes.ndim - 2)
    return message_sum / float(max(1, neighbor_count))


def _diagonal_message_2d(nodes: jnp.ndarray) -> jnp.ndarray:
    return 0.25 * sum(
        jnp.roll(jnp.roll(nodes, dx, axis=1), dy, axis=2)
        for dx in (-1, 1)
        for dy in (-1, 1)
    )


class GraphNQS(nn.Module):
    shape: tuple[int, ...]
    cfg: ModelConfig

    def layer_names(self, kind: str) -> list[str]:
        kind = str(kind)
        if kind in {"embed", "input"}:
            return ["embed"]
        if kind in {"post_with_embed", "post+embed"}:
            return ["embed", *[f"post{block}" for block in range(self.cfg.blocks)]]
        if kind in {"post", "pre", "delta", "msg", "gate"}:
            return [f"{kind}{block}" for block in range(self.cfg.blocks)]
        raise ValueError(f"unknown activation kind: {kind}")

    def _norm(self, nodes: jnp.ndarray, name: str, dtype):
        return nn.LayerNorm(epsilon=self.cfg.layernorm_epsilon, dtype=dtype, param_dtype=dtype, name=name)(nodes)

    def _block(self, nodes: jnp.ndarray, block: int, dtype):
        cfg = self.cfg
        activation = _activation(cfg.activation)
        block_input = nodes
        normalized_nodes = self._norm(nodes, f"block{block}_prenorm", dtype) if cfg.use_prenorm else nodes

        nearest_message = _nearest_message(normalized_nodes)
        message_features = self._message_features(block_input, normalized_nodes, nearest_message)
        block_features = jnp.concatenate(message_features, axis=-1)

        hidden_features = cfg.mlp_hidden_mult * cfg.features
        delta = nn.Dense(hidden_features, dtype=dtype, param_dtype=dtype, name=f"block{block}_dense1")(block_features)
        delta = nn.Dense(cfg.features, dtype=dtype, param_dtype=dtype, name=f"block{block}_dense2")(activation(delta))

        gate = self._gate(block_features, block, dtype) if cfg.gated_updates else None
        delta = gate * delta if gate is not None else delta

        nodes = self._apply_update(block_input, delta, block, dtype)
        nodes = nodes if cfg.use_prenorm else self._norm(nodes, f"block{block}_postnorm", dtype)
        return nodes, delta, nearest_message, gate

    def _message_features(
        self,
        block_input: jnp.ndarray,
        normalized_nodes: jnp.ndarray,
        nearest_message: jnp.ndarray,
    ) -> list[jnp.ndarray]:
        cfg = self.cfg
        features = [normalized_nodes, nearest_message, normalized_nodes * nearest_message]

        if cfg.include_self_in_message:
            features.append(block_input)

        if cfg.use_diag_messages and len(self.shape) == 2:
            diagonal_message = _diagonal_message_2d(normalized_nodes)
            features.extend([diagonal_message, normalized_nodes * diagonal_message])

        return features

    def _gate(self, block_features: jnp.ndarray, block: int, dtype) -> jnp.ndarray:
        gate_logits = nn.Dense(self.cfg.features, dtype=dtype, param_dtype=dtype, name=f"block{block}_gate")(block_features)
        return jax.nn.sigmoid(gate_logits)

    def _apply_update(self, block_input: jnp.ndarray, delta: jnp.ndarray, block: int, dtype) -> jnp.ndarray:
        if not self.cfg.residual:
            return delta

        return block_input + self._layerscale(block, dtype) * delta

    def _layerscale(self, block: int, dtype):
        init = self.cfg.layerscale_init
        init = 1.0 / jnp.sqrt(jnp.asarray(max(1, self.cfg.blocks), dtype=dtype)) if init is None else init

        if not self.cfg.learned_layerscale:
            return jnp.asarray(init, dtype=dtype)

        return self.param(
            f"block{block}_layerscale",
            lambda key, shape, dtype=dtype: jnp.full(shape, init, dtype=dtype),
            (self.cfg.features,),
            dtype,
        )

    def _embed(self, samples: jnp.ndarray, dtype) -> jnp.ndarray:
        n_batch = samples.shape[0]
        nodes = samples.reshape((n_batch, *self.shape, 1))
        nodes = jnp.concatenate([nodes, jnp.ones_like(nodes)], axis=-1)
        nodes = nn.Dense(self.cfg.features, dtype=dtype, param_dtype=dtype, name="embed_dense")(nodes)
        return _activation(self.cfg.activation)(nodes)

    def _readout(self, nodes: jnp.ndarray, dtype) -> jnp.ndarray:
        normalized_nodes = self._norm(nodes, "readout_norm", dtype)
        hidden_features = self.cfg.final_mlp_hidden_mult * self.cfg.features
        site_values = nn.Dense(hidden_features, dtype=dtype, param_dtype=dtype, name="readout_dense1")(normalized_nodes)
        site_values = nn.Dense(1, dtype=dtype, param_dtype=dtype, name="readout_dense2")(_activation(self.cfg.activation)(site_values))[..., 0]

        n_spins = int(prod(int(size) for size in self.shape))
        return jnp.sum(site_values, axis=tuple(range(1, site_values.ndim))) / jnp.sqrt(jnp.asarray(n_spins, dtype=dtype))

    def _one_branch(self, samples: jnp.ndarray, return_activations: bool = False):
        dtype = _dtype(self.cfg.dtype)
        was_single_sample = samples.ndim == 1
        samples = samples[None, :] if was_single_sample else samples
        nodes = self._embed(jnp.asarray(samples, dtype=dtype), dtype)

        activations = {}
        if return_activations:
            activations["embed_node"], activations["embed"] = nodes, _flatten_nodes(nodes)

        for block in range(self.cfg.blocks):
            if return_activations:
                activations[f"pre{block}_node"], activations[f"pre{block}"] = nodes, _flatten_nodes(nodes)

            nodes, delta, message, gate = self._block(nodes, block, dtype)

            if return_activations:
                self._store_block_activations(activations, block, nodes, delta, message, gate)

        logpsi = self._readout(nodes, dtype)
        return (logpsi[0] if was_single_sample else logpsi, activations) if return_activations else logpsi[0] if was_single_sample else logpsi

    @staticmethod
    def _store_block_activations(
            activations: dict[str, jnp.ndarray],
        block: int,
        nodes: jnp.ndarray,
        delta: jnp.ndarray,
        message: jnp.ndarray,
        gate: jnp.ndarray | None,
    ) -> None:
        activations[f"delta{block}_node"], activations[f"delta{block}"] = delta, _flatten_nodes(delta)
        activations[f"msg{block}_node"], activations[f"msg{block}"] = message, _flatten_nodes(message)
        activations[f"post{block}_node"], activations[f"post{block}"] = nodes, _flatten_nodes(nodes)

        if gate is not None:
            activations[f"gate{block}_node"], activations[f"gate{block}"] = gate, _flatten_nodes(gate)

    @nn.compact
    def __call__(self, samples: jnp.ndarray, return_activations: bool = False):
        if self.cfg.symmetrize_spin_flip_output and not return_activations:
            return self._spin_flip_symmetric_output(samples)

        return self._one_branch(samples, return_activations=return_activations)

    def _spin_flip_symmetric_output(self, samples: jnp.ndarray):
        was_single_sample = samples.ndim == 1
        samples = samples[None, :] if was_single_sample else samples

        paired_samples = jnp.concatenate([samples, -samples], axis=0)
        paired_output = self._one_branch(paired_samples, return_activations=False)

        n_samples = samples.shape[0]
        symmetric_output = 0.5 * (paired_output[:n_samples] + paired_output[n_samples:])
        return symmetric_output[0] if was_single_sample else symmetric_output


def apply_with_activations(model: GraphNQS, params, samples) -> dict[str, jnp.ndarray]:
    _, activations = model.apply({"params": params}, samples, return_activations=True)
    return activations
