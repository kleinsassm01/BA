from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn


class ExplicitCircularConv1D(nn.Module):
    features: int
    kernel_size: int = 3
    init_std: float = 0.02
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        k = self.kernel_size
        pad = k // 2
        in_c = x.shape[-1]

        kernel = self.param(
            "kernel",
            nn.initializers.normal(self.init_std),
            (k, in_c, self.features),
        )

        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,))

        patches = jnp.stack(
            [jnp.roll(x, shift=-(j - pad), axis=1) for j in range(k)],
            axis=2,
        )
        y = jnp.einsum("bnkc,kcf->bnf", patches, kernel)

        if self.use_bias:
            y = y + bias
        return y


class CNNNQS(nn.Module):
    channels: Tuple[int, ...] = (8, 8, 8)
    kernels: Tuple[int, ...] = (3, 3, 3)
    activation: str = "gelu"
    init_std: float = 0.02

    def activation_layer_names(self):
        return ["input"] + [f"conv{i + 1}" for i in range(len(self.channels))] + ["readout_pre_sum"]

    def tangent_layer_names(self):
        return [f"conv{i + 1}" for i in range(len(self.channels))] + ["readout"]

    def parameter_block_names(self):
        return [f"conv_{i + 1}" for i in range(len(self.channels))] + ["readout"]

    def _act(self, x):
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "tanh":
            return nn.tanh(x)
        return nn.gelu(x)

    @nn.compact
    def __call__(self, sigma):
        x = sigma.astype(jnp.float32)[..., None]
        self.sow("intermediates", "input", x)

        for i, (c, k) in enumerate(zip(self.channels, self.kernels)):
            x = ExplicitCircularConv1D(
                features=c,
                kernel_size=k,
                init_std=self.init_std,
                name=f"conv_{i + 1}",
            )(x)
            x = self._act(x)
            self.sow("intermediates", f"conv{i + 1}", x)

        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="readout",
        )(x)
        self.sow("intermediates", "readout_pre_sum", x)
        return jnp.sum(x, axis=(1, 2))


class ExplicitBiRNNLayer(nn.Module):
    hidden_dim: int = 16
    init_std: float = 0.03

    @nn.compact
    def __call__(self, x):
        B, N_s, d_in = x.shape
        H = self.hidden_dim

        Wxf = self.param("Wxf", nn.initializers.normal(self.init_std), (d_in, H))
        Whf = self.param("Whf", nn.initializers.normal(self.init_std), (H, H))
        bf = self.param("bf", nn.initializers.zeros, (H,))

        Wxb = self.param("Wxb", nn.initializers.normal(self.init_std), (d_in, H))
        Whb = self.param("Whb", nn.initializers.normal(self.init_std), (H, H))
        bb = self.param("bb", nn.initializers.zeros, (H,))

        hf = jnp.zeros((B, H), x.dtype)
        outs_f = []
        for t in range(N_s):
            hf = jnp.tanh(x[:, t] @ Wxf + hf @ Whf + bf)
            outs_f.append(hf)

        hb = jnp.zeros((B, H), x.dtype)
        outs_b = []
        for t in reversed(range(N_s)):
            hb = jnp.tanh(x[:, t] @ Wxb + hb @ Whb + bb)
            outs_b.append(hb)

        return jnp.concatenate(
            [jnp.stack(outs_f, 1), jnp.stack(list(reversed(outs_b)), 1)],
            axis=-1,
        )


class BiRNNNQS(nn.Module):
    n_layers: int = 1
    hidden_dim: int = 16
    init_std: float = 0.03

    def activation_layer_names(self):
        return ["input"] + [f"rnn{i + 1}" for i in range(self.n_layers)] + ["readout_pre_sum"]

    def tangent_layer_names(self):
        return [f"rnn{i + 1}" for i in range(self.n_layers)] + ["readout"]

    def parameter_block_names(self):
        return [f"rnn_{i + 1}" for i in range(self.n_layers)] + ["readout"]

    @nn.compact
    def __call__(self, sigma):
        x = sigma.astype(jnp.float32)[..., None]
        self.sow("intermediates", "input", x)

        for i in range(self.n_layers):
            x = ExplicitBiRNNLayer(
                hidden_dim=self.hidden_dim,
                init_std=self.init_std,
                name=f"rnn_{i + 1}",
            )(x)
            self.sow("intermediates", f"rnn{i + 1}", x)

        x = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="readout",
        )(x)
        self.sow("intermediates", "readout_pre_sum", x)
        return jnp.sum(x, axis=(1, 2))


class LocalMessagePassingLayer(nn.Module):
    features: int = 7
    init_std: float = 0.02
    activation: str = "gelu"

    def _act(self, x):
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "tanh":
            return nn.tanh(x)
        return nn.gelu(x)

    @nn.compact
    def __call__(self, h):
        left = jnp.roll(h, shift=1, axis=1)
        right = jnp.roll(h, shift=-1, axis=1)
        msg_in = jnp.concatenate([left, h, right], axis=-1)

        m = nn.Dense(
            self.features,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="message",
        )(msg_in)
        m = self._act(m)

        u = nn.Dense(
            self.features,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="update",
        )(m)

        if h.shape[-1] != self.features:
            skip = nn.Dense(
                self.features,
                kernel_init=nn.initializers.normal(self.init_std),
                bias_init=nn.initializers.zeros,
                name="skip",
            )(h)
        else:
            skip = h

        return self._act(skip + u)


class GNNNQS(nn.Module):
    n_layers: int = 3
    channels: int = 7
    activation: str = "gelu"
    init_std: float = 0.02

    def activation_layer_names(self):
        return ["input"] + [f"gnn{i + 1}" for i in range(self.n_layers)] + ["readout_pre_sum"]

    def tangent_layer_names(self):
        return ["embed"] + [f"gnn{i + 1}" for i in range(self.n_layers)] + ["readout"]

    def parameter_block_names(self):
        return ["embed"] + [f"mp_{i + 1}" for i in range(self.n_layers)] + ["readout"]

    @nn.compact
    def __call__(self, sigma):
        x = sigma.astype(jnp.float32)[..., None]
        self.sow("intermediates", "input", x)

        h = nn.Dense(
            self.channels,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="embed",
        )(x)

        for i in range(self.n_layers):
            h = LocalMessagePassingLayer(
                features=self.channels,
                init_std=self.init_std,
                activation=self.activation,
                name=f"mp_{i + 1}",
            )(h)
            self.sow("intermediates", f"gnn{i + 1}", h)

        out = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(self.init_std),
            bias_init=nn.initializers.zeros,
            name="readout",
        )(h)
        self.sow("intermediates", "readout_pre_sum", out)
        return jnp.sum(out, axis=(1, 2))
