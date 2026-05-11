from __future__ import annotations

import numpy as np
import optax
import jax
import netket as nk

from .model_registry import model_family
from .models import BiRNNNQS, CNNNQS, GNNNQS


def train_nqs(hi, H, model, seed: int = 0, n_samples: int = 4096, n_iter: int = 1200, lr: float = 0.005, use_sr: bool = False):
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples, seed=seed)

    try:
        vstate.chunk_size = 512
    except Exception:
        pass

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_iter,
        alpha=0.05,
    )
    optimizer = optax.adam(learning_rate=schedule)
    preconditioner = nk.optimizer.SR(diag_shift=0.01) if use_sr else None

    gs = nk.driver.VMC(
        H,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
    )
    log = nk.logging.RuntimeLog()

    for _ in range(100):
        vstate.sample()

    gs.run(n_iter=n_iter, out=log)

    return {
        "model": model,
        "vstate": vstate,
        "log": log,
        "energy": vstate.expect(H),
    }


def stats_to_jsonable(stats):
    out = {"repr": str(stats)}
    for attr in ("mean", "error_of_mean", "variance", "R_hat", "tau_corr"):
        if hasattr(stats, attr):
            try:
                out[attr] = float(np.asarray(getattr(stats, attr)).real)
            except Exception:
                out[attr] = str(getattr(stats, attr))
    return out


def count_params(params_or_variables) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params_or_variables)))


def model_metadata(model, parameters, energy):
    meta = {
        "family": model_family(model),
        "n_params": count_params(parameters),
        "activation_layer_order": model.activation_layer_names(),
        "tangent_layer_order": model.tangent_layer_names(),
        "parameter_block_order": model.parameter_block_names(),
        "final_energy": stats_to_jsonable(energy),
    }

    if isinstance(model, CNNNQS):
        meta.update(
            channels=list(model.channels),
            kernels=list(model.kernels),
            activation=model.activation,
            init_std=model.init_std,
        )
    elif isinstance(model, BiRNNNQS):
        meta.update(
            n_layers=model.n_layers,
            hidden_dim=model.hidden_dim,
            init_std=model.init_std,
        )
    elif isinstance(model, GNNNQS):
        meta.update(
            n_layers=model.n_layers,
            channels=model.channels,
            activation=model.activation,
            init_std=model.init_std,
        )

    return meta
