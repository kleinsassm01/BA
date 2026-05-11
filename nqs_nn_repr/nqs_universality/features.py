from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze


def _unwrap_sowed(v):
    if isinstance(v, (tuple, list)):
        v = v[-1]
    return np.asarray(jax.device_get(v))


def extract_activations(model, params, samples, batch_size: int = 512):
    variables = {"params": params}
    names = model.activation_layer_names()
    acts = {n: [] for n in names}

    for s in range(0, len(samples), batch_size):
        batch = jnp.asarray(samples[s:s + batch_size])
        _, mut = model.apply(variables, batch, mutable=["intermediates"])
        for n in names:
            acts[n].append(_unwrap_sowed(mut["intermediates"][n]).astype(np.float32))

    return {n: np.concatenate(v, 0) for n, v in acts.items()}


def flatten_activations_for_cka(acts):
    return {n: a.reshape(a.shape[0], -1) for n, a in acts.items()}


def as_plain_dict(tree):
    return unfreeze(tree) if isinstance(tree, FrozenDict) else tree


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def flatten_tree_features(tree, n_samples: int):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return np.zeros((n_samples, 0), dtype=np.float32)

    mats = []
    for lf in leaves:
        a = np.asarray(lf)
        if a.shape[0] != n_samples:
            raise ValueError(f"Leading dim {a.shape[0]} != {n_samples}")
        mats.append(a.reshape(n_samples, -1))

    return np.concatenate(mats, axis=1).astype(np.float32)


def compute_jacobian_tree(model, params, samples, batch_size: int = 8):
    def logpsi_single(pars, sigma):
        return jnp.real(model.apply({"params": pars}, sigma[None]))[0]

    grad_fn = jax.jit(
        jax.vmap(
            jax.grad(logpsi_single, argnums=0),
            in_axes=(None, 0),
        )
    )

    trees = []
    for s in range(0, len(samples), batch_size):
        g = grad_fn(params, jnp.asarray(samples[s:s + batch_size]))
        trees.append(tree_to_numpy(g))

    return as_plain_dict(
        jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, 0), *trees)
    )


def extract_tangent_features(model, params, samples, batch_size: int = 8):
    n = samples.shape[0]

    print("    computing Jacobian tree ...")
    J_tree = compute_jacobian_tree(model, params, samples, batch_size)

    print("    flattening full tangent matrix ...")
    full_J = flatten_tree_features(J_tree, n)

    layer_J = {}
    for disp, blk in zip(model.tangent_layer_names(), model.parameter_block_names()):
        if blk in J_tree:
            layer_J[disp] = flatten_tree_features(J_tree[blk], n)
        else:
            prefix = blk + "_"
            sub_tree = {
                k: v
                for k, v in J_tree.items()
                if isinstance(k, str) and k.startswith(prefix)
            }
            if sub_tree:
                layer_J[disp] = flatten_tree_features(sub_tree, n)
            else:
                print(
                    f"    Warning: block '{blk}' not found. "
                    f"Available blocks: {list(J_tree.keys())}. Using full params."
                )
                layer_J[disp] = full_J

    return full_J, layer_J


def compute_input_saliency(model, params, samples, batch_size: int = 128):
    def logpsi_wrt_input(sigma, pars):
        return jnp.real(model.apply({"params": pars}, sigma[None]))[0]

    sal_fn = jax.jit(
        jax.vmap(
            jax.grad(logpsi_wrt_input, argnums=0),
            in_axes=(0, None),
        )
    )

    parts = []
    for s in range(0, len(samples), batch_size):
        batch = jnp.asarray(samples[s:s + batch_size])
        parts.append(np.asarray(jax.device_get(sal_fn(batch, params))))

    return np.concatenate(parts, 0).astype(np.float32)
