from __future__ import annotations

import numpy as np

from ..physics import local_probe_baseline, local_shell_targets
from .cka import as_2d


def node_as_2d(activations) -> np.ndarray:
    """Flatten node activations from `(samples, *shape, features)` to `(samples * sites, features)`."""
    activations = np.asarray(activations, dtype=float)
    return as_2d(activations) if activations.ndim < 3 else activations.reshape((-1, activations.shape[-1]))


def _split(n_rows: int, rng: np.random.Generator, train_frac: float, val_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one shared train/validation/test split so all probe scores are comparable."""
    indices = rng.permutation(n_rows)
    n_train = int(train_frac * n_rows)
    n_val = int(val_frac * n_rows)
    return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]


def _standardize(features: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Standardize features using only the training split to avoid validation/test leakage."""
    train_mean = features[train_idx].mean(axis=0, keepdims=True)
    train_std = features[train_idx].std(axis=0, keepdims=True) + 1.0e-8
    return (features - train_mean) / train_std


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression weights."""
    n_features = features.shape[1]
    return np.linalg.solve(features.T @ features + alpha * np.eye(n_features), features.T @ target)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    """Coefficient of determination, with a small denominator guard for constant targets."""
    residual_sum = np.sum((target - prediction) ** 2)
    total_sum = np.sum((target - target.mean()) ** 2)
    return float(1.0 - residual_sum / (total_sum + 1.0e-12))


def ridge_r2(
    features,
    target,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas=np.logspace(-4, 4, 9),
) -> float:
    """
    Fit ridge regression and return held-out test R².

    Calculation stays here:
    - convert inputs to 2D features and 1D target,
    - standardize features using the train split,
    - center target using the train split,
    - select ridge alpha by validation R²,
    - refit on train+validation,
    - report test R².
    """
    features = _standardize(as_2d(features), train_idx)
    target = np.asarray(target, dtype=float).reshape(-1)
    target = target - target[train_idx].mean()

    best_alpha = float(alphas[0])
    best_val_r2 = -np.inf

    for alpha in alphas:
        weights = _fit_ridge(features[train_idx], target[train_idx], float(alpha))
        val_r2 = _r2(target[val_idx], features[val_idx] @ weights)

        if val_r2 > best_val_r2:
            best_alpha = float(alpha)
            best_val_r2 = val_r2

    fit_idx = np.concatenate([train_idx, val_idx])
    weights = _fit_ridge(features[fit_idx], target[fit_idx], best_alpha)
    return _r2(target[test_idx], features[test_idx] @ weights)


def local_shell_probe(
    acts,
    layer_names,
    samples,
    shape,
    rng,
    *,
    rows_max: int,
    train_frac: float,
    val_frac: float,
    threshold: float,
    max_distance: int | None = None,
    metric: str = "manhattan",
) -> dict[str, np.ndarray]:
    """
    Probe whether each layer contains graph-distance information beyond simple local baselines.

    Target for each distance shell:
        y[a, i, r] = s[a, i] * mean spin at distance r from site i

    Baseline features for each `(sample, site)`:
        [site spin, nearest-neighbor average, global magnetization]

    Layer score:
        delta_r2[layer, r] =
            R²([baseline, node_activation[layer]] -> y_r)
            - R²(baseline -> y_r)

    Returned arrays:
    - `distances`: shell distances that were tested.
    - `baseline_r2`: R² using only baseline features.
    - `full_r2`: R² using baseline + layer node activations.
    - `delta_r2`: extra R² contributed by layer activations, clipped at zero.
    - `first_layer_by_distance`: first layer whose delta R² reaches `threshold`.
    - `reach_by_layer`: farthest distance each layer reaches at `threshold`.
    """
    shell_targets = local_shell_targets(samples, shape, max_r=max_distance, metric=metric)
    distances = np.array(sorted(shell_targets), dtype=int)

    baseline_features = local_probe_baseline(samples, shape, metric=metric)
    selected_rows = _select_probe_rows(baseline_features.shape[0], rows_max, rng)
    train_idx, val_idx, test_idx = _split(len(selected_rows), rng, train_frac, val_frac)
    baseline_rows = baseline_features[selected_rows]

    baseline_r2 = np.zeros(len(distances))
    full_r2 = np.zeros((len(distances), len(layer_names)))
    delta_r2 = np.zeros_like(full_r2)

    for distance_idx, distance in enumerate(distances):
        target = shell_targets[int(distance)].reshape(-1)[selected_rows]
        baseline_r2[distance_idx] = ridge_r2(baseline_rows, target, train_idx, val_idx, test_idx)

        for layer_idx, layer_name in enumerate(layer_names):
            layer_features = node_as_2d(acts[f"{layer_name}_node"])[selected_rows]
            combined_features = np.concatenate([baseline_rows, layer_features], axis=1)

            full_r2[distance_idx, layer_idx] = ridge_r2(combined_features, target, train_idx, val_idx, test_idx)
            delta_r2[distance_idx, layer_idx] = max(0.0, full_r2[distance_idx, layer_idx] - baseline_r2[distance_idx])

    return {
        "distances": distances,
        "delta_r2": delta_r2,
        "full_r2": full_r2,
        "baseline_r2": baseline_r2,
        "first_layer_by_distance": _first_layer_by_distance(delta_r2, threshold),
        "reach_by_layer": _reach_by_layer(distances, delta_r2, threshold),
    }


def logpsi_probe(
    acts,
    layer_names,
    logpsi,
    rng,
    *,
    train_frac: float,
    val_frac: float,
) -> np.ndarray:
    """
    Measure how well each full-layer activation linearly predicts exact/reference log|ψ(s)|.

    This is a global readout probe: each layer activation is flattened per sample,
    then ridge regression predicts one scalar log-amplitude per sample.
    """
    target = np.asarray(logpsi, dtype=float).reshape(-1)
    train_idx, val_idx, test_idx = _split(len(target), rng, train_frac, val_frac)

    return np.asarray(
        [ridge_r2(acts[layer_name], target, train_idx, val_idx, test_idx) for layer_name in layer_names],
        dtype=float,
    )


def _select_probe_rows(n_rows: int, rows_max: int, rng: np.random.Generator) -> np.ndarray:
    if n_rows <= rows_max:
        return np.arange(n_rows)

    return rng.choice(n_rows, size=rows_max, replace=False)


def _first_layer_by_distance(delta_r2: np.ndarray, threshold: float) -> np.ndarray:
    first_layers = np.full(delta_r2.shape[0], np.nan)

    for distance_idx in range(delta_r2.shape[0]):
        hits = np.where(delta_r2[distance_idx] >= threshold)[0]
        if len(hits):
            first_layers[distance_idx] = hits[0] + 1

    return first_layers


def _reach_by_layer(distances: np.ndarray, delta_r2: np.ndarray, threshold: float) -> np.ndarray:
    reach = np.zeros(delta_r2.shape[1])

    for layer_idx in range(delta_r2.shape[1]):
        hits = np.where(delta_r2[:, layer_idx] >= threshold)[0]
        if len(hits):
            reach[layer_idx] = distances[hits].max()

    return reach