from __future__ import annotations

import numpy as np


def as_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    return x.reshape((x.shape[0], -1))


def _center(x):
    x = as_2d(x)
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x, y):
    """Linear centered kernel alignment.

    Kornblith et al., "Similarity of Neural Network Representations Revisited",
    Eq. (4), PDF p. 4:

        CKA(K,L) = HSIC(K,L) / sqrt(HSIC(K,K) HSIC(L,L))

    For centered linear activations X,Y this becomes

        CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F).

    """
    x = _center(x)
    y = _center(y)
    numerator = np.sum((x.T @ y) ** 2)
    denominator = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(numerator / denominator) if denominator > 0 else 0.0


def within_net_cka(acts: dict[str, np.ndarray], names: list[str]):
    xs = [_center(acts[name]) for name in names]
    norms = [np.linalg.norm(x.T @ x, "fro") for x in xs]
    mat = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            den = norms[i] * norms[j]
            val = float(np.sum((xs[i].T @ xs[j]) ** 2) / den) if den > 0 else 0.0
            mat[i, j] = mat[j, i] = val
    return mat


def between_net_cka(acts_a: dict[str, np.ndarray], names_a: list[str],
                    acts_b: dict[str, np.ndarray], names_b: list[str]):
    xa = [_center(acts_a[name]) for name in names_a]
    xb = [_center(acts_b[name]) for name in names_b]
    na = [np.linalg.norm(x.T @ x, "fro") for x in xa]
    nb = [np.linalg.norm(x.T @ x, "fro") for x in xb]
    mat = np.zeros((len(names_a), len(names_b)))
    for i, x in enumerate(xa):
        for j, y in enumerate(xb):
            den = na[i] * nb[j]
            mat[i, j] = float(np.sum((x.T @ y) ** 2) / den) if den > 0 else 0.0
    return mat
