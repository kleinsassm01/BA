from __future__ import annotations

import numpy as np


def pca_scores(x, n_components: int = 2):
    x = np.asarray(x, dtype=float)
    x = x.reshape((x.shape[0], -1))
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def orthogonal_align(reference, scores):
    ref = np.asarray(reference, dtype=float) - np.mean(reference, axis=0, keepdims=True)
    src = np.asarray(scores, dtype=float) - np.mean(scores, axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(src.T @ ref, full_matrices=False)
    return src @ (u @ vt)
