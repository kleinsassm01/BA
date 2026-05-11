from __future__ import annotations

import numpy as np

from .config import ExperimentConfig


def saliency_covariance_by_distance(saliency, pbc: bool = True):
    n_samp, N_sites = saliency.shape
    sal = saliency.astype(np.float64)
    sal = sal - sal.mean(axis=0, keepdims=True)

    max_d = N_sites // 2
    C = np.zeros(max_d + 1)

    for d in range(max_d + 1):
        acc = 0.0
        cnt = 0
        for i in range(N_sites):
            j = (i + d) % N_sites
            if not pbc and j < i and d > 0:
                continue
            acc += np.mean(sal[:, i] * sal[:, j])
            cnt += 1
        C[d] = acc / max(cnt, 1)
    return C


def linear_cka(X, Y, eps: float = 1e-10):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if np.isnan(X).any() or np.isnan(Y).any():
        return np.nan

    X -= X.mean(0, keepdims=True)
    Y -= Y.mean(0, keepdims=True)

    x_keep = X.std(0) > eps
    y_keep = Y.std(0) > eps
    if x_keep.sum() == 0 or y_keep.sum() == 0:
        return np.nan

    X = X[:, x_keep]
    Y = Y[:, y_keep]

    xv = np.sum(X * X)
    yv = np.sum(Y * Y)
    if xv < eps or yv < eps:
        return np.nan

    n, px = X.shape
    _, py = Y.shape

    if max(px, py) < n:
        num = np.linalg.norm(X.T @ Y, "fro") ** 2
        den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    else:
        K = X @ X.T
        L = Y @ Y.T
        num = np.sum(K * L)
        den = np.sqrt(np.sum(K * K) * np.sum(L * L))

    return num / den if den > eps else np.nan


def cka_matrix(fa, fb, la=None, lb=None):
    la = la or list(fa.keys())
    lb = lb or list(fb.keys())
    M = np.zeros((len(la), len(lb)), dtype=np.float64)

    for i, a in enumerate(la):
        for j, b in enumerate(lb):
            M[i, j] = linear_cka(fa[a], fb[b])
    return M, la, lb


def model_pair_cka_matrix(features_dict, model_order):
    n = len(model_order)
    M = np.zeros((n, n), dtype=np.float64)

    for i, a in enumerate(model_order):
        M[i, i] = 1.0
        for j, b in enumerate(model_order):
            if i == j:
                continue
            M[i, j] = linear_cka(features_dict[a], features_dict[b])
    return M


def rbf_cka(X, Y, sigma_frac: float = 0.5, eps: float = 1e-10):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if np.isnan(X).any() or np.isnan(Y).any():
        return np.nan

    x_keep = X.std(0) > eps
    y_keep = Y.std(0) > eps
    if x_keep.sum() == 0 or y_keep.sum() == 0:
        return np.nan

    X = X[:, x_keep]
    Y = Y[:, y_keep]
    X -= X.mean(0)
    Y -= Y.mean(0)

    n = X.shape[0]

    def rbf_kernel(Z, frac):
        D = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        med = np.median(D[D > 0]) + eps
        return np.exp(-D / (2 * frac * med))

    K = rbf_kernel(X, sigma_frac)
    L = rbf_kernel(Y, sigma_frac)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    num = np.sum(Kc * Lc)
    den = np.sqrt(np.sum(Kc ** 2) * np.sum(Lc ** 2))
    return num / den if den > eps else np.nan


def compute_ntk(tangent_features):
    J = tangent_features.astype(np.float64)
    J -= J.mean(0, keepdims=True)
    return J @ J.T


def ntk_kernel_alignment(K1, K2, eps: float = 1e-10):
    n = K1.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K1c = H @ K1 @ H
    K2c = H @ K2 @ H

    num = np.sum(K1c * K2c)
    den = np.sqrt(np.sum(K1c ** 2) * np.sum(K2c ** 2))
    return num / den if den > eps else np.nan


def ntk_alignment_matrix(full_tangents, model_order):
    ntks = {m: compute_ntk(full_tangents[m]) for m in model_order}
    n = len(model_order)
    M = np.zeros((n, n), dtype=np.float64)

    for i, a in enumerate(model_order):
        for j, b in enumerate(model_order):
            M[i, j] = ntk_kernel_alignment(ntks[a], ntks[b])
    return M


def ntk_eigenspectrum(tangent_features, top_k: int = 50):
    K = compute_ntk(tangent_features)
    evals = np.linalg.eigvalsh(K)[::-1]
    return evals[:min(top_k, len(evals))]


def procrustes_distance(X, Y, eps: float = 1e-10):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)

    x_keep = X.std(0) > eps
    y_keep = Y.std(0) > eps
    if x_keep.sum() == 0 or y_keep.sum() == 0:
        return float("nan")

    X = X[:, x_keep]
    Y = Y[:, y_keep]
    X -= X.mean(0, keepdims=True)
    Y -= Y.mean(0, keepdims=True)
    X /= np.linalg.norm(X, "fro") + eps
    Y /= np.linalg.norm(Y, "fro") + eps

    n_samp = X.shape[0]
    dx = X.shape[1]
    dy = Y.shape[1]

    if dx != dy:
        d = min(dx, dy, n_samp)
        if dx > d:
            U, S, _ = np.linalg.svd(X, full_matrices=False)
            X = U[:, :d] * S[:d]
        if dy > d:
            U, S, _ = np.linalg.svd(Y, full_matrices=False)
            Y = U[:, :d] * S[:d]
        X /= np.linalg.norm(X, "fro") + eps
        Y /= np.linalg.norm(Y, "fro") + eps

    try:
        U, _, Vt = np.linalg.svd(X.T @ Y)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            U, _, Vt = scipy_svd(X.T @ Y)
        except Exception:
            return float("nan")

    R = U @ Vt
    return float(np.linalg.norm(X @ R - Y, "fro"))


def procrustes_matrix(features_dict, model_order):
    n = len(model_order)
    M = np.zeros((n, n), dtype=np.float64)

    for i, a in enumerate(model_order):
        for j, b in enumerate(model_order):
            M[i, j] = 0.0 if i == j else procrustes_distance(features_dict[a], features_dict[b])
    return M


def make_zz_targets(samples, distance: int = 1, pbc: bool = True):
    z = samples.astype(np.float32)
    if pbc:
        return z * np.roll(z, -distance, axis=1)

    shifted = np.zeros_like(z)
    N_sites = z.shape[1]
    if distance < N_sites:
        shifted[:, :N_sites - distance] = z[:, distance:]
    return z * shifted


def ridge_probe_sitewise(site_features, site_targets, alpha: float = 1e-3, train_frac: float = 0.7, seed: int = 0):
    X = site_features.reshape(-1, site_features.shape[-1]).astype(np.float64)
    y = site_targets.reshape(-1).astype(np.float64)

    if np.isnan(X).any() or np.isnan(y).any():
        return {"r2": float("nan"), "accuracy": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(train_frac * len(X))
    tr = idx[:n_tr]
    te = idx[n_tr:]

    Xtr = X[tr]
    ytr = y[tr]
    Xte = X[te]
    yte = y[te]

    mu = Xtr.mean(0, keepdims=True)
    std = Xtr.std(0, keepdims=True)
    keep = std.ravel() > 1e-12

    if keep.sum() == 0:
        pred = np.full_like(yte, ytr.mean())
    else:
        Xtr = (Xtr[:, keep] - mu[:, keep]) / (std[:, keep] + 1e-8)
        Xte = (Xte[:, keep] - mu[:, keep]) / (std[:, keep] + 1e-8)
        Xtr = np.c_[Xtr, np.ones(len(Xtr))]
        Xte = np.c_[Xte, np.ones(len(Xte))]

        reg = alpha * np.eye(Xtr.shape[1])
        reg[-1, -1] = 0.0
        try:
            w = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(Xtr.T @ Xtr + reg) @ (Xtr.T @ ytr)
        pred = Xte @ w

    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "accuracy": float(np.mean(np.sign(pred) == np.sign(yte))),
    }


def compute_local_term_decoding(activations, samples, metadata, cfg: ExperimentConfig):
    targets = make_zz_targets(samples, 1, metadata["hamiltonian"]["pbc"])
    baseline_feat = samples[..., None].astype(np.float32)

    results = {
        "_input_baseline": ridge_probe_sitewise(
            baseline_feat,
            targets,
            cfg.ridge_alpha,
            cfg.probe_train_frac,
            123,
        )
    }

    for name in metadata["model_order"]:
        if name not in activations:
            continue
        results[name] = {}
        for layer in metadata["models"][name]["activation_layer_order"]:
            results[name][layer] = ridge_probe_sitewise(
                activations[name][layer],
                targets,
                cfg.ridge_alpha,
                cfg.probe_train_frac,
                123,
            )
    return results


def compute_multidistance_decoding(activations, samples, metadata, cfg: ExperimentConfig, max_d: int | None = None):
    max_d = cfg.max_decode_distance if max_d is None else max_d
    pbc = metadata["hamiltonian"]["pbc"]
    results = {}

    for name in metadata["model_order"]:
        if name not in activations:
            continue
        layers = metadata["models"][name]["activation_layer_order"]
        first_hidden = layers[1]
        feats = activations[name][first_hidden]
        r2s = []

        for d in range(1, max_d + 1):
            tgt = make_zz_targets(samples, d, pbc)
            r2s.append(
                ridge_probe_sitewise(
                    feats,
                    tgt,
                    cfg.ridge_alpha,
                    cfg.probe_train_frac,
                    42,
                )["r2"]
            )

        results[name] = {"layer": first_hidden, "r2_vs_d": r2s}
    return results


def compute_zz_correlation(samples, max_d: int, pbc: bool = True):
    z = samples.astype(np.float64)
    N_sites = z.shape[1]
    C = np.zeros(max_d + 1)

    for d in range(max_d + 1):
        if pbc:
            C[d] = np.mean(z * np.roll(z, -d, axis=1))
        else:
            C[d] = 0.0 if d >= N_sites else np.mean(z[:, :N_sites - d] * z[:, d:])
    return C
