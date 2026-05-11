from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import Config


torch.set_default_dtype(torch.float64)


@dataclass
class AnalyticGaussianResult:
    W: np.ndarray
    M: np.ndarray
    N: np.ndarray
    freqs: np.ndarray
    riccati_relerr: float
    min_M_eig: float
    max_real_flow_eig: float

    def as_dict(self) -> dict[str, np.ndarray | float]:
        return {
            "W": self.W,
            "M": self.M,
            "N": self.N,
            "freqs": self.freqs,
            "riccati_relerr": self.riccati_relerr,
            "min_M_eig": self.min_M_eig,
            "max_real_flow_eig": self.max_real_flow_eig,
        }


@dataclass
class TrainedGaussianResult:
    M: np.ndarray
    N: np.ndarray
    history: np.ndarray
    best_E: float


def sqrtm_spd_np(M: np.ndarray, floor: float = 1e-14) -> np.ndarray:
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, floor, None)
    return (evecs * np.sqrt(evals)) @ evecs.T


def torch_sqrtm_spd(M_np: np.ndarray) -> torch.Tensor:
    M = torch.tensor(M_np)
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=1e-14)
    return (evecs * torch.sqrt(evals)) @ evecs.T


class ComplexGaussianState(torch.nn.Module):
    # Psi(q) = exp[-1/2 q^T (M + iN) q] with M positive definite (i.e. to verify run configuration)

    def __init__(self, K_np: np.ndarray, init_noise: float = 1e-4):
        super().__init__()
        n = K_np.shape[0]
        Ksqrt = torch_sqrtm_spd(K_np)
        chol = torch.linalg.cholesky(Ksqrt + 1e-10 * torch.eye(n))
        self.raw_L = torch.nn.Parameter(chol + init_noise * torch.randn_like(chol))
        self.raw_N = torch.nn.Parameter(init_noise * torch.randn(n, n))
        self.n = n

    def matrices(self):
        L = torch.tril(self.raw_L)
        M = L @ L.T + 1e-8 * torch.eye(self.n)
        N = 0.5 * (self.raw_N + self.raw_N.T)
        return M, N


def gaussian_energy_torch(M: torch.Tensor, N: torch.Tensor, K: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    chol = torch.linalg.cholesky(M)
    Minv = torch.cholesky_inverse(chol)
    return (
        0.25 * torch.trace(M)
        + 0.25 * torch.trace(N @ Minv @ N)
        + 0.25 * torch.trace(K @ Minv)
        + 0.5 * torch.trace(A @ Minv @ N)
    )


def gaussian_energy_np(M: np.ndarray, N: np.ndarray, K: np.ndarray, A: np.ndarray) -> float:
    Minv = np.linalg.solve(M, np.eye(M.shape[0]))
    return float(
        0.25 * np.trace(M)
        + 0.25 * np.trace(N @ Minv @ N)
        + 0.25 * np.trace(K @ Minv)
        + 0.5 * np.trace(A @ Minv @ N)
    )


def train_gaussian(cfg: Config, K_np: np.ndarray, A_np: np.ndarray) -> TrainedGaussianResult:
    torch.manual_seed(cfg.seed)
    K = torch.tensor(K_np)
    A = torch.tensor(A_np)

    model = ComplexGaussianState(K_np, init_noise=cfg.init_noise)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    best_E = float("inf")
    best_M = None
    best_N = None

    print_every = max(1, cfg.steps // 20)
    save_every = max(1, cfg.steps // 200)

    for step in range(cfg.steps + 1):
        opt.zero_grad()
        M, N = model.matrices()
        E = gaussian_energy_torch(M, N, K, A)
        E_value = float(E.detach().cpu())

        if E_value < best_E:
            best_E = E_value
            best_M = M.detach().clone()
            best_N = N.detach().clone()

        if step % save_every == 0 or step == cfg.steps:
            history.append((step, E_value))
        if step % print_every == 0 or step == cfg.steps:
            print(f"step {step:6d}  E_NQS = {E_value: .12f}", flush=True)

        E.backward()
        if cfg.grad_clip and cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    assert best_M is not None and best_N is not None
    return TrainedGaussianResult(
        M=best_M.cpu().numpy(),
        N=best_N.cpu().numpy(),
        history=np.asarray(history),
        best_E=best_E,
    )


def analytic_gaussian_width(K: np.ndarray, A: np.ndarray, eig_tol: float = 1e-9) -> AnalyticGaussianResult:
    # Exact complex Gaussian width W = M + iN for H = 1/2 p^T p + 1/2 q^T K q - 1/2 (p^T A q + q^T A^T p)
    n = K.shape[0]
    G = np.block([[K, -A.T], [-A, np.eye(n)]])
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    F = J @ G

    evals, evecs = np.linalg.eig(F) # scipy version: evals, evecs = la.eig(F)

    imag = np.imag(evals)
    idx = np.where(imag > eig_tol)[0]
    if len(idx) != n:
        idx = np.argsort(imag)[-n:]
        if np.any(imag[idx] <= 0.0):
            raise RuntimeError(f"Could not identify {n} positive-frequency modes.")

    idx = idx[np.argsort(imag[idx])]
    freqs = imag[idx]
    Q = evecs[:n, idx]
    P = evecs[n:, idx]
    W = -1j * np.linalg.solve(Q.T, P.T).T
    W = 0.5 * (W + W.T)

    M = 0.5 * (np.real(W) + np.real(W).T)
    N = 0.5 * (np.imag(W) + np.imag(W).T)

    riccati = W @ W + 1j * (A.T @ W + W @ A) - K
    riccati_relerr = float(np.linalg.norm(riccati) / max(np.linalg.norm(K), 1e-300))
    min_M_eig = float(np.linalg.eigvalsh(M).min())
    max_real_eval = float(np.max(np.abs(np.real(evals))))

    if min_M_eig <= 0.0:
        raise RuntimeError(f"Analytic Re(W) not positive definite: min eig = {min_M_eig:.6e}")

    return AnalyticGaussianResult(
        W=W,
        M=M,
        N=N,
        freqs=freqs,
        riccati_relerr=riccati_relerr,
        min_M_eig=min_M_eig,
        max_real_flow_eig=max_real_eval,
    )
