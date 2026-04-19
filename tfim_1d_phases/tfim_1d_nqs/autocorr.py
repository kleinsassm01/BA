from __future__ import annotations

import logging

import numpy as np

from .models import AutocorrAnalysis


def autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    if max_lag >= N:
        max_lag = N - 1
    xm = x - x.mean()
    
    # FFT-based autocorrelation is O(N log N) rather than O(N^2).
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(xm, n=nfft)
    acf_full = np.fft.irfft(F * np.conj(F), n=nfft)[:N] / N
    return acf_full[: max_lag + 1]


def integrated_autocorr_time(
    x: np.ndarray,
    max_lag: int = 400,
    sokal_c: float = 5.0,
) -> tuple[float, int, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        # too short (see result dataset)
        return float("nan"), 0, np.array([1.0])

    C = autocovariance(x, max_lag)
    C0 = C[0]
    if C0 <= 0 or not np.isfinite(C0):
        return float("nan"), 0, np.ones_like(C)
    rho = C / C0

    cum = 0.5 + np.cumsum(rho[1:])
    W = None
    for k, tau_hat in enumerate(cum, start=1):
        if k >= sokal_c * tau_hat:
            W = k
            break
    if W is None:
        W = len(cum)
    tau_int = float(cum[W - 1])
    return tau_int, int(W), rho


def analyze_vstate_energy(
    vstate,
    hamiltonian,
    n_samples: int,
    n_chains: int,
    n_discard: int,
    max_lag: int,
    sokal_c: float,
    logger: logging.Logger | None = None,
) -> AutocorrAnalysis:
    
    old_n_samples = vstate.n_samples
    old_n_discard = vstate.n_discard_per_chain
    
    try:
        vstate.n_samples = n_samples
        vstate.n_discard_per_chain = n_discard
        vstate.reset()

        series = _local_energy_series(vstate, hamiltonian)
    finally:
        vstate.n_samples = old_n_samples
        vstate.n_discard_per_chain = old_n_discard
        vstate.reset()

    tau_int, W, rho = integrated_autocorr_time(
        series, max_lag=max_lag, sokal_c=sokal_c
    )

    if logger is not None:
        logger.info(
            "  autocorr: tau_int=%.2f | window=%d | n_series=%d",
            tau_int, W, series.size,
        )

    return AutocorrAnalysis(
        lags=list(range(len(rho))),
        acf=[float(v) for v in rho],
        tau_int=float(tau_int),
        tau_int_window=int(W),
        n_samples=int(series.size),
    )


def _local_energy_series(vstate, hamiltonian) -> np.ndarray:
    import netket as nk  # noqa: F401  (import here so the module is usable
                         #              without NetKet when only the pure
                         #              autocovariance helpers are needed)

    if hasattr(vstate, "local_estimators"):
        try:
            eloc = vstate.local_estimators(hamiltonian)
            return np.asarray(eloc).real.reshape(-1)
        except Exception:
            pass

    for attr in ("local_energy", "local_value"):
        fn = getattr(vstate, attr, None)
        if fn is not None:
            try:
                eloc = fn(hamiltonian)
                return np.asarray(eloc).real.reshape(-1)
            except Exception:
                continue

    import jax.numpy as jnp

    samples = vstate.sample()
    sigma = np.asarray(samples).reshape(-1, samples.shape[-1])
    
    sigma_p, mels = hamiltonian.get_conn_padded(sigma)     # (B, K, N), (B, K)
    sigma_p_flat = np.asarray(sigma_p).reshape(-1, sigma_p.shape[-1])
    logpsi_sigma = np.asarray(vstate.log_value(sigma))     # (B,)
    logpsi_p = np.asarray(vstate.log_value(sigma_p_flat)).reshape(mels.shape)  # (B, K)

    ratio = np.exp(logpsi_p - logpsi_sigma[:, None])
    eloc = np.sum(np.asarray(mels) * ratio, axis=1)
    return np.asarray(eloc).real.reshape(-1)
