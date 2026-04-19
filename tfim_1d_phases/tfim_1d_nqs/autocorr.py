"""
Autocorrelation-time analysis for Markov-chain Monte Carlo samples.

The *integrated* autocorrelation time tau_int is the right quantity to
characterize the efficiency of an MCMC sampler: for a time series of length
N_samples, the effective number of independent samples is

    N_eff ~ N_samples / (2 * tau_int),

and the statistical error of a sample-mean estimator scales like
sqrt(2 * tau_int / N_samples) * sigma, where sigma is the sample std.

Estimation uses the standard automated-windowing procedure of Sokal
(Monte Carlo Methods in Statistical Mechanics, 1996):

    tau_int(W) = 1/2 + sum_{t=1}^{W} rho(t)
    W* = first W such that W >= c * tau_int(W),    c in [4, 10]

where rho(t) = C(t) / C(0) is the normalized autocovariance function.

This module only depends on numpy and is independent of NetKet so that the
same machinery can be applied to any 1-D sample series (energy, order
parameter, ...). For convenience there is also a helper that runs a
dedicated MCMC chain at a fixed variational state and returns the full
AutocorrAnalysis object.
"""
from __future__ import annotations

import logging

import numpy as np

from .models import AutocorrAnalysis


def autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Biased autocovariance estimator C(t) of a 1-D real series x, for
    t = 0, 1, ..., max_lag.  Normalized by N (not N-t) to keep the estimator
    positive semi-definite, which is the standard choice in MCMC analyses.
    """
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
    """
    Compute the integrated autocorrelation time tau_int of series x using
    Sokal's automated windowing.

    Returns
    -------
    tau_int : float
        The integrated autocorrelation time (in units of MC steps).
    window : int
        The window W* chosen by the Sokal criterion.
    rho : np.ndarray
        The normalized autocorrelation function rho(t) = C(t)/C(0) from
        t=0 (where rho=1 by construction) up to t=max_lag.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4:
        # Too short to estimate anything meaningful.
        return float("nan"), 0, np.array([1.0])

    C = autocovariance(x, max_lag)
    C0 = C[0]
    if C0 <= 0 or not np.isfinite(C0):
        return float("nan"), 0, np.ones_like(C)
    rho = C / C0

    # Cumulative sum:  tau_int(W) = 1/2 + sum_{t=1..W} rho(t)
    cum = 0.5 + np.cumsum(rho[1:])  # cum[W-1] == tau_int(W)
    W = None
    for k, tau_hat in enumerate(cum, start=1):
        if k >= sokal_c * tau_hat:
            W = k
            break
    if W is None:
        # Chain too short -- use the largest lag available as a lower bound.
        W = len(cum)
    tau_int = float(cum[W - 1])
    return tau_int, int(W), rho


# ---------------------------------------------------------------------------
# NetKet integration: run a dedicated MC chain at an already-trained state
# and return the full AutocorrAnalysis.
# ---------------------------------------------------------------------------

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
    """
    Run a dedicated MC chain at the current vstate parameters and compute
    tau_int of the local-energy time series.

    We temporarily reconfigure ``vstate`` with n_chains / n_samples / n_discard
    from AutocorrConfig, then call ``vstate.sample()`` to obtain the raw
    samples, and evaluate the local energy of the Hamiltonian on each sample.
    That yields a 1-D series of length n_samples on which we run the Sokal
    windowing estimator.
    """
    # Snapshot previous settings so we can restore them afterwards.
    old_n_samples = vstate.n_samples
    old_n_discard = vstate.n_discard_per_chain
    # NetKet's sampler has n_chains baked in; we don't rebuild the sampler
    # here because we want to measure tau_int *as the trainer saw it*.
    try:
        vstate.n_samples = n_samples
        vstate.n_discard_per_chain = n_discard
        vstate.reset()

        # Draw a long MC chain at the trained parameters and obtain a 1-D
        # series of local-energy values to feed into the ACF estimator.
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
    """
    Return a 1-D real time series of the local-energy values sampled from
    the current vstate, of total length n_chains * n_per_chain.

    NetKet's public API for extracting local energies has changed across
    versions, so we try three paths in order and fall back gracefully.
    """
    import netket as nk  # noqa: F401  (import here so the module is usable
                         #              without NetKet when only the pure
                         #              autocovariance helpers are needed)

    # Path 1: recent NetKet exposes ``local_estimators`` on the MCState.
    if hasattr(vstate, "local_estimators"):
        try:
            eloc = vstate.local_estimators(hamiltonian)
            return np.asarray(eloc).real.reshape(-1)
        except Exception:
            pass

    # Path 2: some NetKet versions expose ``local_energy`` / ``local_value``.
    for attr in ("local_energy", "local_value"):
        fn = getattr(vstate, attr, None)
        if fn is not None:
            try:
                eloc = fn(hamiltonian)
                return np.asarray(eloc).real.reshape(-1)
            except Exception:
                continue

    # Path 3: manual fallback -- draw samples and evaluate O_loc via
    # operator.get_conn_padded. This is exactly what vstate.expect uses
    # internally to build Stats; we just keep every per-sample value instead
    # of folding into a mean.
    import jax.numpy as jnp

    samples = vstate.sample()                    # (n_chains, n_per_chain, N)
    sigma = np.asarray(samples).reshape(-1, samples.shape[-1])

    # get_conn_padded returns: connected configurations sigma' and matrix
    # elements mels, both padded so every row has the same length.
    sigma_p, mels = hamiltonian.get_conn_padded(sigma)     # (B, K, N), (B, K)
    sigma_p_flat = np.asarray(sigma_p).reshape(-1, sigma_p.shape[-1])
    logpsi_sigma = np.asarray(vstate.log_value(sigma))     # (B,)
    logpsi_p = np.asarray(vstate.log_value(sigma_p_flat)).reshape(mels.shape)  # (B, K)

    # O_loc(sigma) = sum_{sigma'} <sigma|H|sigma'> psi(sigma')/psi(sigma)
    ratio = np.exp(logpsi_p - logpsi_sigma[:, None])
    eloc = np.sum(np.asarray(mels) * ratio, axis=1)
    return np.asarray(eloc).real.reshape(-1)
