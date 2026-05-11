from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .gaussian import sqrtm_spd_np
from .geometry import Problem


@dataclass
class ObservableDefinition:
    nqs: np.ndarray
    analytic: np.ndarray
    title: str
    label: str
    signed: bool = True


def observables_np(
    M: np.ndarray,
    N: np.ndarray,
    Dx: np.ndarray,
    Dy: np.ndarray,
    Dz: np.ndarray,
    mass: float,
    V: np.ndarray,
):
    Minv = np.linalg.solve(M, np.eye(M.shape[0]))
    Cqq = 0.5 * Minv
    Cpp = 0.5 * (M + N @ Minv @ N)

    q_var = np.diag(Cqq)
    p_var = np.diag(Cpp)
    gradx_var = np.diag(Dx @ Cqq @ Dx.T)
    grady_var = np.diag(Dy @ Cqq @ Dy.T)
    gradz_var = np.diag(Dz @ Cqq @ Dz.T)
    gradxy_cov = np.diag(Dx @ Cqq @ Dy.T)
    gradxz_cov = np.diag(Dx @ Cqq @ Dz.T)

    Cpq_sym = -0.5 * (N @ Minv)
    pDxq_sym = np.sum(Cpq_sym * Dx, axis=1)
    pDyq_sym = np.sum(Cpq_sym * Dy, axis=1)
    pDzq_sym = np.sum(Cpq_sym * Dz, axis=1)

    m2q = (mass ** 2) * q_var
    transverse = grady_var + gradz_var + m2q

    rho = 0.5 * (p_var + gradx_var + transverse)
    shift_density = -V * pDxq_sym
    h = rho + shift_density

    Ttt_cov = (
        0.5 * (1.0 + V ** 2) * (p_var + gradx_var)
        - 2.0 * V * pDxq_sym
        + 0.5 * (1.0 - V ** 2) * transverse
    )
    Ttx_cov = pDxq_sym - 0.5 * V * (p_var + gradx_var) + 0.5 * V * transverse
    Tty_cov = pDyq_sym - V * gradxy_cov
    Ttz_cov = pDzq_sym - V * gradxz_cov

    return {
        "Cqq": Cqq,
        "Cpp": Cpp,
        "q_var": q_var,
        "p_var": p_var,
        "gradx_var": gradx_var,
        "grady_var": grady_var,
        "gradz_var": gradz_var,
        "gradxy_cov": gradxy_cov,
        "gradxz_cov": gradxz_cov,
        "pDxq_sym": pDxq_sym,
        "pDyq_sym": pDyq_sym,
        "pDzq_sym": pDzq_sym,
        "rho": rho,
        "shift_density": shift_density,
        "h": h,
        "Ttt_cov": Ttt_cov,
        "Ttx_cov": Ttx_cov,
        "Tty_cov": Tty_cov,
        "Ttz_cov": Ttz_cov,
    }


def build_derived_observables(
    cfg: Config,
    prob: Problem,
    M_nqs: np.ndarray,
    N_nqs: np.ndarray,
    M_an: np.ndarray,
    N_an: np.ndarray,
) -> dict[str, ObservableDefinition]:
    M0 = sqrtm_spd_np(prob.K)
    N0 = np.zeros_like(M0)

    obs0 = observables_np(M0, N0, prob.Dx, prob.Dy, prob.Dz, cfg.mass, np.zeros_like(prob.V))
    obs_ref_same_metric = observables_np(M0, N0, prob.Dx, prob.Dy, prob.Dz, cfg.mass, prob.V)
    obs_nqs = observables_np(M_nqs, N_nqs, prob.Dx, prob.Dy, prob.Dz, cfg.mass, prob.V)
    obs_an = observables_np(M_an, N_an, prob.Dx, prob.Dy, prob.Dz, cfg.mass, prob.V)

    dVol = prob.dVol

    return {
        "rho_sub_c": ObservableDefinition(
            (obs_nqs["rho"] - obs0["rho"]) / dVol,
            (obs_an["rho"] - obs0["rho"]) / dVol,
            r"subtracted $\rho/dV$",
            r"$\rho_{\rm sub}/dV$",
            True,
        ),
        "h_sub_c": ObservableDefinition(
            (obs_nqs["h"] - obs_ref_same_metric["h"]) / dVol,
            (obs_an["h"] - obs_ref_same_metric["h"]) / dVol,
            r"same-metric subtracted $h/dV$",
            r"$h_{\rm sub}/dV$",
            True,
        ),
        "shift_sub_c": ObservableDefinition(
            (obs_nqs["shift_density"] - obs_ref_same_metric["shift_density"]) / dVol,
            (obs_an["shift_density"] - obs_ref_same_metric["shift_density"]) / dVol,
            r"same-metric subtracted shift contribution$/dV$",
            "shift contribution / dV",
            True,
        ),
        "Ttt_sub_c": ObservableDefinition(
            (obs_nqs["Ttt_cov"] - obs_ref_same_metric["Ttt_cov"]) / dVol,
            (obs_an["Ttt_cov"] - obs_ref_same_metric["Ttt_cov"]) / dVol,
            r"same-metric subtracted $T_{tt}/dV$",
            r"$T_{tt}^{\rm sub}/dV$",
            True,
        ),
        "Ttx_sub_c": ObservableDefinition(
            (obs_nqs["Ttx_cov"] - obs_ref_same_metric["Ttx_cov"]) / dVol,
            (obs_an["Ttx_cov"] - obs_ref_same_metric["Ttx_cov"]) / dVol,
            r"same-metric subtracted $T_{tx}/dV$",
            r"$T_{tx}^{\rm sub}/dV$",
            True,
        ),
        "pDxq_sym": ObservableDefinition(
            obs_nqs["pDxq_sym"],
            obs_an["pDxq_sym"],
            r"phase correlation $\langle pD_xq\rangle_{\rm sym}$",
            r"$\langle pD_xq\rangle_{\rm sym}$",
            True,
        ),
        "q_var_sub": ObservableDefinition(
            obs_nqs["q_var"] - obs0["q_var"],
            obs_an["q_var"] - obs0["q_var"],
            r"subtracted field variance $\langle q^2\rangle - \langle q^2\rangle_0$",
            r"$\Delta\langle q^2\rangle$",
            True,
        ),
    }
