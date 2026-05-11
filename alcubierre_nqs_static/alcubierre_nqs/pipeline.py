from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config, save_config
from .gaussian import analytic_gaussian_width, gaussian_energy_np, train_gaussian
from .geometry import make_problem
from .observables import build_derived_observables
from .plots import choose_plot_halfwidth, generate_all_plots


def rel_frob(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300))


def rel_vec(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-300))


def print_key_files(cfg: Config, derived_keys: list[str]) -> None:
    print("Key files:")
    print("  validation_config_3p1.json")
    print("  validation_summary_3p1.json")
    print("  training_history.csv")
    print("  energy_training_vs_analytic.png")
    print(f"  bubble_profile_compare_{cfg.bubble_compare_key}.png")
    for key in derived_keys:
        print(f"  compare_{key}_zoom_heatmap.png")
        print(f"  error_{key}_zoom_heatmap.png")
        print(f"  surface_{key}_zoom.png")
    print("  linecut_rho_sub_c.png")
    print("  linecut_Ttx_sub_c.png")
    print("  linecut_q_var_sub.png")


def build_summary(
    cfg: Config,
    prob,
    exact,
    M_nqs: np.ndarray,
    N_nqs: np.ndarray,
    M_an: np.ndarray,
    N_an: np.ndarray,
    best_E: float,
    E_nqs: float,
    E_an: float,
    E_half_freq: float,
    derived,
) -> dict:
    summary = {
        **asdict(cfg),
        "n_sites": cfg.Nx * cfg.Ny * cfg.Nz,
        "dx": prob.dx,
        "dy": prob.dy,
        "dz": prob.dz,
        "dVol": prob.dVol,
        "plot_halfwidth_used": choose_plot_halfwidth(cfg),
        "min_omega2": float(np.linalg.eigvalsh(prob.Omega2).min()),
        "analytic_E_gaussian_formula": E_an,
        "analytic_E_half_sum_freq": E_half_freq,
        "nqs_best_E": best_E,
        "nqs_recomputed_E": E_nqs,
        "energy_abs_error": E_nqs - E_an,
        "energy_rel_error": (E_nqs - E_an) / max(abs(E_an), 1e-300),
        "M_rel_frobenius_error": rel_frob(M_nqs, M_an),
        "N_rel_frobenius_error": rel_frob(N_nqs, N_an),
        "analytic_riccati_relerr": exact.riccati_relerr,
        "analytic_min_M_eig": exact.min_M_eig,
        "max_real_part_flow_eigs": exact.max_real_flow_eig,
    }

    for key, obs in derived.items():
        summary[f"{key}_rel_l2_error"] = rel_vec(obs.nqs, obs.analytic)
        summary[f"{key}_max_abs_error"] = float(np.max(np.abs(obs.nqs - obs.analytic)))

    return summary


def save_numerical_outputs(
    outdir: Path,
    cfg: Config,
    hist: np.ndarray,
    summary: dict,
    M_nqs: np.ndarray,
    N_nqs: np.ndarray,
    M_an: np.ndarray,
    N_an: np.ndarray,
    freqs: np.ndarray,
) -> None:
    save_config(cfg, outdir / "validation_config_3p1.json")

    with (outdir / "validation_summary_3p1.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    np.savetxt(outdir / "training_history.csv", hist, delimiter=",", header="step,E_NQS", comments="")

    if cfg.save_matrices:
        np.savez_compressed(
            outdir / "validation_matrices_3p1.npz",
            M_nqs=M_nqs,
            N_nqs=N_nqs,
            M_analytic=M_an,
            N_analytic=N_an,
            freqs=freqs,
        )


def train_validate(cfg: Config) -> dict:
    """Full run."""
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prob = make_problem(cfg)
    n = cfg.Nx * cfg.Ny * cfg.Nz
    min_omega2 = float(np.linalg.eigvalsh(prob.Omega2).min())

    print("\n3+1D Gaussian baseline validation")
    print(f"grid: {cfg.Nx} x {cfg.Ny} x {cfg.Nz}  sites={n}")
    print(f"v={cfg.v}, R={cfg.R}, sigma={cfg.sigma}, mass={cfg.mass}")
    print(f"plot halfwidth = {choose_plot_halfwidth(cfg):.6f}")
    print(f"bubble profile mode = {cfg.bubble_surface_mode}")
    print(f"bubble comparison key = {cfg.bubble_compare_key}")
    print(f"min eig(K - A^T A) = {min_omega2:.12e}")
    if min_omega2 <= 0.0:
        print("WARNING: K - A^T A is not positive - might leed to unstable version.")

    print("\nComputing analytic Gaussian width...")
    exact = analytic_gaussian_width(prob.K, prob.A)
    M_an, N_an = exact.M, exact.N
    E_an = gaussian_energy_np(M_an, N_an, prob.K, prob.A)
    E_half_freq = 0.5 * float(np.sum(exact.freqs))
    print(f"analytic E from Gaussian formula = {E_an:.12f}")
    print(f"analytic E from 1/2 sum omega    = {E_half_freq:.12f}")
    print(f"Riccati relative error          = {exact.riccati_relerr:.3e}")
    print(f"min eig Re(W)                   = {exact.min_M_eig:.3e}")

    print("\nTraining NQS Gaussian...")
    trained = train_gaussian(cfg, prob.K, prob.A)
    M_nqs, N_nqs, hist, best_E = trained.M, trained.N, trained.history, trained.best_E
    E_nqs = gaussian_energy_np(M_nqs, N_nqs, prob.K, prob.A)
    print(f"best NQS E                      = {best_E:.12f}")
    print(f"recomputed NQS E                = {E_nqs:.12f}")
    print(f"energy error E_NQS - E_analytic = {E_nqs - E_an:.12e}")

    derived = build_derived_observables(cfg, prob, M_nqs, N_nqs, M_an, N_an)
    summary = build_summary(
        cfg, prob, exact, M_nqs, N_nqs, M_an, N_an, best_E, E_nqs, E_an, E_half_freq, derived
    )

    save_numerical_outputs(outdir, cfg, hist, summary, M_nqs, N_nqs, M_an, N_an, exact.freqs)

    if cfg.make_plots:
        print("\nGenerating plots...")
        generate_all_plots(outdir, cfg, prob, derived, hist, E_an)
    else:
        print("\nSkipping plot generation because --no-plots was used.")

    print("\nValidation errors")
    print(f"relative Frobenius error M   = {summary['M_rel_frobenius_error']:.6e}")
    print(f"relative Frobenius error N   = {summary['N_rel_frobenius_error']:.6e}")
    print(f"relative L2 error rho_sub_c  = {summary['rho_sub_c_rel_l2_error']:.6e}")
    print(f"relative L2 error Ttx_sub_c  = {summary['Ttx_sub_c_rel_l2_error']:.6e}")
    print(f"relative L2 error q_var_sub  = {summary['q_var_sub_rel_l2_error']:.6e}")

    print(f"\nSaved validation outputs to: {outdir.resolve()}")
    print_key_files(cfg, list(derived.keys()))
    return summary


def load_history(outdir: Path, E_nqs: float) -> np.ndarray:
    hist_path = outdir / "training_history.csv"
    if hist_path.exists():
        hist = np.loadtxt(hist_path, delimiter=",", skiprows=1)
        if hist.ndim == 1:
            hist = hist.reshape(1, -1)
        return hist
    return np.asarray([[0.0, E_nqs]])


def generate_plots_from_saved(cfg: Config, matrices_path: str | Path | None = None) -> None:
    """Plot-only run."""
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    matrices_path = Path(matrices_path) if matrices_path else outdir / "validation_matrices_3p1.npz"
    if not matrices_path.exists():
        raise FileNotFoundError(
            f"No saved matrices found at {matrices_path}. Run train_validate.py once with --save-matrices first."
        )

    print("\nPlots-only mode")
    print(f"loading matrices from: {matrices_path}")

    prob = make_problem(cfg)
    data = np.load(matrices_path)
    M_nqs = data["M_nqs"]
    N_nqs = data["N_nqs"]

    if "M_analytic" in data and "N_analytic" in data:
        M_an = data["M_analytic"]
        N_an = data["N_analytic"]
        freqs = data["freqs"] if "freqs" in data else np.array([])
        exact = analytic_gaussian_width(prob.K, prob.A) if freqs.size == 0 else None
    else:
        exact = analytic_gaussian_width(prob.K, prob.A)
        M_an, N_an = exact.M, exact.N
        freqs = exact.freqs

    E_an = gaussian_energy_np(M_an, N_an, prob.K, prob.A)
    E_nqs = gaussian_energy_np(M_nqs, N_nqs, prob.K, prob.A)
    hist = load_history(outdir, E_nqs)
    derived = build_derived_observables(cfg, prob, M_nqs, N_nqs, M_an, N_an)

    print(f"recomputed NQS E                = {E_nqs:.12f}")
    print(f"analytic E                      = {E_an:.12f}")
    print(f"energy error E_NQS - E_analytic = {E_nqs - E_an:.12e}")
    print("Generating plots...")

    generate_all_plots(outdir, cfg, prob, derived, hist, E_an)

    save_config(cfg, outdir / "last_plot_config_3p1.json")

    print(f"\nSaved regenerated plots to: {outdir.resolve()}")
    print_key_files(cfg, list(derived.keys()))
