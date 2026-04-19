from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    # scienceplots is optional -- if it's not installed we fall back to the
    # default matplotlib style so the extension still works on a bare env.
    import scienceplots  # noqa: F401
    plt.style.use(["science"])
except Exception:
    pass

from .config import CriticalZoomConfig
from .exact_solver import ExactIsingSolver
from .models import ExperimentDataset, TrainingResult


def _cmap_for_N(N_values: list[int]):
    """Deterministic viridis sampling: one color per N, in sorted order."""
    N_sorted = sorted(N_values)
    cmap = plt.cm.viridis
    denom = max(len(N_sorted) - 1, 1)
    return {N: cmap(i / denom * 0.9) for i, N in enumerate(N_sorted)}


class ResultPlotter:
    def __init__(self, exact_solver: ExactIsingSolver | None = None) -> None:
        self.exact_solver = exact_solver or ExactIsingSolver()

        self.colors = {
            "m2": "#0072B2",
            "n2": "#009E73",
            "nqs": "#960008",
            "finite": "#1f1f1f",
            "thermo": "#7A7A7A",
            "guide": "#B8B8B8",
        }

        plt.rcParams.update({
            "figure.dpi": 150,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "axes.grid": False,
        })

    # ======================================================================
    # ORIGINAL PLOTS (unchanged behaviour)
    # ======================================================================

    def plot_phase_diagram(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        J_vals = np.array([r.J for r in results])
        order = np.argsort(J_vals)
        J_vals = J_vals[order]
        m2_vals = np.array([results[i].m2_final for i in order])
        n2_vals = np.array([results[i].n2_final for i in order])
        e_nqs = np.array([results[i].e_final for i in order])
        e_exact_f = np.array([results[i].e_exact_finite for i in order])
        h_val = results[0].h

        J_dense = np.linspace(J_vals.min(), J_vals.max(), 600)
        e_exact_dense_f = np.array(
            [self.exact_solver.energy_finite(N, J, h_val) for J in J_dense]
        )
        e_exact_dense_t = np.array(
            [self.exact_solver.energy_thermodynamic(J, h_val) for J in J_dense]
        )

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.35, wspace=0.30)

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(J_vals, m2_vals, "--", color=self.colors["guide"], lw=1.0, zorder=1)
        ax_main.plot(J_vals, n2_vals, "--", color=self.colors["guide"], lw=1.0, zorder=1)
        ax_main.plot(J_vals, m2_vals, "o", ms=4, color=self.colors["m2"],
                     label=r"$\langle m^2 \rangle$", zorder=3)
        ax_main.plot(J_vals, n2_vals, "o", ms=4, color=self.colors["n2"],
                     label=r"$\langle n^2 \rangle$", zorder=3)
        ax_main.set_xlabel(r"$J$")
        ax_main.set_ylabel(r"$\langle m^2 \rangle / \langle n^2 \rangle$")
        ax_main.set_title(rf"1D TFIM phase diagram ($h={h_val:.1f},\, N={N}$)")
        ax_main.set_xlim(J_vals.min() - 0.15, J_vals.max() + 0.15)
        ax_main.set_ylim(-0.02, 1.03)
        ax_main.legend(frameon=False, loc="best")
        ax_main.grid(True, alpha=0.18)

        ax_e = fig.add_subplot(gs[1, 0])
        ax_e.plot(J_dense, e_exact_dense_f, "-", color=self.colors["finite"], lw=1.5,
                  label=rf"exact finite $N={N}$")
        ax_e.plot(J_dense, e_exact_dense_t, "--", color=self.colors["thermo"], lw=1,
                  label=r"exact $N\to\infty$")
        ax_e.plot(J_vals, e_nqs, "o", ms=4, color=self.colors["nqs"], label="NQS", zorder=4)
        ax_e.set_xlabel(r"$J$")
        ax_e.set_ylabel(r"$E_0/N$")
        ax_e.set_title("ground-state energy")
        ax_e.legend(frameon=False, loc="best")
        ax_e.grid(True, alpha=0.18)

        ax_err = fig.add_subplot(gs[1, 1])
        rel_error = np.abs(e_nqs - e_exact_f) / np.maximum(np.abs(e_exact_f), 1e-14) * 100
        ax_err.plot(J_vals, rel_error, "--", color=self.colors["guide"], lw=1.0, zorder=1)
        ax_err.plot(J_vals, rel_error, "o", ms=4, color=self.colors["m2"], zorder=3)
        ax_err.set_xlabel(r"$J$")
        ax_err.set_ylabel(r"relative error (\%)")
        ax_err.set_title("accuracy")
        ax_err.set_yscale("log")
        ax_err.grid(True, which="both", alpha=0.18)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_training_convergence(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        J_vals = np.array([r.J for r in results])
        idx_af = int(np.argmin(J_vals))
        idx_para = int(np.argmin(np.abs(J_vals)))
        idx_ferro = int(np.argmax(J_vals))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        panels = [
            (axes[0], idx_af, f"antiferro ($J={J_vals[idx_af]:.1f}$)"),
            (axes[1], idx_para, f"para ($J={J_vals[idx_para]:.1f}$)"),
            (axes[2], idx_ferro, f"ferro ($J={J_vals[idx_ferro]:.1f}$)"),
        ]
        for ax, idx, title in panels:
            r = results[idx]
            ax.plot(r.history.iters, r.history.energy, "o", ms=2.0,
                    color=self.colors["nqs"], alpha=0.85, label="NQS", zorder=3)
            ax.axhline(r.e_exact_finite, color=self.colors["finite"], lw=1.0,
                       linestyle="-", label=rf"exact finite $N={N}$", zorder=1)
            ax.axhline(r.e_exact_thermo, color=self.colors["thermo"], lw=1.0,
                       linestyle="--", label=r"exact $N\to\infty$", zorder=1)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.set_ylabel(r"$E_0/N$")
            ax.grid(True, alpha=0.18)
            ax.legend(frameon=False, loc="best")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_training_histories(self, results: list[TrainingResult], save_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        cmap = plt.cm.coolwarm
        J_vals = np.array([r.J for r in results])
        norm = plt.Normalize(J_vals.min(), J_vals.max())
        for r in results:
            color = cmap(norm(r.J))
            axes[0].plot(r.history.iters, r.history.m2, "o", ms=1.7, color=color, alpha=0.55)
            axes[1].plot(r.history.iters, r.history.n2, "o", ms=1.7, color=color, alpha=0.55)
        axes[0].set_title(r"$\langle m^2 \rangle$")
        axes[1].set_title(r"$\langle n^2 \rangle$")
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.18)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    # ======================================================================
    # NEW PLOTS: multi-N overlay, critical zoom, Binder, autocorrelation
    # ======================================================================

    def plot_multi_N_overlay(self, dataset: ExperimentDataset, save_path: Path) -> None:
        """
        Overlay the full J-sweep for every N in the dataset.

        Three panels (top-wide for order parameters, bottom-left for energy
        per site, bottom-right for relative error vs. the finite-N exact
        solution):

          * <m^2>(J) and <n^2>(J): broad plateaus at J > h and J < -h
            steepen as N grows; the crossover narrows.
          * E_0/N(J): finite-N curves converge from above to the exact
            N -> infinity curve (thermodynamic limit). This is the *direct*
            picture of approaching the thermodynamic limit.
          * relative error vs N: spread across J shows where NQS struggles
            most (typically near criticality).
        """
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.35, wspace=0.30)
        ax_main = fig.add_subplot(gs[0, :])
        ax_e = fig.add_subplot(gs[1, 0])
        ax_err = fig.add_subplot(gs[1, 1])

        # m2/n2 overlay
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            m2 = np.array([r.m2_final for r in rs])
            n2 = np.array([r.n2_final for r in rs])
            c = color_by_N[N]
            ax_main.plot(J, m2, "-o", ms=3, color=c, label=rf"$N={N}$")
            ax_main.plot(J, n2, "--s", ms=3, color=c, alpha=0.8)

        # Vertical guides at |J| = h (second-order critical points).
        for J_c in (-h_val, h_val):
            ax_main.axvline(J_c, color=self.colors["guide"], lw=0.8, linestyle=":")

        ax_main.set_xlabel(r"$J$")
        ax_main.set_ylabel(r"$\langle m^2 \rangle$ (solid), $\langle n^2 \rangle$ (dashed)")
        ax_main.set_title(
            rf"Phase diagram across system sizes ($h={h_val:.1f}$) -- "
            rf"dashed vertical lines mark $|J|=h$ (quantum critical points)"
        )
        ax_main.set_ylim(-0.02, 1.03)
        ax_main.grid(True, alpha=0.18)
        ax_main.legend(frameon=False, loc="upper center", ncol=min(len(N_values), 5))

        # Energy-per-site overlay with exact N -> infinity curve.
        J_dense = np.linspace(
            min(r.J for r in dataset.results),
            max(r.J for r in dataset.results),
            600,
        )
        e_thermo = np.array([self.exact_solver.energy_thermodynamic(J, h_val) for J in J_dense])
        ax_e.plot(J_dense, e_thermo, "-", color="k", lw=1.2, label=r"exact $N\to\infty$")
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            e_nqs = np.array([r.e_final for r in rs])
            ax_e.plot(J, e_nqs, "o", ms=3, color=color_by_N[N], label=rf"NQS $N={N}$")
        ax_e.set_xlabel(r"$J$")
        ax_e.set_ylabel(r"$E_0/N$")
        ax_e.set_title("ground-state energy density")
        ax_e.grid(True, alpha=0.18)
        ax_e.legend(frameon=False, loc="best", fontsize=9)

        # Relative error vs finite-N exact, overlaid.
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            err = np.array([r.rel_error_pct for r in rs])
            ax_err.plot(J, np.clip(err, 1e-4, None), "-o", ms=3,
                        color=color_by_N[N], label=rf"$N={N}$")
        ax_err.set_xlabel(r"$J$")
        ax_err.set_ylabel(r"relative error (\%)")
        ax_err.set_title(r"NQS accuracy vs finite-$N$ exact")
        ax_err.set_yscale("log")
        ax_err.grid(True, which="both", alpha=0.18)
        ax_err.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_critical_zoom(
        self,
        dataset: ExperimentDataset,
        zoom_cfg: CriticalZoomConfig,
        save_path: Path,
    ) -> None:
        """
        Zoom in on the critical regions at J = +- h. Two panels -- left is
        around J = -h (antiferromagnetic QPT, diagnosed by <n^2>), right is
        around J = +h (ferromagnetic QPT, diagnosed by <m^2>).

        Each N is a separate curve. As N grows, the crossover
            * sharpens (the slope at the transition steepens),
            * converges towards a step function in the thermodynamic limit.
        This is the finite-size signature of a *second-order* QPT with
        diverging correlation length xi ~ |J - J_c|^{-nu}.

        A third panel shows -d^2 E_0 / dJ^2 (numerical, from NQS energies),
        which acts as an order-parameter susceptibility: it develops a
        sharper and sharper peak at J_c as N grows.
        """
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]
        Jf_lo, Jf_hi = zoom_cfg.J_center_ferro - zoom_cfg.zoom_halfwidth, zoom_cfg.J_center_ferro + zoom_cfg.zoom_halfwidth
        Ja_lo, Ja_hi = zoom_cfg.J_center_antiferro - zoom_cfg.zoom_halfwidth, zoom_cfg.J_center_antiferro + zoom_cfg.zoom_halfwidth

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
        ax_af, ax_f, ax_chi = axes

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            m2 = np.array([r.m2_final for r in rs])
            n2 = np.array([r.n2_final for r in rs])
            e = np.array([r.e_final for r in rs])
            c = color_by_N[N]

            mask_af = (J >= Ja_lo) & (J <= Ja_hi)
            mask_f = (J >= Jf_lo) & (J <= Jf_hi)

            ax_af.plot(J[mask_af], n2[mask_af], "-o", ms=4, color=c, label=rf"$N={N}$")
            ax_f.plot(J[mask_f], m2[mask_f], "-o", ms=4, color=c, label=rf"$N={N}$")

            # Second derivative of E_0/N wrt J via finite differences.
            if len(J) >= 3:
                # Use only the interior points.
                d2 = np.gradient(np.gradient(e, J), J)
                # Focus on the region around the ferro critical point.
                mask = (J >= Jf_lo - 0.1) & (J <= Jf_hi + 0.1)
                if mask.any():
                    ax_chi.plot(J[mask], -d2[mask], "-o", ms=3, color=c, label=rf"$N={N}$")

        ax_af.axvline(-h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_af.set_xlabel(r"$J$")
        ax_af.set_ylabel(r"$\langle n^2 \rangle$")
        ax_af.set_title(rf"antiferro critical region ($J\approx -h = -{h_val:.1f}$)")
        ax_af.set_ylim(-0.02, 1.03)
        ax_af.grid(True, alpha=0.18)
        ax_af.legend(frameon=False, loc="best", fontsize=9)

        ax_f.axvline(h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_f.set_xlabel(r"$J$")
        ax_f.set_ylabel(r"$\langle m^2 \rangle$")
        ax_f.set_title(rf"ferro critical region ($J\approx h = {h_val:.1f}$)")
        ax_f.set_ylim(-0.02, 1.03)
        ax_f.grid(True, alpha=0.18)
        ax_f.legend(frameon=False, loc="best", fontsize=9)

        ax_chi.axvline(h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_chi.set_xlabel(r"$J$")
        ax_chi.set_ylabel(r"$-d^2 E_0/dJ^2 \,/\, N$")
        ax_chi.set_title(r"energy curvature (peak $\to\infty$ for 2nd order)")
        ax_chi.grid(True, alpha=0.18)
        ax_chi.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_binder_cumulant(
        self,
        dataset: ExperimentDataset,
        zoom_cfg: CriticalZoomConfig,
        save_path: Path,
    ) -> None:
        """
        Binder cumulant U_4 = 1 - <m^4>/(3 <m^2>^2) vs J for every N.

        For a *second-order* phase transition, curves at different N all pass
        through a common value at J = J_c (crossing point). This crossing is
        a finite-size-scaling estimator of the critical coupling, and its
        existence is the textbook signature of 2nd order.

        For a 1st-order transition the curves would instead develop a
        size-dependent dip and no universal crossing.
        """
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        ax_full, ax_zoom = axes

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            U4 = np.array([r.binder_U4 for r in rs])
            valid = np.isfinite(U4)
            c = color_by_N[N]
            ax_full.plot(J[valid], U4[valid], "-o", ms=3, color=c, label=rf"$N={N}$")

            # zoom on the ferro transition window
            Jf_lo = zoom_cfg.J_center_ferro - zoom_cfg.zoom_halfwidth
            Jf_hi = zoom_cfg.J_center_ferro + zoom_cfg.zoom_halfwidth
            mask = valid & (J >= Jf_lo) & (J <= Jf_hi)
            ax_zoom.plot(J[mask], U4[mask], "-o", ms=4, color=c, label=rf"$N={N}$")

        for ax in axes:
            ax.axvline(h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
            ax.axvline(-h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
            ax.set_xlabel(r"$J$")
            ax.set_ylabel(r"$U_4 = 1 - \langle m^4\rangle/(3\langle m^2\rangle^2)$")
            ax.grid(True, alpha=0.18)
            ax.legend(frameon=False, loc="best", fontsize=9)
        ax_full.set_title("Binder cumulant -- full range")
        ax_zoom.set_title(rf"Binder cumulant -- ferro crossing near $J={h_val:.1f}$")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_tau_corr_vs_step(self, dataset: ExperimentDataset, save_path: Path) -> None:
        """
        Per-step integrated autocorrelation time tau_corr (as reported by
        NetKet's Stats object during training), plotted vs the training
        step. One subplot per N; curves colored by J.

        What to look for:
            * tau_corr should settle to a roughly constant level once the
              variational state has converged.
            * Points near the critical coupling |J| = h typically show the
              largest tau_corr -- the hallmark of critical slowing down.
        """
        N_values = dataset.N_values()
        n_panels = len(N_values)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(4.6 * n_panels, 4.6),
            constrained_layout=True, squeeze=False,
        )
        axes = axes[0]

        for ax, N in zip(axes, N_values):
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J_vals = np.array([r.J for r in rs])
            if len(J_vals) == 0:
                continue
            norm = plt.Normalize(J_vals.min(), J_vals.max())
            cmap = plt.cm.coolwarm
            for r in rs:
                color = cmap(norm(r.J))
                tau = np.array(r.history.tau_corr, dtype=float)
                it = np.array(r.history.iters, dtype=float)
                mask = np.isfinite(tau)
                if not mask.any():
                    continue
                ax.plot(it[mask], tau[mask], "-", lw=1.0, color=color, alpha=0.7)
            ax.set_xlabel("training step")
            ax.set_ylabel(r"$\tau_{\mathrm{corr}}$ (MC steps)")
            ax.set_title(rf"$N={N}$")
            ax.grid(True, alpha=0.18)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, pad=0.02)
            cb.set_label(r"$J$")

        fig.suptitle(
            r"Per-step autocorrelation time $\tau_{\mathrm{corr}}$ during training",
            fontsize=14,
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_tau_int_vs_J(self, dataset: ExperimentDataset, save_path: Path) -> None:
        """
        Two panels:

          (a) Integrated autocorrelation time tau_int(J) from the dedicated
              post-training MC chain (Sokal windowing), one curve per N.
              Critical slowing down -> tau_int peaks at |J| = h.

          (b) The autocorrelation function rho(t) = C(t)/C(0) at the
              critical point J = +h, for each N. Slower decay at larger N is
              the visual fingerprint of critical slowing down.
        """
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        ax_tau, ax_acf = axes

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            tau_int = np.array([
                r.autocorr.tau_int if r.autocorr is not None else np.nan for r in rs
            ])
            valid = np.isfinite(tau_int)
            if valid.any():
                ax_tau.plot(J[valid], tau_int[valid], "-o", ms=4,
                            color=color_by_N[N], label=rf"$N={N}$")

            # Pick the result closest to +h for the ACF curve.
            idx_crit = int(np.argmin(np.abs(J - h_val)))
            r_crit = rs[idx_crit]
            if r_crit.autocorr is not None:
                lags = np.array(r_crit.autocorr.lags)
                rho = np.array(r_crit.autocorr.acf)
                ax_acf.plot(lags, rho, "-", lw=1.4, color=color_by_N[N],
                            label=rf"$N={N}$  ($J={r_crit.J:.2f}$)")

        for J_c in (-h_val, h_val):
            ax_tau.axvline(J_c, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_tau.set_xlabel(r"$J$")
        ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}$ (MC steps)")
        ax_tau.set_title(r"Integrated autocorrelation time (post-training, Sokal window)")
        ax_tau.set_yscale("log")
        ax_tau.grid(True, which="both", alpha=0.18)
        ax_tau.legend(frameon=False, loc="best", fontsize=9)

        ax_acf.axhline(0.0, color="k", lw=0.6)
        ax_acf.set_xlabel(r"lag $t$ (MC steps)")
        ax_acf.set_ylabel(r"$\rho(t) = C(t)/C(0)$")
        ax_acf.set_title(rf"Energy autocorrelation at $J\approx{h_val:.1f}$ (critical)")
        ax_acf.grid(True, alpha=0.18)
        ax_acf.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
