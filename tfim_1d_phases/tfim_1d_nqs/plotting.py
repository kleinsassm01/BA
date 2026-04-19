from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science"])
except Exception:
    pass

from .config import CriticalZoomConfig
from .exact_solver import ExactIsingSolver
from .models import ExperimentDataset, TrainingResult


def _cmap_for_N(N_values: list[int]):
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

    def plot_phase_diagram(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        order = np.argsort([r.J for r in results])
        results_sorted = [results[i] for i in order]
        J_vals = np.array([r.J for r in results_sorted])
        m2_vals = np.array([r.m2_final for r in results_sorted])
        n2_vals = np.array([r.n2_final for r in results_sorted])
        e_nqs = np.array([r.e_final for r in results_sorted])
        e_exact_f = np.array([r.e_exact_finite for r in results_sorted])
        h_val = results_sorted[0].h

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
                     label=r"$\langle m^2 \rangle$ (ferro, $J<0$)", zorder=3)
        ax_main.plot(J_vals, n2_vals, "o", ms=4, color=self.colors["n2"],
                     label=r"$\langle n^2 \rangle$ (antiferro, $J>0$)", zorder=3)
        for J_c in (-h_val, h_val):
            ax_main.axvline(J_c, color=self.colors["guide"], lw=0.8, linestyle=":")
        ax_main.set_xlabel(r"$J$")
        ax_main.set_ylabel(r"$\langle m^2 \rangle,\, \langle n^2 \rangle$")
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
        # Label panels by PHYSICAL PHASE (NetKet sign convention):
        #   J << 0 -> ferro, J ~ 0 -> para, J >> 0 -> antiferro
        idx_ferro = int(np.argmin(J_vals))
        idx_para = int(np.argmin(np.abs(J_vals)))
        idx_af = int(np.argmax(J_vals))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
        panels = [
            (axes[0], idx_ferro, f"ferro ($J={J_vals[idx_ferro]:.1f}$)"),
            (axes[1], idx_para, f"para ($J={J_vals[idx_para]:.1f}$)"),
            (axes[2], idx_af, f"antiferro ($J={J_vals[idx_af]:.1f}$)"),
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
        axes[0].set_title(r"$\langle m^2 \rangle$ (ferro, $J<0$)")
        axes[1].set_title(r"$\langle n^2 \rangle$ (antiferro, $J>0$)")
        
        for ax in axes:
            ax.set_xlabel("step")
            ax.set_ylim(-0.02, 1.03)
            ax.grid(True, alpha=0.18)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_multi_N_overlay(self, dataset: ExperimentDataset, save_path: Path) -> None:
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.35, wspace=0.30)
        ax_main = fig.add_subplot(gs[0, :])
        ax_e = fig.add_subplot(gs[1, 0])
        ax_err = fig.add_subplot(gs[1, 1])

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            m2 = np.array([r.m2_final for r in rs])
            n2 = np.array([r.n2_final for r in rs])
            c = color_by_N[N]
            ax_main.plot(J, m2, "-o", ms=3, color=c, label=rf"$N={N}$")
            ax_main.plot(J, n2, "--s", ms=3, color=c, alpha=0.8)

        for J_c in (-h_val, h_val):
            ax_main.axvline(J_c, color=self.colors["guide"], lw=0.8, linestyle=":")

        ax_main.set_xlabel(r"$J$")
        ax_main.set_ylabel(
            r"$\langle m^2 \rangle$ (solid, ferro), "
            r"$\langle n^2 \rangle$ (dashed, antiferro)"
        )
        ax_main.set_title(
            rf"Phase diagram across system sizes ($h={h_val:.1f}$)  --  "
            rf"dotted: $|J|=h$ quantum critical points"
        )
        ax_main.set_ylim(-0.02, 1.03)
        ax_main.grid(True, alpha=0.18)
        ax_main.legend(frameon=False, loc="upper center", ncol=min(len(N_values), 5))

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
        try:
            from scipy.signal import savgol_filter
            from scipy.interpolate import interp1d
            have_scipy = True
        except Exception:
            have_scipy = False

        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        J_ferro_center = -h_val   # ferro transition at J = -h
        J_af_center    = +h_val   # antiferro transition at J = +h
        hw = zoom_cfg.zoom_halfwidth
        Jf_lo, Jf_hi = J_ferro_center - hw, J_ferro_center + hw
        Ja_lo, Ja_hi = J_af_center    - hw, J_af_center    + hw

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
        ax_ferro, ax_af, ax_chi = axes

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            m2 = np.array([r.m2_final for r in rs])
            n2 = np.array([r.n2_final for r in rs])
            e = np.array([r.e_final for r in rs])
            c = color_by_N[N]

            mask_f = (J >= Jf_lo) & (J <= Jf_hi)
            mask_a = (J >= Ja_lo) & (J <= Ja_hi)
            if mask_f.any():
                ax_ferro.plot(J[mask_f], m2[mask_f], "-o", ms=4, color=c, label=rf"$N={N}$")
            if mask_a.any():
                ax_af.plot(J[mask_a], n2[mask_a], "-o", ms=4, color=c, label=rf"$N={N}$")

            mask_full = np.argsort(J)
            J_sorted = J[mask_full]
            e_sorted = e[mask_full]
            if have_scipy and len(J_sorted) >= 5:
                J_uni = np.linspace(J_sorted.min(), J_sorted.max(), 4 * len(J_sorted))
                try:
                    e_uni = interp1d(J_sorted, e_sorted, kind="cubic")(J_uni)
                except Exception:
                    e_uni = np.interp(J_uni, J_sorted, e_sorted)
                win = 11 if len(J_uni) >= 11 else (len(J_uni) // 2) * 2 + 1
                win = max(win, 5)
                delta = J_uni[1] - J_uni[0]
                d2 = savgol_filter(e_uni, window_length=win, polyorder=3,
                                   deriv=2, delta=delta)
                m_chi = (J_uni >= Ja_lo - 0.1) & (J_uni <= Ja_hi + 0.1)
                if m_chi.any():
                    ax_chi.plot(J_uni[m_chi], -d2[m_chi], "-", lw=1.4,
                                color=c, label=rf"$N={N}$")
            elif len(J_sorted) >= 3:
                d1 = np.gradient(e_sorted, J_sorted)
                d2 = np.gradient(d1, J_sorted)
                m_chi = (J_sorted >= Ja_lo - 0.1) & (J_sorted <= Ja_hi + 0.1)
                if m_chi.any():
                    ax_chi.plot(J_sorted[m_chi], -d2[m_chi], "-o", ms=3,
                                color=c, label=rf"$N={N}$")

        ax_ferro.axvline(J_ferro_center, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_ferro.set_xlabel(r"$J$")
        ax_ferro.set_ylabel(r"$\langle m^2 \rangle$")
        ax_ferro.set_title(rf"ferro critical region ($J\approx -h = {J_ferro_center:.1f}$)")
        ax_ferro.set_ylim(-0.02, 1.03)
        ax_ferro.grid(True, alpha=0.18)
        ax_ferro.legend(frameon=False, loc="best", fontsize=9)

        ax_af.axvline(J_af_center, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_af.set_xlabel(r"$J$")
        ax_af.set_ylabel(r"$\langle n^2 \rangle$")
        ax_af.set_title(rf"antiferro critical region ($J\approx h = {J_af_center:.1f}$)")
        ax_af.set_ylim(-0.02, 1.03)
        ax_af.grid(True, alpha=0.18)
        ax_af.legend(frameon=False, loc="best", fontsize=9)

        ax_chi.axvline(J_af_center, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_chi.set_xlabel(r"$J$")
        ax_chi.set_ylabel(r"$-d^2 E_0/dJ^2 \,/\, N$")
        ax_chi.set_title(r"energy curvature (SavGol-smoothed)")
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
        J_range: tuple[float, float] = (-2.0, 2.0),
        U4_ylim: tuple[float, float] = (-0.5, 1.0),
    ) -> None:
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]
        J_lo, J_hi = J_range
        U4_lo, U4_hi = U4_ylim

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        ax_full, ax_zoom = axes

        total_clipped = 0
        total_hidden = 0

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            U4 = np.array([r.binder_U4 for r in rs])

            valid = np.isfinite(U4) & (J >= J_lo) & (J <= J_hi)

            span = U4_hi - U4_lo
            hide_mask = (U4 < U4_lo - span) | (U4 > U4_hi + span)
            total_hidden += int((hide_mask & valid).sum())
            valid = valid & ~hide_mask

            U4_clipped = np.clip(U4, U4_lo, U4_hi)
            total_clipped += int((valid & ((U4 < U4_lo) | (U4 > U4_hi))).sum())

            c = color_by_N[N]
            ax_full.plot(
                J[valid], U4_clipped[valid],
                "-o", ms=3, color=c, label=rf"$N={N}$",
            )

            Jf_lo = -h_val - zoom_cfg.zoom_halfwidth
            Jf_hi = -h_val + zoom_cfg.zoom_halfwidth
            mask_zoom = valid & (J >= Jf_lo) & (J <= Jf_hi)
            ax_zoom.plot(
                J[mask_zoom], U4_clipped[mask_zoom],
                "-o", ms=4, color=c, label=rf"$N={N}$",
            )

        if total_clipped or total_hidden:
            note = f"clipped: {total_clipped}  hidden: {total_hidden}"
            ax_full.text(
                0.02, 0.02, note,
                transform=ax_full.transAxes,
                fontsize=8, color=self.colors["guide"],
                verticalalignment="bottom",
            )

        for ax in axes:
            ax.axhline(2.0 / 3.0, color=self.colors["guide"], lw=0.7,
                       linestyle="--", alpha=0.6)
            ax.axhline(0.0, color=self.colors["guide"], lw=0.7,
                       linestyle="--", alpha=0.6)
            ax.axvline(h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
            ax.axvline(-h_val, color=self.colors["guide"], lw=0.9, linestyle=":")
            ax.set_xlabel(r"$J$")
            ax.set_ylabel(
                r"$U_4 = 1 - \langle m^4\rangle/(3\langle m^2\rangle^2)$"
            )
            ax.set_ylim(U4_lo, U4_hi)
            ax.grid(True, alpha=0.18)
            ax.legend(frameon=False, loc="best", fontsize=9)

        ax_full.set_xlim(J_lo, J_hi)
        ax_full.set_title(rf"Binder cumulant -- restricted to $J\in[{J_lo},{J_hi}]$")
        ax_zoom.set_title(rf"Binder -- ferro crossing near $J={-h_val:.1f}$")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_tau_corr_vs_step(self, dataset: ExperimentDataset, save_path: Path) -> None:
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
        ax_acf.set_title(rf"Energy autocorrelation at $J\approx{h_val:.1f}$ (antiferro critical)")
        ax_acf.grid(True, alpha=0.18)
        ax_acf.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
