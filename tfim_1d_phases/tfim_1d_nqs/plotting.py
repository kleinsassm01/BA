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

    # =========================================================================
    # NEW PLOTS
    # =========================================================================

    def plot_energy_variance(self, dataset: ExperimentDataset, save_path: Path) -> None:
        """Variational energy variance per site.

        Left panel: variance-per-site during training for every chain
        (colour-coded by J, one sub-panel per system size).
        Right panel: late-training variance per site as a function of J,
        for every N. The variance is strictly zero only at an eigenstate,
        so this is a fidelity proxy that peaks at the critical points,
        where the variational ansatz has the hardest time.
        """
        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
        ax_train, ax_final = axes

        # Left: envelope (mean over J) of var/N during training, per N
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            iters_ref = None
            stack = []
            for r in rs:
                it = np.array(r.history.iters, dtype=float)
                ev = np.array(r.history.e_var, dtype=float) / N
                mask = np.isfinite(ev)
                if not mask.any():
                    continue
                if iters_ref is None:
                    iters_ref = it[mask]
                stack.append(np.interp(iters_ref, it[mask], ev[mask]))
            if not stack:
                continue
            arr = np.vstack(stack)
            med = np.nanmedian(arr, axis=0)
            q25 = np.nanquantile(arr, 0.25, axis=0)
            q75 = np.nanquantile(arr, 0.75, axis=0)
            c = color_by_N[N]
            ax_train.plot(iters_ref, med, "-", lw=1.5, color=c, label=rf"$N={N}$")
            ax_train.fill_between(iters_ref, q25, q75, color=c, alpha=0.18, linewidth=0)

        ax_train.set_xlabel("training step")
        ax_train.set_ylabel(r"$\sigma_E^2 / N$ (median $\pm$ IQR over $J$)")
        ax_train.set_title(r"Energy variance per site during training")
        ax_train.set_yscale("log")
        ax_train.grid(True, which="both", alpha=0.18)
        ax_train.legend(frameon=False, loc="best", fontsize=9)

        # Right: late-training variance (last 20% of iterations) as a function of J
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J_arr, var_arr = [], []
            for r in rs:
                ev = np.array(r.history.e_var, dtype=float)
                mask = np.isfinite(ev)
                if not mask.any():
                    continue
                tail = ev[mask][int(0.8 * mask.sum()):]
                if tail.size == 0:
                    continue
                J_arr.append(r.J)
                var_arr.append(float(np.nanmean(tail)) / N)
            if not J_arr:
                continue
            order = np.argsort(J_arr)
            J_arr = np.array(J_arr)[order]
            var_arr = np.array(var_arr)[order]
            ax_final.plot(J_arr, np.clip(var_arr, 1e-8, None), "-o", ms=4,
                          color=color_by_N[N], label=rf"$N={N}$")

        for J_c in (-h_val, h_val):
            ax_final.axvline(J_c, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_final.set_xlabel(r"$J$")
        ax_final.set_ylabel(r"$\sigma_E^2 / N$ (late-training mean)")
        ax_final.set_title(r"Late-training variance vs $J$ (zero for eigenstate)")
        ax_final.set_yscale("log")
        ax_final.grid(True, which="both", alpha=0.18)
        ax_final.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_binder_crossings(
        self,
        dataset: ExperimentDataset,
        zoom_cfg: CriticalZoomConfig,
        save_path: Path,
    ) -> None:
        """Binder cumulant zoomed around the ferro transition with explicit
        pairwise-crossing estimates of J_c.

        Left: Binder curves interpolated with a cubic spline on a fine J grid.
        Estimated J_c is read off from the consecutive-N crossings
        (N_1, N_2) -> J_c(N_1, N_2).

        Right: J_c estimates as a function of 1/N_eff, with a linear
        extrapolation to 1/N_eff -> 0 (i.e. N -> infinity).
        """
        from scipy.interpolate import interp1d

        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        J_center = -h_val
        hw = zoom_cfg.zoom_halfwidth
        J_lo, J_hi = J_center - hw, J_center + hw

        # Build per-N interpolators on the zoom window
        splines: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            U4 = np.array([r.binder_U4 for r in rs])
            mask = np.isfinite(U4) & (J >= J_lo - 0.05) & (J <= J_hi + 0.05)
            if mask.sum() < 4:
                continue
            J_m, U_m = J[mask], U4[mask]
            order = np.argsort(J_m)
            J_m, U_m = J_m[order], U_m[order]
            J_grid = np.linspace(J_m.min(), J_m.max(), 400)
            try:
                U_grid = interp1d(J_m, U_m, kind="cubic")(J_grid)
            except Exception:
                U_grid = np.interp(J_grid, J_m, U_m)
            splines[N] = (J_grid, U_grid)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        ax_zoom, ax_extrap = axes

        # Plot interpolated Binder curves
        for N, (Jg, Ug) in splines.items():
            ax_zoom.plot(Jg, Ug, "-", lw=1.4, color=color_by_N[N], label=rf"$N={N}$")
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            U4 = np.array([r.binder_U4 for r in rs])
            mask = np.isfinite(U4) & (J >= J_lo) & (J <= J_hi)
            ax_zoom.plot(J[mask], U4[mask], "o", ms=3, color=color_by_N[N], alpha=0.5)

        # Find crossings between consecutive-N pairs
        N_sorted = sorted(splines.keys())
        crossings_list: list[tuple[int, int, float]] = []
        J_common = None
        for N1, N2 in zip(N_sorted[:-1], N_sorted[1:]):
            J1, U1 = splines[N1]
            J2, U2 = splines[N2]
            J_common = np.linspace(
                max(J1.min(), J2.min()), min(J1.max(), J2.max()), 600,
            )
            f1 = np.interp(J_common, J1, U1)
            f2 = np.interp(J_common, J2, U2)
            diff = f1 - f2
            sign = np.sign(diff)
            candidates = []
            for i in range(len(diff) - 1):
                if sign[i] == 0 or sign[i] * sign[i + 1] < 0:
                    a, b = diff[i], diff[i + 1]
                    if abs(b - a) < 1e-12:
                        xc = J_common[i]
                    else:
                        xc = J_common[i] - a * (J_common[i + 1] - J_common[i]) / (b - a)
                    candidates.append(xc)
            if not candidates:
                continue
            # Keep only physically reasonable candidates (close to the
            # expected critical point) and pick the closest one.
            crossing_window = 0.15
            candidates = [x for x in candidates
                          if abs(x - J_center) < crossing_window]
            if not candidates:
                continue
            xc_best = min(candidates, key=lambda x: abs(x - J_center))
            crossings_list.append((N1, N2, xc_best))
            y_c = float(np.interp(xc_best, J1, U1))
            ax_zoom.plot([xc_best], [y_c], "k*", ms=8, zorder=6)

        ax_zoom.axvline(J_center, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_zoom.axhline(2.0 / 3.0, color=self.colors["guide"], lw=0.7,
                        linestyle="--", alpha=0.6)
        ax_zoom.set_xlabel(r"$J$")
        ax_zoom.set_ylabel(
            r"$U_4 = 1 - \langle m^4\rangle/(3\langle m^2\rangle^2)$"
        )
        ax_zoom.set_title(rf"Binder cumulant near $J_c^{{\mathrm{{ferro}}}} = {J_center:.1f}$")
        ax_zoom.set_xlim(J_lo, J_hi)
        ax_zoom.set_ylim(-0.1, 0.75)
        ax_zoom.grid(True, alpha=0.18)
        ax_zoom.legend(frameon=False, loc="best", fontsize=9)

        # Right panel: extrapolation in 1/N_eff = 2 / (1/N1 + 1/N2)
        if crossings_list:
            xs, ys, labels = [], [], []
            for (N1, N2, xc) in crossings_list:
                N_eff = 2.0 / (1.0 / N1 + 1.0 / N2)
                xs.append(1.0 / N_eff)
                ys.append(xc)
                labels.append(f"({N1},{N2})")
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            ax_extrap.plot(xs_arr, ys_arr, "o", ms=8, color=self.colors["m2"], zorder=5)
            for x, y, lab in zip(xs, ys, labels):
                ax_extrap.annotate(lab, (x, y), xytext=(5, 4),
                                   textcoords="offset points", fontsize=9)
            if len(xs_arr) >= 2:
                slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
                x_line = np.linspace(0, xs_arr.max() * 1.1, 50)
                y_line = slope * x_line + intercept
                ax_extrap.plot(x_line, y_line, "--", lw=1.2, color=self.colors["nqs"],
                               label=rf"linear fit: $J_c^\infty={intercept:.3f}$")
                ax_extrap.plot([0], [intercept], "*", ms=14, color=self.colors["nqs"],
                               zorder=6)
            ax_extrap.axhline(J_center, color=self.colors["guide"], lw=0.9,
                              linestyle=":", label=rf"exact: $J_c={J_center:.1f}$")
            ax_extrap.set_xlabel(
                r"$1/N_{\mathrm{eff}} = \frac{1}{2}(1/N_1 + 1/N_2)$"
            )
            ax_extrap.set_ylabel(r"crossing $J_c(N_1, N_2)$")
            ax_extrap.set_title(r"Finite-size extrapolation of the Binder crossing")
            ax_extrap.grid(True, alpha=0.18)
            ax_extrap.legend(frameon=False, loc="best", fontsize=9)
        else:
            ax_extrap.text(0.5, 0.5, "No clean crossings found",
                           transform=ax_extrap.transAxes,
                           ha="center", va="center")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_fss_order_parameter(self, dataset: ExperimentDataset, save_path: Path) -> None:
        """Log-log scaling of the order parameter at criticality.

        At a second-order quantum phase transition, the order parameter
        squared should decay with system size as a power law

          <m^2>(J_c, N) ~ N^(-2 beta / nu)

        For the 1D TFIM (= 2D classical Ising universality class) the
        expected exponents are beta = 1/8 and nu = 1, so the expected
        slope in log-log is -1/4 = -0.25.

        Left:  <m^2>(N) at J ~ -h  (ferro critical)
        Right: <n^2>(N) at J ~ +h  (antiferro critical)
        """
        N_values = dataset.N_values()
        h_val = dataset.metadata["model_config"]["h"]

        def _collect_at(J_center: float, field: str) -> tuple[np.ndarray, np.ndarray]:
            xs, ys = [], []
            for N in N_values:
                rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
                idx = int(np.argmin([abs(r.J - J_center) for r in rs]))
                val = getattr(rs[idx], field)
                if np.isfinite(val) and val > 0:
                    xs.append(N)
                    ys.append(val)
            return np.array(xs, dtype=float), np.array(ys, dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
        ax_ferro, ax_af = axes

        expected = -0.25  # -2 beta/nu for 2D Ising universality

        for ax, J_c, field, label, panel_title in [
            (ax_ferro, -h_val, "m2_final",
             rf"$\langle m^2 \rangle$ at $J\approx-h$",
             r"Ferro critical point ($J\approx -h$)"),
            (ax_af, +h_val, "n2_final",
             rf"$\langle n^2 \rangle$ at $J\approx+h$",
             r"Antiferro critical point ($J\approx +h$)"),
        ]:
            Ns, ys = _collect_at(J_c, field)
            if Ns.size < 2:
                ax.text(0.5, 0.5, "insufficient data",
                        transform=ax.transAxes, ha="center")
                continue
            ax.loglog(Ns, ys, "o", ms=8, color=self.colors["m2"], label="NQS")
            slope, intercept = np.polyfit(np.log(Ns), np.log(ys), 1)
            N_line = np.linspace(Ns.min() * 0.9, Ns.max() * 1.1, 100)
            y_fit = np.exp(intercept) * N_line ** slope
            ax.loglog(N_line, y_fit, "-", lw=1.3, color=self.colors["nqs"],
                      label=rf"fit: slope $={slope:.3f}$")
            # expected 2D-Ising slope reference, anchored at largest N
            y_ref = ys[-1] * (N_line / Ns[-1]) ** expected
            ax.loglog(N_line, y_ref, "--", lw=1.0, color=self.colors["thermo"],
                      label=rf"2D Ising: slope $={expected}$")
            ax.set_xlabel(r"$N$")
            ax.set_ylabel(label)
            ax.set_title(panel_title)
            ax.grid(True, which="both", alpha=0.18)
            ax.legend(frameon=False, loc="best", fontsize=9)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_curvature_peak_scaling(
        self,
        dataset: ExperimentDataset,
        zoom_cfg: CriticalZoomConfig,
        save_path: Path,
    ) -> None:
        """Peak of the energy curvature as a diagnostic of criticality.

        For a second-order transition the second derivative of the
        ground-state energy density with respect to J develops a
        diverging peak at J_c in the thermodynamic limit. In the 2D
        Ising universality class the specific heat exponent alpha = 0
        means the divergence is logarithmic, so the peak height should
        grow with system size roughly as log(N), and the peak location
        converges to J_c with a finite-size shift that vanishes as
        1/N.

        Left:  -d^2 E_0/dJ^2 zoomed on the antiferro transition, with
               peak positions marked.
        Upper right: peak height vs log(N).
        Lower right: peak location J*(N) vs 1/N, with linear
               extrapolation to J_c^infty.
        """
        from scipy.signal import savgol_filter
        from scipy.interpolate import interp1d

        N_values = dataset.N_values()
        color_by_N = _cmap_for_N(N_values)
        h_val = dataset.metadata["model_config"]["h"]

        J_af_center = +h_val
        hw = zoom_cfg.zoom_halfwidth
        Ja_lo, Ja_hi = J_af_center - hw, J_af_center + hw

        fig = plt.figure(figsize=(14, 5.8))
        gs = GridSpec(2, 2, width_ratios=[1.3, 1], hspace=0.45, wspace=0.28)
        ax_curv = fig.add_subplot(gs[:, 0])
        ax_hgt = fig.add_subplot(gs[0, 1])
        ax_loc = fig.add_subplot(gs[1, 1])

        peak_heights: list[tuple[int, float]] = []
        peak_locations: list[tuple[int, float]] = []

        for N in N_values:
            rs = sorted(dataset.results_for_N(N), key=lambda r: r.J)
            J = np.array([r.J for r in rs])
            e = np.array([r.e_final for r in rs])
            order = np.argsort(J)
            J = J[order]
            e = e[order]
            if J.size < 5:
                continue
            J_uni = np.linspace(J.min(), J.max(), 4 * len(J))
            try:
                e_uni = interp1d(J, e, kind="cubic")(J_uni)
            except Exception:
                e_uni = np.interp(J_uni, J, e)
            win = 11 if len(J_uni) >= 11 else (len(J_uni) // 2) * 2 + 1
            win = max(win, 5)
            delta = J_uni[1] - J_uni[0]
            d2 = savgol_filter(e_uni, window_length=win, polyorder=3,
                               deriv=2, delta=delta)
            neg = -d2

            # plot zoom
            m_zoom = (J_uni >= Ja_lo - 0.1) & (J_uni <= Ja_hi + 0.1)
            c = color_by_N[N]
            ax_curv.plot(J_uni[m_zoom], neg[m_zoom], "-", lw=1.4,
                         color=c, label=rf"$N={N}$")

            # find peak in antiferro window
            m_search = (J_uni >= Ja_lo) & (J_uni <= Ja_hi)
            if not m_search.any():
                continue
            idx_rel = int(np.argmax(neg[m_search]))
            J_peak = J_uni[m_search][idx_rel]
            y_peak = neg[m_search][idx_rel]
            ax_curv.plot([J_peak], [y_peak], "*", ms=12, color=c,
                         markeredgecolor="k", markeredgewidth=0.5, zorder=6)
            peak_heights.append((N, y_peak))
            peak_locations.append((N, J_peak))

        ax_curv.axvline(J_af_center, color=self.colors["guide"], lw=0.9, linestyle=":")
        ax_curv.set_xlabel(r"$J$")
        ax_curv.set_ylabel(r"$-d^2 E_0/dJ^2 \,/\, N$")
        ax_curv.set_title(
            rf"Energy curvature near $J_c^{{\mathrm{{AF}}}} = {J_af_center:.1f}$"
        )
        ax_curv.grid(True, alpha=0.18)
        ax_curv.legend(frameon=False, loc="best", fontsize=9)

        # Peak-height panel: log(N) on x-axis (alpha=0 -> log divergence)
        if peak_heights:
            Ns = np.array([p[0] for p in peak_heights], dtype=float)
            hs = np.array([p[1] for p in peak_heights], dtype=float)
            ax_hgt.semilogx(Ns, hs, "o", ms=8, color=self.colors["m2"])
            if Ns.size >= 2:
                slope, intercept = np.polyfit(np.log(Ns), hs, 1)
                N_line = np.logspace(np.log10(Ns.min() * 0.9),
                                     np.log10(Ns.max() * 1.1), 80)
                ax_hgt.semilogx(
                    N_line, slope * np.log(N_line) + intercept,
                    "--", lw=1.2, color=self.colors["nqs"],
                    label=rf"$a\,\log N + b$, $a={slope:.3f}$",
                )
                ax_hgt.legend(frameon=False, loc="best", fontsize=9)
        ax_hgt.set_xlabel(r"$N$")
        ax_hgt.set_ylabel(r"peak of $-d^2 E_0/dJ^2$")
        ax_hgt.set_title(r"Peak height (expected $\sim\log N$ for $\alpha=0$)")
        ax_hgt.grid(True, which="both", alpha=0.18)

        # Peak-location panel: extrapolate in 1/N
        if peak_locations:
            Ns = np.array([p[0] for p in peak_locations], dtype=float)
            Js = np.array([p[1] for p in peak_locations], dtype=float)
            inv = 1.0 / Ns
            ax_loc.plot(inv, Js, "o", ms=8, color=self.colors["m2"])
            if Ns.size >= 2:
                slope, intercept = np.polyfit(inv, Js, 1)
                x_line = np.linspace(0, inv.max() * 1.1, 60)
                ax_loc.plot(x_line, slope * x_line + intercept, "--",
                            lw=1.2, color=self.colors["nqs"],
                            label=rf"fit: $J_c^\infty={intercept:.3f}$")
                ax_loc.plot([0], [intercept], "*", ms=14,
                            color=self.colors["nqs"], zorder=6)
            ax_loc.axhline(J_af_center, color=self.colors["guide"], lw=0.9,
                           linestyle=":", label=rf"exact: $J_c={J_af_center:.1f}$")
            ax_loc.legend(frameon=False, loc="best", fontsize=9)
        ax_loc.set_xlabel(r"$1/N$")
        ax_loc.set_ylabel(r"peak location $J^{\star}(N)$")
        ax_loc.set_title(r"Peak location converges to $J_c$")
        ax_loc.grid(True, alpha=0.18)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
