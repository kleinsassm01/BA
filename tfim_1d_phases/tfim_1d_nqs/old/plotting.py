from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.gridspec import GridSpec

from .exact_solver import ExactIsingSolver
from .models import TrainingResult

plt.style.use(["science"])


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
        J_vals = np.array([r.J for r in results])
        m2_vals = np.array([r.m2_final for r in results])
        n2_vals = np.array([r.n2_final for r in results])
        e_nqs = np.array([r.e_final for r in results])
        e_exact_f = np.array([r.e_exact_finite for r in results])
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

        ax_main.plot(
            J_vals, m2_vals,
            linestyle="None", marker="o", ms=4,
            color=self.colors["m2"],
            label=r"$\langle m^2 \rangle$",
            zorder=3,
        )
        ax_main.plot(
            J_vals, n2_vals,
            linestyle="None", marker="o", ms=4,
            color=self.colors["n2"],
            label=r"$\langle n^2 \rangle$",
            zorder=3,
        )

        ax_main.set_xlabel(r"$J$")
        ax_main.set_ylabel(r"$\langle m^2 \rangle / \langle n^2 \rangle$")
        ax_main.set_title(rf"1D TFIM phase diagram ($h={h_val:.1f},\, N={N}$)")
        ax_main.set_xlim(J_vals.min() - 0.15, J_vals.max() + 0.15)
        ax_main.set_ylim(-0.02, 1.03)
        ax_main.legend(frameon=False, loc="best")
        ax_main.grid(True, alpha=0.18)

        ax_e = fig.add_subplot(gs[1, 0])

        ax_e.plot(
            J_dense, e_exact_dense_f,
            "-", color=self.colors["finite"], lw=1.5,
            label=rf"exact finite $N={N}$",
        )
        ax_e.plot(
            J_dense, e_exact_dense_t,
            "--", color=self.colors["thermo"], lw=1,
            label=r"exact $N\to\infty$",
        )
        ax_e.plot(
            J_vals, e_nqs,
            linestyle="None", marker="o", ms=4,
            color=self.colors["nqs"],
            label="NQS",
            zorder=4,
        )

        ax_e.set_xlabel(r"$J$")
        ax_e.set_ylabel(r"$E_0/N$")
        ax_e.set_title("ground-state energy")
        ax_e.legend(frameon=False, loc="best")
        ax_e.grid(True, alpha=0.18)

        ax_err = fig.add_subplot(gs[1, 1])

        rel_error = np.abs(e_nqs - e_exact_f) / np.maximum(np.abs(e_exact_f), 1e-14) * 100
        ax_err.plot(
            J_vals, rel_error,
            "--", color=self.colors["guide"], lw=1.0, zorder=1
        )
        ax_err.plot(
            J_vals, rel_error,
            linestyle="None", marker="o", ms=4,
            color=self.colors["m2"], zorder=3
        )

        ax_err.set_xlabel(r"$J$")
        ax_err.set_ylabel("relative error (\%)")
        ax_err.set_title("accuracy")
        ax_err.set_yscale("log")
        ax_err.grid(True, which="both", alpha=0.18)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

    def plot_training_convergence(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        J_vals = np.array([r.J for r in results])
        idx_af = np.argmin(J_vals)
        idx_para = np.argmin(np.abs(J_vals))
        idx_ferro = np.argmax(J_vals)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)

        panels = [
            (axes[0], idx_af, f"antiferro ($J={J_vals[idx_af]:.1f}$)"),
            (axes[1], idx_para, f"para ($J={J_vals[idx_para]:.1f}$)"),
            (axes[2], idx_ferro, f"ferro ($J={J_vals[idx_ferro]:.1f}$)"),
        ]

        for ax, idx, title in panels:
            r = results[idx]

            ax.plot(
                r.history.iters,
                r.history.energy,
                linestyle="None",
                marker="o",
                ms=2.0,
                color=self.colors["nqs"],
                alpha=0.85,
                label="NQS",
                zorder=3,
            )

            ax.axhline(
                r.e_exact_finite,
                color=self.colors["finite"],
                lw=1.0,
                linestyle="-",
                label=rf"exact finite $N={N}$",
                zorder=1,
            )

            ax.axhline(
                r.e_exact_thermo,
                color=self.colors["thermo"],
                lw=1.0,
                linestyle="--",
                label=r"exact $N\to\infty$",
                zorder=1,
            )

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
            axes[0].plot(
                r.history.iters,
                r.history.m2,
                linestyle="None",
                marker="o",
                ms=1.7,
                color=color,
                alpha=0.55,
            )
            axes[1].plot(
                r.history.iters,
                r.history.n2,
                linestyle="None",
                marker="o",
                ms=1.7,
                color=color,
                alpha=0.55,
            )

        axes[0].set_title(r"$\langle m^2 \rangle$")
        axes[1].set_title(r"$\langle n^2 \rangle$")

        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(True, alpha=0.18)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()