from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .exact_solver import ExactIsingSolver
from .models import TrainingResult


class ResultPlotter:
    def __init__(self, exact_solver: ExactIsingSolver | None = None) -> None:
        self.exact_solver = exact_solver or ExactIsingSolver()

    def plot_phase_diagram(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        J_vals = np.array([r.J for r in results])
        m2_vals = np.array([r.m2_final for r in results])
        n2_vals = np.array([r.n2_final for r in results])
        e_nqs = np.array([r.e_final for r in results])
        e_exact_f = np.array([r.e_exact_finite for r in results])
        h_val = results[0].h

        J_dense = np.linspace(J_vals.min(), J_vals.max(), 500)
        e_exact_dense_f = np.array([self.exact_solver.energy_finite(N, J, h_val) for J in J_dense])
        e_exact_dense_t = np.array([self.exact_solver.energy_thermodynamic(J, h_val) for J in J_dense])

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, height_ratios=[1.3, 1], hspace=0.35, wspace=0.3)

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(J_vals, m2_vals, 'o-', label=r'$\langle m^2 \rangle$')
        ax_main.plot(J_vals, n2_vals, 's-', label=r'$\langle n^2 \rangle$')
        ax_main.set_xlabel(r'$J$')
        ax_main.set_ylabel(r'Order parameter')
        ax_main.set_title(rf'1D TFIM Phase Diagram ($h={h_val:.1f}, N={N}$)')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        ax_e = fig.add_subplot(gs[1, 0])
        ax_e.plot(J_dense, e_exact_dense_f, '-', label=rf'Exact finite $N={N}$')
        ax_e.plot(J_dense, e_exact_dense_t, '--', label=r'Exact $N\to\infty$')
        ax_e.plot(J_vals, e_nqs, 'o', label='NQS')
        ax_e.set_xlabel(r'$J$')
        ax_e.set_ylabel(r'$E_0/N$')
        ax_e.set_title('Energy')
        ax_e.legend()
        ax_e.grid(True, alpha=0.3)

        ax_err = fig.add_subplot(gs[1, 1])
        rel_error = np.abs(e_nqs - e_exact_f) / np.abs(e_exact_f) * 100
        ax_err.semilogy(J_vals, rel_error, 'D-')
        ax_err.set_xlabel(r'$J$')
        ax_err.set_ylabel('Relative error (%)')
        ax_err.set_title('NQS Accuracy')
        ax_err.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_training_convergence(self, results: list[TrainingResult], N: int, save_path: Path) -> None:
        J_vals = np.array([r.J for r in results])
        idx_af = np.argmin(J_vals)
        idx_para = np.argmin(np.abs(J_vals))
        idx_ferro = np.argmax(J_vals)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, idx, title in [
            (axes[0], idx_af, f'Antiferro (J={J_vals[idx_af]:.1f})'),
            (axes[1], idx_para, f'Para (J={J_vals[idx_para]:.1f})'),
            (axes[2], idx_ferro, f'Ferro (J={J_vals[idx_ferro]:.1f})'),
        ]:
            r = results[idx]
            ax.plot(r.history.iters, r.history.energy, label='NQS')
            ax.axhline(r.e_exact_finite, linestyle='-', label=f'Exact N={N}')
            ax.axhline(r.e_exact_thermo, linestyle='--', label='Exact thermodynamic')
            ax.set_title(title)
            ax.set_xlabel('Training step')
            ax.set_ylabel(r'$E_0/N$')
            ax.grid(True, alpha=0.3)
            ax.legend()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_training_histories(self, results: list[TrainingResult], save_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cmap = plt.cm.coolwarm
        J_vals = np.array([r.J for r in results])
        norm = plt.Normalize(J_vals.min(), J_vals.max())

        for r in results:
            color = cmap(norm(r.J))
            axes[0].plot(r.history.iters, r.history.m2, color=color, alpha=0.7)
            axes[1].plot(r.history.iters, r.history.n2, color=color, alpha=0.7)

        axes[0].set_title(r'$\langle m^2 \rangle$ during training')
        axes[1].set_title(r'$\langle n^2 \rangle$ during training')
        for ax in axes:
            ax.set_xlabel('Training step')
            ax.grid(True, alpha=0.3)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()