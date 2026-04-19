from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    N: int = 10
    h: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    n_iter: int = 300
    alpha: int = 4
    n_samples: int = 512
    lr: float = 0.01
    n_chains: int = 16
    n_discard_per_chain: int = 100
    sr_diag_shift: float = 0.01
    log_every: int = 5


@dataclass(frozen=True)
class ScanConfig:
    """Coarse J-sweep covering the whole phase diagram."""
    J_values: np.ndarray

    @staticmethod
    def default() -> "ScanConfig":
        return ScanConfig(
            J_values=np.concatenate([
                np.linspace(-3.0, -0.3, 7),
                np.array([-0.1, 0.0, 0.1]),
                np.linspace(0.3, 3.0, 7),
            ])
        )


# ============================================================================
# EXTENSION: multi-N sweep + critical-region zoom + autocorrelation analysis
# ============================================================================

@dataclass(frozen=True)
class MultiNConfig:
    """
    List of system sizes N for the overlay scan.

    Default follows the user's request: the original N=10, times 2, 4, 6, 8.
    """
    N_values: tuple[int, ...] = (10, 20, 40, 60, 80)


@dataclass(frozen=True)
class CriticalZoomConfig:
    """
    Fine-grained J-scan around the quantum critical points at J = +/- h.

    The 1D TFIM has a *second-order* quantum phase transition at |J|=h
    (with h=1, so at J=+-1). We zoom in on both sides to resolve how the
    order parameter <m^2> and the Binder cumulant U_4 behave as N grows.
    """
    J_center_ferro: float = 1.0
    J_center_antiferro: float = -1.0
    zoom_halfwidth: float = 0.5
    n_points_per_side: int = 11   # odd -> includes the exact center point

    def ferro_window(self) -> np.ndarray:
        return np.linspace(
            self.J_center_ferro - self.zoom_halfwidth,
            self.J_center_ferro + self.zoom_halfwidth,
            self.n_points_per_side,
        )

    def antiferro_window(self) -> np.ndarray:
        return np.linspace(
            self.J_center_antiferro - self.zoom_halfwidth,
            self.J_center_antiferro + self.zoom_halfwidth,
            self.n_points_per_side,
        )

    def combined_window(self) -> np.ndarray:
        """Both sides merged + sorted (de-duped)."""
        return np.unique(np.concatenate([self.antiferro_window(), self.ferro_window()]))


@dataclass(frozen=True)
class AutocorrConfig:
    """
    Settings for the *post-training* dedicated MCMC run used to measure the
    integrated autocorrelation time tau_int of the energy at the optimized
    variational parameters.

    The built-in per-step tau_corr is always recorded during training; this
    block only controls the extra dedicated analysis.
    """
    enabled: bool = True
    n_samples: int = 4096          # MC steps per chain in the dedicated run
    n_chains: int = 1              # single chain is cleanest for ACF estimation
    n_discard: int = 200           # burn-in for the dedicated chain
    max_lag: int = 400             # maximum lag in the autocorrelation function
    sokal_c: float = 5.0           # Sokal window constant; W = min t with t >= c*tau(t)


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("./outputs")
    dataset_file: str = "dataset.json"
    phase_plot: str = "phase_diagram.png"
    convergence_plot: str = "training_convergence.png"
    histories_plot: str = "training_histories.png"
    # --- new plots (multi-N extension) --------------------------------------
    overlay_plot: str = "multi_N_overlay.png"
    critical_zoom_plot: str = "critical_zoom.png"
    binder_plot: str = "binder_cumulant.png"
    tau_step_plot: str = "tau_corr_vs_step.png"
    tau_vs_J_plot: str = "tau_int_vs_J.png"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
