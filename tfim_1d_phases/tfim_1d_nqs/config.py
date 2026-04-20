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
    n_samples: int = 2048
    lr: float = 0.01
    n_chains: int = 16
    n_discard_per_chain: int = 100
    sr_diag_shift: float = 0.01
    log_every: int = 5


@dataclass(frozen=True)
class ScanConfig:
    # 11 points, concentrated away from |J|=h so that the total
    # sweep (coarse + zoom) stays under ~35 points per N i.e. because long training time
    J_values: np.ndarray

    @staticmethod
    def default() -> "ScanConfig":
        return ScanConfig(
            J_values=np.concatenate([
                np.linspace(-3.0, -1.5, 4),
                np.linspace(-0.5, 0.5, 3),
                np.linspace(1.5, 3.0, 4),
            ])
        )


@dataclass(frozen=True)
class MultiNConfig:
    N_values: tuple[int, ...] = (4, 8, 16, 32, 64)


@dataclass(frozen=True)
class CriticalZoomConfig:
    J_center_ferro: float = 1.0        # J > 0 -> AF transition in NetKet sign
    J_center_antiferro: float = -1.0   # J < 0 -> FM transition in NetKet sign
    zoom_halfwidth: float = 0.4
    n_points_per_side: int = 12

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
        return np.unique(np.concatenate([self.antiferro_window(), self.ferro_window()]))


@dataclass(frozen=True)
class AutocorrConfig:
    enabled: bool = True
    n_samples: int = 4096
    n_chains: int = 1
    n_discard: int = 200
    max_lag: int = 400
    sokal_c: float = 5.0


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("./outputs")
    dataset_file: str = "dataset.json"
    phase_plot: str = "phase_diagram.png"
    convergence_plot: str = "training_convergence.png"
    histories_plot: str = "training_histories.png"
    overlay_plot: str = "multi_N_overlay.png"
    critical_zoom_plot: str = "critical_zoom.png"
    binder_plot: str = "binder_cumulant.png"
    tau_step_plot: str = "tau_corr_vs_step.png"
    tau_vs_J_plot: str = "tau_int_vs_J.png"
    energy_variance_plot: str = "energy_variance.png"
    binder_crossings_plot: str = "binder_crossings.png"
    fss_order_parameter_plot: str = "fss_order_parameter.png"
    curvature_peak_scaling_plot: str = "curvature_peak_scaling.png"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
