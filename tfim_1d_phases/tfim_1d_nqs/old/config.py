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


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path("./outputs")
    dataset_file: str = "dataset.json"
    phase_plot: str = "phase_diagram.png"
    convergence_plot: str = "training_convergence.png"
    histories_plot: str = "training_histories.png"

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
