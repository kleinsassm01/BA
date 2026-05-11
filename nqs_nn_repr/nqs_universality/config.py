from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:

    out_dir: Path = Path("results")

    N: int = 20
    J: float = 1.0
    h: float = 1.0
    pbc: bool = True

    n_samples_train: int = 4096
    n_iter: int = 800
    lr: float = 0.005
    use_sr: bool = False

    n_activation_samples: int = 4096
    n_uniform_probe_samples: int = 4096
    n_tangent_samples: int = 256
    n_saliency_samples: int = 1024

    activation_batch_size: int = 512
    saliency_batch_size: int = 128
    jacobian_batch_size: int = 8

    ridge_alpha: float = 1e-3
    probe_train_frac: float = 0.7
    max_decode_distance: int = 10
    max_corr_distance: int = 10
    ntk_top_k: int = 50

    ref_name: str = "CNN-3layer-k3"

    cka_cmap: str = "magma"
    cka_vmin: float = 0.0
    cka_vmax: float = 1.0
    sim_vmin: float = 0.70
    sim_vmax: float = 1.0
    save_pdf: bool = True

    def with_out_dir(self, out_dir: str | Path) -> "ExperimentConfig":
        self.out_dir = Path(out_dir)
        return self


def default_config(out_dir: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if out_dir is not None:
        cfg.out_dir = Path(out_dir)
    return cfg
