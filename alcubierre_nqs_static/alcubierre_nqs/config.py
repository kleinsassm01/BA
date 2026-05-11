from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    Nx: int = 12
    Ny: int = 12
    Nz: int = 12
    Lx: float = 24.0
    Ly: float = 24.0
    Lz: float = 24.0
    mass: float = 0.25
    v: float = 0.6
    R: float = 4.0
    sigma: float = 1.5

    steps: int = 4000
    lr: float = 8e-5
    seed: int = 1234
    init_noise: float = 1e-4
    grad_clip: float = 10.0

    outdir: str = "validation_3p1_surfaceplots_improved"
    save_matrices: bool = False
    make_plots: bool = True

    heatmap_percentile: float = 100.0
    wall_overlay: bool = True
    plot_halfwidth: Optional[float] = None
    surface_interp_factor: int = 6
    surface_target_relief: float = 0.16
    manual_surface_height_scale: Optional[float] = None
    surface_elev: float = 34.0
    surface_azim: float = -60.0
    surface_levels: int = 11

    bubble_compare_key: str = "rho_sub_c"
    bubble_surface_mode: str = "wall"  # "wall": v d_x f lobes; "top_hat": f(r)
    bubble_surface_target_relief: float = 0.32
    bubble_plane_alpha: float = 0.22
    bubble_plot_points: int = 181
    bubble_wire_stride: int = 8
    bubble_observable_contours: int = 13


def config_field_names() -> set[str]:
    return {field.name for field in fields(Config)}


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)


def config_from_dict(data: dict[str, Any]) -> Config:
    allowed = config_field_names()
    clean = {key: value for key, value in data.items() if key in allowed}
    return Config(**clean)


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2, sort_keys=True)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))
