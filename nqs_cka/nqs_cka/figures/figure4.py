from __future__ import annotations

from .figure3 import make_figure3


def make_figure4(items, shape, out_dir: str, filename: str = "figure3-simple_2d_ising.png"):
    return make_figure3(
        items,
        shape,
        out_dir,
        min_distance=2,
        filename=filename,
        title="Simple 2D Ising / TFIM baseline",
    )
