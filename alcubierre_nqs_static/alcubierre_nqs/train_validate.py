from __future__ import annotations

import argparse

from .config import Config
from .pipeline import train_validate


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="3+1D Alcubierre Gaussian NQS"
    )
    
    p.add_argument("--Nx", type=int, default=12)
    p.add_argument("--Ny", type=int, default=12)
    p.add_argument("--Nz", type=int, default=12)
    p.add_argument("--Lx", type=float, default=24.0)
    p.add_argument("--Ly", type=float, default=24.0)
    p.add_argument("--Lz", type=float, default=24.0)
    p.add_argument("--mass", type=float, default=0.25)
    p.add_argument("--v", type=float, default=0.6)
    p.add_argument("--R", type=float, default=4.0)
    p.add_argument("--sigma", type=float, default=1.5)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--init-noise", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=10.0)

    p.add_argument("--outdir", type=str, default="validation_3p1_surfaceplots_improved")
    p.add_argument("--save-matrices", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Train/save diagnostics but skip image generation.")

    p.add_argument("--heatmap-percentile", type=float, default=100.0)
    p.add_argument("--no-wall-overlay", action="store_true")
    p.add_argument("--plot-halfwidth", type=float, default=None)
    p.add_argument("--surface-interp-factor", type=int, default=6)
    p.add_argument("--surface-target-relief", type=float, default=0.16)
    p.add_argument("--manual-surface-height-scale", type=float, default=None)
    p.add_argument("--surface-elev", type=float, default=34.0)
    p.add_argument("--surface-azim", type=float, default=-60.0)
    p.add_argument("--surface-levels", type=int, default=11)

    p.add_argument("--bubble-plot-points", type=int, default=181)
    p.add_argument("--bubble-wire-stride", type=int, default=8)
    p.add_argument("--bubble-observable-contours", type=int, default=13)
    p.add_argument(
        "--bubble-compare-key",
        type=str,
        default="rho_sub_c",
        help=(
            "Observable key used for the NQS/analytic color comparison on the bubble profile. "
            "Examples: rho_sub_c, h_sub_c, Ttt_sub_c, Ttx_sub_c, pDxq_sym, q_var_sub."
        ),
    )
    p.add_argument(
        "--bubble-surface-mode",
        type=str,
        choices=["wall", "top_hat"],
        default="wall",
        help="3D bubble height: 'wall' = v partial_x f lobes; 'top_hat' = f(r).",
    )
    p.add_argument("--bubble-surface-target-relief", type=float, default=0.32)
    p.add_argument("--bubble-plane-alpha", type=float, default=0.22)

    args = p.parse_args()
    args_dict = vars(args).copy()

    no_wall_overlay = args_dict.pop("no_wall_overlay")
    no_plots = args_dict.pop("no_plots")

    cfg = Config(**args_dict)
    cfg.wall_overlay = not no_wall_overlay
    cfg.make_plots = not no_plots
    return cfg


def main() -> None:
    train_validate(parse_args())


if __name__ == "__main__":
    main()
