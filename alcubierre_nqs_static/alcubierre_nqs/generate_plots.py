from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, load_config
from .pipeline import generate_plots_from_saved


def parse_args() -> tuple[Config, str | None]:
    p = argparse.ArgumentParser(
        description=(
            "Plot only."
        )
    )
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--config-path", type=str, default=None)
    p.add_argument("--matrices-path", type=str, default=None)

    p.add_argument("--heatmap-percentile", type=float, default=None)
    p.add_argument("--wall-overlay", action="store_true", default=None, help="Force wall overlay on.")
    p.add_argument("--no-wall-overlay", action="store_true", help="Force wall overlay off.")
    p.add_argument("--plot-halfwidth", type=float, default=None)
    p.add_argument("--surface-interp-factor", type=int, default=None)
    p.add_argument("--surface-target-relief", type=float, default=None)
    p.add_argument("--manual-surface-height-scale", type=float, default=None)
    p.add_argument("--surface-elev", type=float, default=None)
    p.add_argument("--surface-azim", type=float, default=None)
    p.add_argument("--surface-levels", type=int, default=None)

    p.add_argument("--bubble-compare-key", type=str, default=None)
    p.add_argument("--bubble-surface-mode", type=str, choices=["wall", "top_hat"], default=None)
    p.add_argument("--bubble-surface-target-relief", type=float, default=None)
    p.add_argument("--bubble-plane-alpha", type=float, default=None)
    p.add_argument("--bubble-plot-points", type=int, default=None)
    p.add_argument("--bubble-wire-stride", type=int, default=None)
    p.add_argument("--bubble-observable-contours", type=int, default=None)

    args = p.parse_args()

    outdir = Path(args.outdir)
    config_path = Path(args.config_path) if args.config_path else outdir / "validation_config_3p1.json"
    cfg = load_config(config_path)
    cfg.outdir = str(outdir)

    # Apply only explicit plotting overrides.
    for name in [
        "heatmap_percentile",
        "plot_halfwidth",
        "surface_interp_factor",
        "surface_target_relief",
        "manual_surface_height_scale",
        "surface_elev",
        "surface_azim",
        "surface_levels",
        "bubble_compare_key",
        "bubble_surface_mode",
        "bubble_surface_target_relief",
        "bubble_plane_alpha",
        "bubble_plot_points",
        "bubble_wire_stride",
        "bubble_observable_contours",
    ]:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    if args.no_wall_overlay:
        cfg.wall_overlay = False
    elif args.wall_overlay:
        cfg.wall_overlay = True

    return cfg, args.matrices_path


def main() -> None:
    cfg, matrices_path = parse_args()
    generate_plots_from_saved(cfg, matrices_path=matrices_path)


if __name__ == "__main__":
    main()
