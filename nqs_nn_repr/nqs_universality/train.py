from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

from .config import default_config
from .experiment import run_experiment
from .io_utils import load_all_data
from .model_registry import make_configs, select_train_names
from .plotting import make_all_plots


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default=None
    )
    parser.add_argument(
        "--train_only",
        nargs="*",
        metavar="MODEL",
        help="train only specified exact model names. Example: --train_only GNN-3layer-local",
    )
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory.")
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="regenerate all plots after training.",
    )
    parser.add_argument(
        "--plots_only",
        "--plot_only",
        action="store_true",
        help="regenerate plots from saved data without training.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    cfg = default_config(Path(args.out_dir))

    if args.plots_only:
        (
            metadata,
            act_cka,
            act_phys,
            _act_unif,
            full_tangents,
            layer_tangents,
            saliency_all,
            local_all,
            multidist,
            correlations,
        ) = load_all_data(cfg)
        make_all_plots(
            cfg,
            metadata,
            act_cka,
            act_phys,
            full_tangents,
            layer_tangents,
            saliency_all,
            local_all,
            multidist,
            correlations,
        )
        return

    configs = make_configs()
    train_names = select_train_names(args.train, args.train_only, configs)
    run_experiment(cfg, train_names, make_plots=args.make_plots)


if __name__ == "__main__":
    main()
