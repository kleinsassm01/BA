from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

from .config import default_config
from .io_utils import load_all_data
from .plotting import make_all_plots


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="results")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = default_config(Path(args.out_dir))
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


if __name__ == "__main__":
    main()
