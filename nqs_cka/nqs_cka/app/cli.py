from __future__ import annotations

import argparse

from ..config import load_config
from ..workflow import run


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    args = _parser().parse_args()
    run(load_config(args.config), force=args.force, plot_only=False)


def plot_main():
    args = _parser().parse_args()
    run(load_config(args.config), force=False, plot_only=True)
