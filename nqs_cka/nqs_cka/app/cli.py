from __future__ import annotations

import argparse
import logging

from ..config import load_config
from ..workflow import run


def configure_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    configure_logging(verbose=not args.quiet)
    run(load_config(args.config), force=args.force, plot_only=False)


def plot_main() -> None:
    args = _parser().parse_args()
    configure_logging(verbose=not args.quiet)
    run(load_config(args.config), force=False, plot_only=True)