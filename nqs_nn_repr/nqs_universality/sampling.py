from __future__ import annotations

import numpy as np

from .config import ExperimentConfig


def get_shared_samples(vstate, n_samples: int, n_therm: int = 50):
    old = vstate.n_samples
    vstate.n_samples = n_samples

    for _ in range(n_therm):
        vstate.sample()

    s = np.asarray(vstate.samples).reshape(-1, vstate.hilbert.size)
    s = s[:n_samples].astype(np.float32)

    vstate.n_samples = old
    return s


def make_uniform_spin_samples(N: int, n_samples: int, seed: int = 12345):
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(n_samples, N),
    )


def load_existing_probe_samples(cfg: ExperimentConfig):
    samples_phys = None
    samples_unif = None
    samples_tang = None

    if (cfg.out_dir / "samples_physical.npy").exists():
        samples_phys = np.load(cfg.out_dir / "samples_physical.npy")
    if (cfg.out_dir / "samples_uniform.npy").exists():
        samples_unif = np.load(cfg.out_dir / "samples_uniform.npy")
    if (cfg.out_dir / "samples_tangent.npy").exists():
        samples_tang = np.load(cfg.out_dir / "samples_tangent.npy")

    return samples_phys, samples_unif, samples_tang
