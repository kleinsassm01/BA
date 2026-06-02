from __future__ import annotations

import gzip
import hashlib
import logging
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import netket as nk
import numpy as np

from ..config import Config, ModelConfig, PhysicsConfig, TrainingConfig, model_with_blocks, training_for_depth
from ..models import GraphNQS


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainRun:
    model: GraphNQS
    params: Any
    energy: float
    samples: np.ndarray

    def to_record(self) -> dict[str, Any]:
        return {"model": self.model, "params": self.params, "energy": self.energy, "samples": self.samples}


@dataclass(frozen=True)
class CacheSpec:
    problem: PhysicsConfig
    model_cfg: ModelConfig
    train_cfg: TrainingConfig
    mult: int
    blocks: int
    seed: int
    retry: int

    @property
    def filename(self) -> str:
        data = {
            "problem": asdict(self.problem),
            "model": asdict(self.model_cfg),
            "train": asdict(self.train_cfg),
            "mult": int(self.mult),
            "blocks": int(self.blocks),
            "seed": int(self.seed),
            "retry": int(self.retry),
        }
        digest = hashlib.sha1(repr(sorted(data.items())).encode()).hexdigest()[:12]
        architecture = "res" if self.model_cfg.residual else "plain"
        return f"{self.problem.name}_{architecture}_{self.mult}x_B{self.blocks}_seed{self.seed}_r{self.retry}_{digest}.pkl.gz"


def rounded_samples(n_samples: int, n_chains: int) -> int:
    chain_count = max(1, int(n_chains))
    return int(math.ceil(int(n_samples) / chain_count) * chain_count)


def largest_divisor(n: int, requested: int) -> int:
    requested = max(1, min(int(requested), int(n)))

    for chunk_size in range(requested, 0, -1):
        if n % chunk_size == 0:
            return chunk_size

    return 1


def cache_key(
    problem: PhysicsConfig,
    model_cfg: ModelConfig,
    train: TrainingConfig,
    mult: int,
    blocks: int,
    seed: int,
    retry: int,
) -> str:
    return CacheSpec(problem, model_cfg, train, int(mult), int(blocks), int(seed), int(retry)).filename


def save_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_record(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def train_one(
    hamiltonian: Any,
    hi: Any,
    shape: tuple[int, ...],
    model_cfg: ModelConfig,
    train: TrainingConfig,
    seed: int,
) -> TrainRun:
    effective_samples = rounded_samples(train.n_samples, train.n_chains)
    safe_chunk_size = largest_divisor(effective_samples, train.chunk_size)

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=train.n_chains)
    model = GraphNQS(shape=shape, cfg=model_cfg)
    vstate = nk.vqs.MCState(sampler, model, n_samples=train.n_samples, seed=seed, chunk_size=safe_chunk_size)
    optimizer = nk.optimizer.Sgd(learning_rate=train.learning_rate)

    driver = nk.driver.VMC_SR(
        hamiltonian,
        optimizer,
        variational_state=vstate,
        diag_shift=train.diag_shift,
        use_ntk=True,
        on_the_fly=True,
        chunk_size_bwd=safe_chunk_size,
    )
    driver.run(n_iter=train.n_iter)

    energy = float(np.real(vstate.expect(hamiltonian).mean))
    samples = np.asarray(vstate.samples).reshape((-1, int(np.prod(shape))))
    return TrainRun(model=model, params=vstate.parameters, energy=energy, samples=samples)


def _record_metadata(seed: int, retry: int, mult: int, blocks: int, residual: bool, problem: PhysicsConfig, model_cfg: ModelConfig) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "retry": int(retry),
        "mult": int(mult),
        "blocks": int(blocks),
        "residual": bool(residual),
        "problem": asdict(problem),
        "model_cfg": asdict(model_cfg),
    }


def _cache_path(cfg: Config, problem: PhysicsConfig, model_cfg: ModelConfig, train_cfg: TrainingConfig, mult: int, blocks: int, seed: int, retry: int) -> Path:
    return cfg.cache_path / cache_key(problem, model_cfg, train_cfg, mult, blocks, seed, retry)


def _load_cached_record(path: Path, problem: PhysicsConfig, model_cfg: ModelConfig) -> dict[str, Any]:
    record = load_record(path)
    record["model"] = GraphNQS(shape=problem.shape, cfg=model_cfg)
    return record


def _train_and_cache_record(
    path: Path,
    problem: PhysicsConfig,
    hamiltonian: Any,
    hi: Any,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    *,
    seed: int,
    retry: int,
    mult: int,
    blocks: int,
    residual: bool,
) -> dict[str, Any] | None:
    logger.info(
        "seed %s: residual=%s, B=%s, lr=%g, shift=%g",
        seed,
        residual,
        blocks,
        train_cfg.learning_rate,
        train_cfg.diag_shift,
    )

    run = train_one(hamiltonian, hi, problem.shape, model_cfg, train_cfg, seed)
    if not np.isfinite(run.energy):
        logger.warning("seed %s retry %s produced non-finite energy; skipping", seed, retry)
        return None

    record = run.to_record()
    record.update(_record_metadata(seed, retry, mult, blocks, residual, problem, model_cfg))
    save_record(path, {key: value for key, value in record.items() if key != "model"})
    return record


def train_or_load_depth(
    cfg: Config,
    problem: PhysicsConfig,
    hamiltonian: Any,
    hi: Any,
    *,
    mult: int,
    residual: bool,
    seeds: list[int],
    force: bool = False,
    train_missing: bool = True,
) -> list[dict[str, Any]]:
    blocks = int(mult) * int(cfg.analysis.blocks_per_unit)
    records: list[dict[str, Any]] = []

    for seed in seeds:
        for retry in range(cfg.training.retries + 1):
            model_cfg = model_with_blocks(cfg.model, blocks, residual=residual, retry=retry, train=cfg.training)
            train_cfg = training_for_depth(cfg.training, mult, retry=retry)
            path = _cache_path(cfg, problem, model_cfg, train_cfg, mult, blocks, seed, retry)

            if path.exists() and not force:
                records.append(_load_cached_record(path, problem, model_cfg))
                break

            if not path.exists() and not train_missing:
                continue

            record = _train_and_cache_record(
                path,
                problem,
                hamiltonian,
                hi,
                model_cfg,
                train_cfg,
                seed=seed,
                retry=retry,
                mult=mult,
                blocks=blocks,
                residual=residual,
            )
            if record is not None:
                records.append(record)
                break

    return sorted(records, key=lambda record: record["energy"])