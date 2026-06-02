from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .physics import born_sample_exact


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeRecord:
    samples: np.ndarray

    @staticmethod
    def from_raw(raw_record: dict[str, Any]) -> ProbeRecord:
        return ProbeRecord(samples=np.asarray(raw_record["samples"]))


@dataclass(frozen=True)
class ProbeBatch:
    samples: np.ndarray

    @property
    def size(self) -> int:
        return int(self.samples.shape[0])


def pooled_probe(records: list[dict[str, Any]], n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a shared probe by pooling samples from several training records.

    What happens:
    - Split the requested probe budget roughly evenly across records.
    - Randomly take up to that many samples from each record.
    - Concatenate those per-record chunks.
    - Draw the final probe of exactly `n_samples` rows from the pooled candidates.

    Example:
        records = [
            {"samples": np.array([[0, 1], [1, 0], [1, 1]])},
            {"samples": np.array([[0, 0], [1, 1], [0, 1]])},
        ]

        pooled_probe(records, n_samples=4, rng=rng)

        With two records, this first takes about ceil(4 / 2) = 2 samples
        from each record, giving 4 pooled candidates. It then shuffles/resamples
        those candidates to return the final shared probe.

    This is used when we do not have an exact wavefunction. In that case, we use
    samples produced by trained models as the probe distribution.
    """
    probe_records = [ProbeRecord.from_raw(record) for record in records]
    probe_batch = _sample_pooled_probe(probe_records, n_samples, rng)
    return probe_batch.samples


def shared_probe(
    hilbert_space: Any,
    exact_wavefunction: Any | None,
    records: list[dict[str, Any]],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build the common probe samples used by all analyses for one problem.

    If an exact wavefunction is available, samples are drawn from the exact Born
    distribution. Otherwise, samples are pooled from the available trained runs.

    Example:
        # Small system: exact wavefunction exists.
        probe = shared_probe(hi, psi0, records, 2048, rng)
        # -> samples from |psi0|^2

        # Larger system: exact wavefunction is unavailable.
        probe = shared_probe(hi, None, records, 2048, rng)
        # -> samples pooled from records[i]["samples"]
    """
    if exact_wavefunction is not None:
        logger.info("building shared probe from exact Born samples")
        return born_sample_exact(hilbert_space, exact_wavefunction, n_samples, rng)

    logger.info("building shared probe from pooled model samples")
    return pooled_probe(records, n_samples, rng)


def _sample_pooled_probe(records: list[ProbeRecord], n_samples: int, rng: np.random.Generator) -> ProbeBatch:
    samples_per_record = _samples_per_record(n_samples, len(records))
    sample_chunks = [_sample_record_chunk(record, samples_per_record, rng) for record in records]
    pooled_samples = np.concatenate(sample_chunks, axis=0)
    selected_indices = rng.choice(pooled_samples.shape[0], size=n_samples, replace=pooled_samples.shape[0] < n_samples)
    return ProbeBatch(samples=pooled_samples[selected_indices])


def _sample_record_chunk(record: ProbeRecord, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    sample_count = min(max_samples, record.samples.shape[0])
    selected_indices = rng.choice(record.samples.shape[0], size=sample_count, replace=False)
    return record.samples[selected_indices]


def _samples_per_record(n_samples: int, n_records: int) -> int:
    return max(1, int(np.ceil(n_samples / max(1, n_records))))