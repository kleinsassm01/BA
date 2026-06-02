from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np


DistanceMetric = Literal["manhattan", "chebyshev"]


@dataclass(frozen=True)
class Shell:
    """All periodic lattice displacements at one graph distance."""

    distance: int
    displacements: list[tuple[int, ...]]


@dataclass(frozen=True)
class SpinSamples:
    """Spin samples reshaped as `(n_samples, *lattice_shape)` for local calculations."""

    values: np.ndarray
    shape: tuple[int, ...]

    @staticmethod
    def from_samples(samples, shape: tuple[int, ...]) -> SpinSamples:
        return SpinSamples(values=np.asarray(samples, dtype=float).reshape((-1, *shape)), shape=shape)

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def flat(self) -> np.ndarray:
        return self.values.reshape(self.n_samples, -1)

    @property
    def magnetization(self) -> np.ndarray:
        return self.flat.mean(axis=1, keepdims=True)

    @property
    def nearest_neighbor_average(self) -> np.ndarray:
        nearest_neighbors = np.zeros_like(self.values, dtype=float)
        neighbor_count = 0

        for axis in range(len(self.shape)):
            nearest_neighbors += np.roll(self.values, 1, axis=axis + 1) + np.roll(self.values, -1, axis=axis + 1)
            neighbor_count += 2

        return (nearest_neighbors / float(neighbor_count)).reshape(self.n_samples, -1)


def magnetization(samples) -> np.ndarray:
    """
    Compute the average spin value for each sample.

    Calculation:
        For samples with shape `(n_samples, n_sites)`, this returns:

            m_a = mean_i s_a(i)

        where `a` indexes the sample/configuration and `i` indexes lattice sites.

    Example:
        samples = np.array([
            [1, 1, -1, -1],
            [1, 1,  1, -1],
        ])

        magnetization(samples)
        # -> np.array([0.0, 0.5])
    """
    return np.asarray(samples, dtype=float).mean(axis=1)


def distance_shells(shape: tuple[int, ...], max_r: int | None = None, metric: DistanceMetric = "manhattan") -> dict[int, list[tuple[int, ...]]]:
    """
    Group periodic lattice displacements by distance.

    What is calculated:
    - Enumerates every displacement on a periodic lattice.
    - Converts each displacement into its shortest periodic step length.
    - Groups displacements into shells by Manhattan or Chebyshev distance.
    - Excludes the zero displacement, because that would be the site itself.

    Example for a 1D periodic chain of length 6:
        displacement 1 -> distance 1
        displacement 5 -> distance 1, because moving -1 is shorter
        displacement 3 -> distance 3

    Example:
        distance_shells((4,), metric="manhattan")
        # -> {
        #      1: [(1,), (3,)],
        #      2: [(2,)],
        #    }

    For a 2D lattice, each displacement is a tuple like `(dx, dy)`.
    """
    shells: dict[int, list[tuple[int, ...]]] = {}

    for displacement in _all_nonzero_displacements(shape):
        steps = [_periodic_distance(delta, length) for delta, length in zip(displacement, shape)]
        distance = _shell_distance(steps, metric)

        if distance == 0 or (max_r is not None and distance > max_r):
            continue

        shells.setdefault(int(distance), []).append(displacement)

    return dict(sorted(shells.items()))


def local_shell_targets(
    samples,
    shape: tuple[int, ...],
    max_r: int | None = None,
    metric: DistanceMetric = "manhattan",
) -> dict[int, np.ndarray]:
    """
    Build local correlation targets for every distance shell.

    Calculation:
        y_{a,i,r} = s_a(i) * mean_{j: d(i,j)=r} s_a(j)

    Meaning:
    - `a` indexes sampled spin configurations.
    - `i` indexes lattice sites.
    - `r` is the graph-distance shell.
    - `s_a(i)` is the spin at site `i` in sample `a`.
    - The shell mean averages spins around site `i` at distance `r`.

    These targets are later used to test whether local graph-distance-r
    information is linearly decodable from a node representation h_i^(b).

    Example:
        On a 1D ring, for distance r=1, each site has two neighbors:
        left and right. The target is:

            spin_at_site * average(left_neighbor_spin, right_neighbor_spin)

        If sample = [1, -1, 1, 1], then for site 0:
            neighbors are site 3 and site 1 -> [1, -1]
            shell average = 0
            target = 1 * 0 = 0
    """
    spin_samples = SpinSamples.from_samples(samples, shape)
    targets: dict[int, np.ndarray] = {}

    for shell in _distance_shell_objects(shape, max_r=max_r, metric=metric):
        shell_average = _shell_average(spin_samples.values, shell.displacements)
        targets[shell.distance] = (spin_samples.values * shell_average).reshape(spin_samples.n_samples, -1)

    return targets


def local_probe_baseline(samples, shape: tuple[int, ...], metric: DistanceMetric = "manhattan") -> np.ndarray:
    """
    Build simple baseline features for each `(sample, site)` pair.

    Returned columns:
    1. Site spin: `s_a(i)`
    2. Nearest-neighbor average around site `i`
    3. Global magnetization of sample `a`

    Calculation:
        baseline[a, i] = [
            s_a(i),
            mean nearest-neighbor spin around i,
            mean spin over the whole sample,
        ]

    The returned array is flattened over sample and site dimensions:

        shape == (n_samples * n_sites, 3)

    Example:
        For a 1D ring sample `[1, -1, 1, 1]`, site 0 has:
        - site spin = 1
        - nearest neighbors = site 3 and site 1 -> [1, -1], average = 0
        - global magnetization = (1 - 1 + 1 + 1) / 4 = 0.5

        Feature row for `(sample, site 0)`:
            [1, 0, 0.5]

    Note:
        `metric` is accepted for API compatibility with callers, but the current
        baseline always uses immediate nearest neighbors along each lattice axis.
    """
    spin_samples = SpinSamples.from_samples(samples, shape)
    flat_spins = spin_samples.flat
    nearest_neighbor_average = spin_samples.nearest_neighbor_average
    repeated_magnetization = np.repeat(spin_samples.magnetization, flat_spins.shape[1], axis=1)

    return np.stack([flat_spins.reshape(-1), nearest_neighbor_average.reshape(-1), repeated_magnetization.reshape(-1)], axis=1)


def _distance_shell_objects(shape: tuple[int, ...], max_r: int | None = None, metric: DistanceMetric = "manhattan") -> list[Shell]:
    return [Shell(distance=distance, displacements=displacements) for distance, displacements in distance_shells(shape, max_r=max_r, metric=metric).items()]


def _all_nonzero_displacements(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        tuple(int(delta) for delta in displacement)
        for displacement in itertools.product(*(range(axis_size) for axis_size in shape))
        if any(delta != 0 for delta in displacement)
    ]


def _periodic_distance(delta, length: int) -> int:
    delta = abs(int(delta))
    return min(delta, int(length) - delta)


def _shell_distance(steps: list[int], metric: DistanceMetric) -> int:
    return max(steps) if metric == "chebyshev" else sum(steps)


def _shell_average(spins: np.ndarray, displacements: list[tuple[int, ...]]) -> np.ndarray:
    shell_sum = np.zeros_like(spins, dtype=float)

    for displacement in displacements:
        shell_sum += _rolled_spins(spins, displacement)

    return shell_sum / float(len(displacements))


def _rolled_spins(spins: np.ndarray, displacement: tuple[int, ...]) -> np.ndarray:
    rolled = spins

    for axis, delta in enumerate(displacement):
        rolled = np.roll(rolled, -delta, axis=axis + 1)

    return rolled