from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import netket as nk
import numpy as np
from netket.operator.spin import sigmax, sigmaz

from ..config import PhysicsConfig


@dataclass(frozen=True)
class Lattice:
    """Periodic hypercubic lattice helper.

    Coordinates are mapped to flat spin indices in row-major order.

    Example:
        shape = (2, 3)

        coordinates:
            (0, 0) -> 0
            (0, 1) -> 1
            (0, 2) -> 2
            (1, 0) -> 3
            (1, 1) -> 4
            (1, 2) -> 5

        Periodic wrapping means:
            (0, 3) maps to (0, 0)
            (2, 1) maps to (0, 1)
    """

    shape: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.shape)

    @property
    def n_sites(self) -> int:
        n_sites = 1
        for axis_size in self.shape:
            n_sites *= int(axis_size)
        return n_sites

    def site_index(self, coordinates: tuple[int, ...]) -> int:
        index = 0
        stride = 1

        for coordinate, axis_size in zip(reversed(coordinates), reversed(self.shape)):
            index += (int(coordinate) % int(axis_size)) * stride
            stride *= int(axis_size)

        return index

    def nearest_neighbor_edges(self) -> list[tuple[int, int]]:
        """Return unique periodic nearest-neighbor edges.

        For each lattice site, we connect the site to its +1 neighbor along each
        axis. Edges are sorted before insertion, so `(i, j)` and `(j, i)` are
        treated as the same undirected edge.

        Example:
            On a 1D ring of length 4, the nearest-neighbor edges are:
                (0, 1), (1, 2), (2, 3), (0, 3)

            The final edge `(0, 3)` is the periodic wraparound connection.
        """
        edges: set[tuple[int, int]] = set()

        for coordinates in self._all_coordinates():
            site = self.site_index(coordinates)
            for axis in range(self.dimension):
                neighbor_coordinates = list(coordinates)
                neighbor_coordinates[axis] += 1
                neighbor = self.site_index(tuple(neighbor_coordinates))

                if site != neighbor:
                    edges.add(_sorted_edge(site, neighbor))

        return sorted(edges)

    def diagonal_edges_2d(self) -> list[tuple[int, int]]:
        """Return unique periodic diagonal edges for a 2D lattice.

        Diagonal edges connect `(x, y)` to:
            `(x + 1, y + 1)` and `(x + 1, y - 1)`

        These are used for the optional next-nearest-neighbor Ising term `J2`.
        For non-2D lattices, no diagonal edges are defined.
        """
        if self.dimension != 2:
            return []

        edges: set[tuple[int, int]] = set()
        x_size, y_size = self.shape

        for x in range(x_size):
            for y in range(y_size):
                site = self.site_index((x, y))
                for dx, dy in ((1, 1), (1, -1)):
                    diagonal_neighbor = self.site_index((x + dx, y + dy))

                    if site != diagonal_neighbor:
                        edges.add(_sorted_edge(site, diagonal_neighbor))

        return sorted(edges)

    def _all_coordinates(self):
        return itertools.product(*(range(axis_size) for axis_size in self.shape))


@dataclass(frozen=True)
class ExactGroundState:
    """Exact diagonalization result for small Hilbert spaces."""

    energy: float
    wavefunction: np.ndarray


def lattice_edges(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    return Lattice(shape).nearest_neighbor_edges()


def diagonal_edges_2d(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    return Lattice(shape).diagonal_edges_2d()


def build_hilbert(problem: PhysicsConfig):
    """Build the spin-1/2 Hilbert space for `problem.n_spins` sites."""
    return nk.hilbert.Spin(s=0.5, N=problem.n_spins)


def build_hamiltonian(problem: PhysicsConfig):
    """Build a stoquastic transverse-field Ising Hamiltonian.

    Calculation:
        H = -J  * sum_<ij>  Z_i Z_j
            +J2 * sum_<<ij>> Z_i Z_j
            -h  * sum_i     X_i

    Meaning:
    - `-h X_i` is the transverse-field spin-flip term.
    - `-J Z_i Z_j` couples nearest-neighbor lattice sites.
    - `+J2 Z_i Z_j` optionally couples 2D diagonal neighbors.
    - Periodic boundary conditions are used for both nearest and diagonal edges.

    Example:
        For a 1D ring of length 4, the nearest-neighbor Ising terms are:
            Z_0 Z_1, Z_1 Z_2, Z_2 Z_3, Z_0 Z_3

        For a 2D lattice, `J2` additionally adds diagonal couplings when
        `problem.J2 != 0`.
    """
    hilbert_space = build_hilbert(problem)
    lattice = Lattice(problem.shape)

    hamiltonian = sum(-problem.h * sigmax(hilbert_space, site) for site in range(problem.n_spins))

    for left_site, right_site in lattice.nearest_neighbor_edges():
        hamiltonian += -problem.J * (sigmaz(hilbert_space, left_site) @ sigmaz(hilbert_space, right_site))

    if abs(problem.J2) > 0 and lattice.dimension == 2:
        for left_site, right_site in lattice.diagonal_edges_2d():
            hamiltonian += problem.J2 * (sigmaz(hilbert_space, left_site) @ sigmaz(hilbert_space, right_site))

    return hamiltonian, hilbert_space


def exact_ground_state(hamiltonian: Any, hilbert_space: Any) -> tuple[float, np.ndarray]:
    """Compute the lowest-energy eigenstate by exact diagonalization.

    This is intended for small systems only, because exact diagonalization scales
    exponentially with the number of spins.

    Returns:
        `(energy, wavefunction)`, where `wavefunction` is the ground-state vector
        indexed in NetKet's Hilbert-space basis order.
    """
    eigenvalues, eigenvectors = nk.exact.lanczos_ed(hamiltonian, k=1, compute_eigenvectors=True)
    ground_state = ExactGroundState(energy=float(np.real(eigenvalues[0])), wavefunction=np.asarray(eigenvectors[:, 0]).reshape(-1))
    return ground_state.energy, ground_state.wavefunction


def born_sample_exact(hilbert_space: Any, wavefunction: Any, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample spin configurations from the exact Born distribution.

    Calculation:
        p(s) = |psi(s)|^2 / sum_s |psi(s)|^2

    Steps:
    - Enumerate all basis states from the Hilbert space.
    - Convert the wavefunction into probabilities using squared amplitudes.
    - Draw `n_samples` basis states according to those probabilities.

    Example:
        If one basis state has probability 0.7 and another has probability 0.3,
        this function will sample the first state about 70% of the time.
    """
    states = np.asarray(hilbert_space.all_states(), dtype=np.float64)
    probabilities = _born_probabilities(wavefunction)
    sampled_indices = rng.choice(states.shape[0], size=int(n_samples), replace=True, p=probabilities)
    return states[sampled_indices]


def logpsi_exact(hilbert_space: Any, wavefunction: Any, samples: Any) -> np.ndarray:
    """Return exact log-amplitudes for selected spin samples.

    Calculation:
        logpsi(s) = log(|psi(s)| + 1e-300)

    The tiny constant avoids `log(0)` for numerically zero amplitudes.

    Steps:
    - Convert spin configurations to Hilbert-space basis indices.
    - Look up the corresponding exact wavefunction amplitudes.
    - Return the log absolute amplitude.
    """
    sample_array = np.asarray(samples)
    basis_indices = np.asarray(hilbert_space.states_to_numbers(sample_array), dtype=np.int64)
    amplitudes = np.abs(np.asarray(wavefunction).reshape(-1)[basis_indices])
    return np.log(amplitudes + 1.0e-300)


def _site(shape: tuple[int, ...], coords: tuple[int, ...]) -> int:
    """Compatibility wrapper for older callers."""
    return Lattice(shape).site_index(coords)


def _born_probabilities(wavefunction: Any) -> np.ndarray:
    probabilities = np.abs(np.asarray(wavefunction).reshape(-1)) ** 2
    return probabilities / probabilities.sum()


def _sorted_edge(left_site: int, right_site: int) -> tuple[int, int]:
    return tuple(sorted((int(left_site), int(right_site))))