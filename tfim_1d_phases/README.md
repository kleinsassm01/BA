# 1D Transverse Field Ising Model phases with nqs and exact

## Table of Contents
1. [Setup](#1-setup)
2. [Results](#2-results)
3. [The Model](#1-the-transverse-field-ising-model)
4. [Order Parameters](#8-order-parameters)

---

## 1. Setup
To setup the application, run `uv sync`. After that, there are two endpoints. From project root `uv run tfim-1d-train` runs the training, while `uv run tfim-1d-plot` can run the plotting separately without retraining everything. For installing `SciencePlots` visit https://github.com/garrettj403/SciencePlots

## 2. Results

The figure shows the ground-state phase diagram of the one-dimensional transverse-field Ising model (TFIM) at fixed transverse field $h = 1.0$ and system size $N = 10$, as a function of the coupling parameter $J$. The results are obtained using a neural quantum state (NQS) ansatz and compared with exact solutions.

Two order parameters are displayed: the squared magnetization $\langle m^2 \rangle$ and the squared staggered magnetization $\langle n^2 \rangle$. These quantities characterize different types of magnetic order. The plotted points correspond to converged NQS estimates, while the faint grey lines serve only as visual guides between discrete values of $J$.
![phase_diagram](./outputs/phase_diagram.png)
For large negative $J$, the system is in an antiferromagnetic phase, indicated by a large value of $\langle m^2 \rangle$ and a vanishing $\langle n^2 \rangle$. As $J$ approaches zero, both order parameters decrease due to increasing quantum fluctuations induced by the transverse field. Around $J \approx 0$, the system enters a paramagnetic phase, where both $\langle m^2 \rangle$ and $\langle n^2 \rangle$ are small.

For positive $J$, the system transitions into a ferromagnetic phase, where $\langle n^2 \rangle$ increases and eventually saturates, while $\langle m^2 \rangle$ approaches zero.

The lower-left panel shows the ground-state energy per site $E_0/N$. The NQS results closely match the exact finite-size solution and follow the thermodynamic limit prediction $N \to \infty$.

The lower-right panel displays the relative error of the NQS energy with respect to the exact finite-$N$ result. The error remains small across the entire parameter range and is minimized near the critical region, indicating robust performance even in regimes with strong quantum correlations.

![training_convergence](./outputs/training_convergence.png)

The training convergence plots show the optimization dynamics of the NQS for three representative points in the phase diagram.

Each panel displays the evolution of the variational energy per site $E_0/N$ during training. The NQS estimates are shown as discrete points, highlighting the stochastic nature of the optimization process. Two reference lines are included: the exact finite-size result for the system ($N = 10$, solid line) and the thermodynamic limit prediction ($N \to \infty$, dashed line).

In all three regimes, the NQS rapidly converges toward the exact ground-state energy within a relatively small number of training steps. The initial fluctuations reflect the stochastic sampling and parameter updates, but these oscillations diminish as training progresses and the model approaches the variational minimum.

![training_histories](./outputs/training_histories.png)

The training history plots show the evolution of the order parameters $\langle m^2 \rangle$ and $\langle n^2 \rangle$ during optimization for different values of the coupling $J$. Each curve corresponds to a separate training run, and the color encodes the value of $J$, ranging from antiferromagnetic (blue, $J < 0$) to ferromagnetic (red, $J > 0$) regimes.

At the beginning of training, both order parameters start from similar initial values, reflecting the untrained or weakly structured state of the variational wavefunction. As training progresses, the system rapidly organizes into the appropriate phase, and the order parameters evolve toward their characteristic values.

For negative $J$, $\langle m^2 \rangle$ quickly increases and saturates near unity, while $\langle n^2 \rangle$ is suppressed, indicating antiferromagnetic order. Conversely, for positive $J$, $\langle n^2 \rangle$ becomes dominant, signaling ferromagnetic ordering, while $\langle m^2 \rangle$ approaches zero. Near $J \approx 0$, both quantities remain small, consistent with a paramagnetic phase where no long-range order is present.

The convergence of the order parameters is typically fast, with most of the evolution occurring within the first few tens of training steps. After this initial phase, the values fluctuate slightly around their steady-state values due to stochastic sampling in the optimization procedure.

The smooth and consistent separation of trajectories across different $J$ shows that the system captures the underlying phase structure and corresponding order parameters throughout training.

---
## 3. The Transverse Field Ising Model

The one-dimensional Transverse Field Ising Model (TFIM) is one of the simplest quantum many-body systems that exhibits a quantum phase transition. It describes a chain of $N$
spin-$\frac{1}{2}$ particles with nearest-neighbor Ising interaction and a uniform transverse
magnetic field.

The Hamiltonian is:

$$
H = -J \sum_{i=1}^{N} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^{N} \sigma_i^x
$$

where:

- $\sigma_i^x, \sigma_i^y, \sigma_i^z$ are the Pauli matrices acting on site $i$,
- $J$ is the Ising coupling constant ($J > 0$: ferromagnetic, $J < 0$: antiferromagnetic),
- $h$ is the transverse field strength,
- periodic boundary conditions (PBC) are assumed: $\sigma_{N+1} \equiv \sigma_1$.

The Pauli matrices satisfy the algebra:

$$
[\sigma_i^\alpha, \sigma_j^\beta] = 2i\,\delta_{ij}\,\epsilon^{\alpha\beta\gamma}\,\sigma_i^\gamma, \qquad (\sigma_i^\alpha)^2 = \mathbb{1}
$$

Crucially, operators on different sites commute:
$[\sigma_i^\alpha, \sigma_j^\beta] = 0$ for $i \neq j$. This will be important when we
compare to fermionic operators, which anticommute.

---

## 4. Order Parameters

To distinguish the phases, we measure two order parameters:

### 4.1 Magnetization (Ferromagnetic Order)

$$
m = \frac{1}{N}\sum_i \sigma_i^z
$$

The squared magnetization detects ferromagnetic long-range order:

$$
\langle m^2 \rangle = \frac{1}{N^2}\sum_{i,j} \langle \sigma_i^z \sigma_j^z \rangle
$$

- $\langle m^2 \rangle > 0$: **Ferromagnetic phase** ($J > 0$, $|J| > h$)
- $\langle m^2 \rangle \to 0$: No uniform magnetic order

### 4.2 Néel (Staggered) Magnetization (Antiferromagnetic Order)

$$
n = \frac{1}{N}\sum_i (-1)^i \sigma_i^z
$$

$$
\langle n^2 \rangle = \frac{1}{N^2}\sum_{i,j} (-1)^{i+j} \langle \sigma_i^z \sigma_j^z \rangle
$$

- $\langle n^2 \rangle > 0$: **Antiferromagnetic phase** ($J < 0$, $|J| > h$)
- $\langle n^2 \rangle \to 0$: No staggered order

### 4.3 Phase Diagram Summary

| Regime | $J$ | Condition | $\langle m^2 \rangle$ | $\langle n^2 \rangle$ |
|--------|-----|-----------|----------------------|----------------------|
| Ferro  | $J > 0$ | $J > h$ | $> 0$ | $\approx 0$ |
| Para   | any | $\|J\| < h$ | $\approx 0$ | $\approx 0$ |
| Antiferro | $J < 0$ | $\|J\| > h$ | $\approx 0$ | $> 0$ |

---

## 5. Background Jordan-Wigner transformation
Details: https://theory.leeds.ac.uk/interaction-distance/applications/ising/map-to-free/
The 1D transverse-field Ising model (TFIM) can be mapped to a system of free fermions via the Jordan–Wigner transformation. The mapping from spin operators to spinless fermions is given by

$$
c_j = \left( \prod_{l<j} \sigma^z_l \right) \frac{\sigma^z_j + i \sigma^y_j}{2}.
$$

After performing a Fourier transform to momentum space, the Hamiltonian decouples into independent $2 \times 2$ blocks for each momentum $k$:

$$
H = \sum_k \epsilon(k)\,\left(\eta^\dagger_k \eta_k - \tfrac{1}{2}\right),
$$

where the single-particle dispersion (Bogoliubov quasiparticle energy) is

$$
\epsilon(k) = 2 \sqrt{J^2 + h^2 + 2Jh \cos(k)}.
$$

For a system of $N$ sites with periodic boundary conditions, the allowed momenta are

$$
k_n = \frac{2\pi n}{N}, \quad n = 0, 1, \dots, N-1.
$$

The ground-state energy is obtained by summing over all modes:

$$
E_0 = -\frac{1}{2} \sum_k \epsilon(k).
$$

In the thermodynamic limit ($N \to \infty$), this sum becomes an integral:

$$
\frac{E_0}{N} = -\frac{1}{2\pi} \int_0^{2\pi} \sqrt{J^2 + h^2 + 2Jh \cos(k)} \, dk.
$$

This integral can be expressed in terms of the complete elliptic integral of the second kind $E(m)$:

$$
\frac{E_0}{N} = -\frac{2}{\pi} \, \max(|J|, h)\, E(m),
$$

where

$$
m = \left( \frac{\min(|J|, h)}{\max(|J|, h)} \right)^2.
$$

The system undergoes a quantum phase transition at

$$
|J| = h,
$$

which separates the ordered (ferromagnetic or antiferromagnetic) phases from the paramagnetic phase.