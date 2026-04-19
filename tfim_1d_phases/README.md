# 1D Transverse Field Ising Model phases with nqs and exact

## Table of Contents
1. [Setup](#1-setup)
2. [Results](#2-results)
3. [The Transverse Field Ising Model](#3-the-transverse-field-ising-model)
4. [Order Parameters](#4-order-parameters)
5. [Background: Jordan-Wigner Transformation](#5-background-jordan-wigner-transformation)
6. [Finite-Size Analysis and the Quantum Phase Transition](#6-finite-size-analysis-and-the-quantum-phase-transition)
7. [Autocorrelation Time of the Monte Carlo Sampler](#7-autocorrelation-time-of-the-monte-carlo-sampler)

---

## 1. Setup
To setup the application, run `uv sync`. After that, there are four endpoints. From project root `uv run tfim-1d-train` runs the training for a single system size, while `uv run tfim-1d-plot` can run the plotting separately without retraining. The multi-$N$ counterparts are `uv run tfim-1d-train_N` and `uv run tfim-1d-plot_N`: they scan several system sizes $N$ over the same $J$ values, render overlay plots, and additionally estimate the Monte Carlo autocorrelation time at every point. For installing `SciencePlots` visit https://github.com/garrettj403/SciencePlots.

## 2. Results

The figure shows the ground-state phase diagram of the one-dimensional transverse-field Ising model (TFIM) at fixed transverse field $h = 1.0$ and system size $N = 4$, as a function of the coupling parameter $J$. The results are obtained using a neural quantum state (NQS) ansatz and compared with exact solutions.

Two order parameters are displayed: the squared magnetization $\langle m^2 \rangle$ and the squared staggered magnetization $\langle n^2 \rangle$. These quantities characterize different types of magnetic order. The plotted points correspond to converged NQS estimates, while the faint grey lines serve only as visual guides between discrete values of $J$. The dotted vertical lines at $J = \pm h$ mark the quantum critical points predicted by the exact Jordan-Wigner solution.

![phase_diagram](./outputs/phase_diagram.png)

For large negative $J$ the system is in a ferromagnetic phase, indicated by a large value of $\langle m^2 \rangle$ and a vanishing $\langle n^2 \rangle$. As $|J|$ decreases toward zero, both order parameters decrease due to increasing quantum fluctuations induced by the transverse field. Around $J \approx 0$ the system is paramagnetic, where both $\langle m^2 \rangle$ and $\langle n^2 \rangle$ are small. For large positive $J$ the system transitions into an antiferromagnetic phase, where $\langle n^2 \rangle$ saturates near unity while $\langle m^2 \rangle$ approaches zero. The phase diagram is perfectly symmetric under $J \to -J$ because the Ising coupling enters the Hamiltonian quadratically through $\sigma^z_i \sigma^z_j$, and the transformation $\sigma^z_i \to (-1)^i \sigma^z_i$ on alternating sites maps the ferromagnet onto the antiferromagnet.

The lower-left panel shows the ground-state energy per site $E_0/N$. The NQS results closely match the exact finite-size solution and follow the thermodynamic limit prediction $N \to \infty$. The lower-right panel displays the relative error of the NQS energy with respect to the exact finite-$N$ result. The error remains small across the entire parameter range, typically between $10^{-5}$ and $10^{-3}$, and is minimized near the critical region, indicating robust performance even in regimes with strong quantum correlations.

![training_convergence](./outputs/training_convergence.png)

The training convergence plots show the optimization dynamics of the NQS for three representative points in the phase diagram. Each panel displays the evolution of the variational energy per site $E_0/N$ during training. The NQS estimates are shown as discrete points, highlighting the stochastic nature of the optimization process. Two reference lines are included: the exact finite-size result for the system ($N = 4$, solid line) and the thermodynamic limit prediction ($N \to \infty$, dashed line).

In all three regimes the NQS rapidly converges toward the exact ground-state energy within a relatively small number of training steps. The initial fluctuations reflect the stochastic sampling and parameter updates, but these oscillations diminish as training progresses and the model approaches the variational minimum. At $J = 0$ the chain decouples into independent spins in a transverse field with exact energy $E_0/N = -h$, which our NQS reproduces to better than $10^{-4}$ of a percent.

![training_histories](./outputs/training_histories.png)

The training history plots show the evolution of the order parameters $\langle m^2 \rangle$ and $\langle n^2 \rangle$ during optimization for different values of the coupling $J$. Each curve corresponds to a separate training run, and the color encodes the value of $J$, ranging from ferromagnetic (blue, $J < 0$) to antiferromagnetic (red, $J > 0$) regimes.

At the beginning of training both order parameters start from similar initial values, reflecting the untrained or weakly structured state of the variational wavefunction. As training progresses the system rapidly organizes into the appropriate phase, and the order parameters evolve toward their characteristic values. For negative $J$, $\langle m^2 \rangle$ quickly increases and saturates near unity while $\langle n^2 \rangle$ is suppressed, indicating ferromagnetic order. Conversely, for positive $J$, $\langle n^2 \rangle$ becomes dominant, signalling antiferromagnetic ordering, while $\langle m^2 \rangle$ approaches zero. Near $J \approx 0$ both quantities remain small, consistent with a paramagnetic phase where no long-range order is present.

The convergence of the order parameters is typically fast, with most of the evolution occurring within the first few tens of training steps. After this initial phase the values fluctuate slightly around their steady-state values due to stochastic sampling in the optimization procedure. The smooth and consistent separation of trajectories across different $J$ shows that the system captures the underlying phase structure and corresponding order parameters throughout training.

---
## 3. The Transverse Field Ising Model

The one-dimensional Transverse Field Ising Model (TFIM) is one of the simplest quantum many-body systems that exhibits a quantum phase transition. It describes a chain of $N$ spin-$\tfrac{1}{2}$ particles with nearest-neighbor Ising interaction and a uniform transverse magnetic field.

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

Crucially, operators on different sites commute: $[\sigma_i^\alpha, \sigma_j^\beta] = 0$ for $i \neq j$. This will be important during comparison to fermionic operators, which anticommute.

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
| Ferro  | $J > 0$ | $\|J\| > h$ | $> 0$ | $\approx 0$ |
| Para   | any | $\|J\| < h$ | $\approx 0$ | $\approx 0$ |
| Antiferro | $J < 0$ | $\|J\| > h$ | $\approx 0$ | $> 0$ |

### 4.4 Higher Moments

In addition to $\langle m^2 \rangle$ and $\langle n^2 \rangle$, the fourth moment of the uniform magnetization is also recordered,

$$
\langle m^4 \rangle = \left\langle \left(\tfrac{1}{N}\sum_i \sigma_i^z\right)^{\!4}\,\right\rangle,
$$

which is needed to form the dimensionless **Binder cumulant**

$$
U_4 \;=\; 1 \;-\; \frac{\langle m^4 \rangle}{3\,\langle m^2 \rangle^{2}}.
$$

The Binder cumulant is the standard diagnostic for a continuous (second-order) phase transition: curves for different system sizes $N$ all cross at a common value at the critical point, so the crossing is itself a size-independent estimator of $J_c$. Deep in an Ising-ordered phase the magnetization distribution is bimodal and $U_4 \to \tfrac{2}{3}$; in the paramagnet the distribution becomes Gaussian and $U_4 \to 0$.

Because $\sigma_i^z$ is diagonal in the computational basis, both $\langle m^2 \rangle$ and $\langle m^4 \rangle$ can be computed directly from Monte Carlo samples without building an $O(N^4)$ operator, which keeps the calculation tractable at the largest system sizes considered.

---

## 5. Background Jordan-Wigner transformation
Details: https://theory.leeds.ac.uk/interaction-distance/applications/ising/map-to-free/
The 1D transverse-field Ising model (TFIM) can be mapped to a system of free fermions via the Jordan–Wigner transformation. The mapping from spin operators to spinless fermions is given by

$$
c_j = \left( \prod_{l=1}^{j-1} \sigma^z_l \right) \frac{\sigma^z_j + i \sigma^y_j}{2}.
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

---

## 6. Finite-Size Analysis and the Quantum Phase Transition

Near the quantum critical points $|J| = h$ the correlation length diverges and finite-size effects become dominant. To resolve the transition in detail, the same sweep is repeated for several system sizes $N \in \{4, 8, 16, 32, 64\}$, and the resulting curves are overlaid in a single figure.

![multi_N_overlay](./outputs_multiN/multi_N_overlay.png)

The upper panel shows the order parameters $\langle m^2 \rangle$ (solid) and $\langle n^2 \rangle$ (dashed) as functions of $J$ for all five system sizes. For $|J| \gg h$ the curves are essentially size-independent: the two ordered phases are insensitive to $N$ once the system is deep in the relevant basin of attraction. The behaviour close to $|J| = h$ is markedly different: as $N$ grows, the crossover region narrows and the curves approach a step function, which is the expected signature of a continuous phase transition with a correlation length that diverges in the thermodynamic limit.

The lower-left panel confirms the convergence of the ground-state energy density $E_0/N$ toward the exact $N \to \infty$ elliptic-integral curve. The cusps at $J = \pm h$ become increasingly pronounced with $N$, reflecting the non-analyticity that develops in the thermodynamic limit and defines the quantum phase transition. The lower-right panel displays the relative accuracy of the NQS ground-state energy with respect to the finite-$N$ exact diagonalisation; errors remain in the $10^{-5}$ to $10^{-1}\%$ range across the full sweep and all system sizes, and are typically largest in the critical region where the ground state has the most entanglement and the variational ansatz has to work hardest.

### 6.1 Critical Zoom and Energy Curvature

To sharpen the picture we zoom in on a window of halfwidth $0.4$ around each critical point and additionally compute the second derivative of the ground-state energy with respect to $J$, smoothed with a Savitzky-Golay filter to remove the noise amplification inherent in finite differences.

![critical_zoom](./outputs_multiN/critical_zoom.png)

Both left panels show the expected sharpening of the order parameters: the $N = 4$ curve changes smoothly over the full window, while the $N = 64$ curve has already developed a near-discontinuous drop from $\langle m^2 \rangle \approx 0.7$ to $\langle m^2 \rangle \approx 0.05$ within $|J - J_c| \lesssim 0.1$.

The right panel shows $-d^2 E_0/dJ^2$ as a function of $J$ for each $N$. For a second-order transition the ground-state energy density is continuous at $J_c$ but its second derivative develops a divergent peak in the thermodynamic limit, scaling as $|J - J_c|^{-\alpha}$ with $\alpha \geq 0$ for the 2D-Ising universality class to which this model belongs. The numerical curves show exactly this behaviour: the peak height grows monotonically with $N$ and its location converges towards $J_c = 1$. Importantly, $d E_0/dJ$ remains finite and continuous at $J_c$, which rules out a first-order transition (which would produce a finite jump in the first derivative and a $\delta$-function in the second derivative).

### 6.2 Binder Cumulant

The cleanest finite-size-scaling diagnostic is the Binder cumulant $U_4$ introduced in Section 4.4. The figure below shows $U_4$ against $J$ for all system sizes.

![binder_cumulant](./outputs_multiN/binder_cumulant.png)

The left panel covers the restricted range $J \in [-2, 2]$ and the right panel zooms on the ferromagnetic critical point at $J = -1$. Because $U_4$ is constructed from the uniform magnetization $m$, it is naturally suited to the ferromagnetic transition; at the antiferromagnetic transition the corresponding quantity would be the staggered Binder cumulant built from $\langle n^4 \rangle$, and $U_4$ of the uniform $m$ is correspondingly noisier for $J > 0$.

Deep in the ferromagnetic phase ($J \ll -h$) all curves approach the value $U_4 = \tfrac{2}{3}$ expected for a fully ordered Ising state, as marked by the upper reference line. In the paramagnetic region the Gaussian limit $U_4 \to 0$ is approached from above. The curves for different $N$ visibly cross in the neighbourhood of $J = -1$: for $N = 4$ the curve decays earliest (the ordered plateau is shortest), while for $N = 64$ the plateau extends furthest into the critical region before dropping sharply. The size-independent value at which the curves cross provides a finite-size-scaling estimate of $J_c^{\text{ferro}} \approx -1$, consistent with the analytical prediction $|J_c| = h = 1$.

---

## 7. Autocorrelation Time of the Monte Carlo Sampler

Every Monte Carlo estimate in this project is based on samples drawn from a Markov chain generated by the `MetropolisLocal` sampler of NetKet. Successive samples along a chain are not independent; they are correlated over a characteristic timescale known as the integrated autocorrelation time $\tau_\text{int}$. The effective number of independent samples in a chain of length $T$ is

$$
T_\text{eff} \;\approx\; \frac{T}{2\,\tau_\text{int}},
$$

and the statistical uncertainty of a sample-mean estimator for an observable $\mathcal{O}$ scales as

$$
\sigma_{\bar{\mathcal{O}}} \;=\; \sqrt{\frac{2\,\tau_\text{int}}{T}}\,\sigma_\mathcal{O},
$$

where $\sigma_\mathcal{O}$ is the population standard deviation. A large $\tau_\text{int}$ therefore degrades the effective sample size and inflates the statistical error of every measured quantity, including the energy gradient that drives the NQS optimization.

The autocorrelation time is estimated via the standard automated-windowing procedure of Sokal. For a time series $x_1, x_2, \dots, x_T$ first compute the normalized autocovariance

$$
\rho(t) \;=\; \frac{C(t)}{C(0)},
\qquad
C(t) \;=\; \frac{1}{T}\sum_{k=1}^{T-t} (x_k - \bar{x})(x_{k+t} - \bar{x}),
$$

and then define the windowed estimator

$$
\tau_\text{int}(W) \;=\; \tfrac{1}{2} \;+\; \sum_{t=1}^{W} \rho(t).
$$

The Sokal criterion selects the smallest window $W^{\star}$ such that $W^{\star} \geq c\,\tau_\text{int}(W^{\star})$, with $c$ between $4$ and $10$; use $c = 5$. The final estimate is $\tau_\text{int} \equiv \tau_\text{int}(W^{\star})$.

Two complementary views of the autocorrelation are produced. First, NetKet reports an online estimate $\tau_\text{corr}$ of the energy autocorrelation alongside every expectation-value evaluation during training. Recording this quantity at each logging step tells us whether the sampler is well-mixed as the variational parameters evolve.

![tau_corr_vs_step](./outputs_multiN/tau_corr_vs_step.png)

Each subplot corresponds to one system size, and the curves are coloured by the value of $J$, ranging from ferromagnetic (blue, $J < 0$) through paramagnetic (grey, $J \approx 0$) to antiferromagnetic (red, $J > 0$). A common pattern emerges across all $N$: at the very first training steps the sampler has to adapt to a rapidly changing variational state and $\tau_\text{corr}$ transiently rises, but by step $\sim 50$ the autocorrelation has largely settled to a small value. There is no strong systematic trend with $N$ on the scale of the training loop, but the points closest to the critical couplings $|J| = h$ (roughly the purple/red extremes of the colour range near $J = \pm 1$) consistently show the highest plateau, which is a weak in-training hint of the critical slowing down that resolve more cleanly in the dedicated post-training analysis.

Second, once training has converged at a given $(N, J)$ point draw a long dedicated Markov chain from the sampler and compute the integrated autocorrelation time of the local-energy series by applying the Sokal windowing procedure described above.

![tau_int_vs_J](./outputs_multiN/tau_int_vs_J.png)

The left panel shows $\tau_\text{int}(J)$ on a logarithmic scale for every system size, with the dotted vertical lines marking the quantum critical points. Deep in the ordered or paramagnetic regions the sampler decorrelates within less than one Metropolis sweep, which is the ideal regime for Monte Carlo estimation. Narrow peaks develop near $J = \pm 1$ at all system sizes, where the chain requires several MC steps to decorrelate. This is the phenomenon of **critical slowing down**: near a continuous phase transition the correlation length of the physical state diverges, which drives the MCMC dynamics towards a long-tailed transition-time distribution. Physically, the variational wavefunction near criticality is spread out over a large, rugged basin of the Hilbert space, and the local spin-flip updates of `MetropolisLocal` are less effective at moving between typical configurations.

The right panel shows the normalized autocorrelation function $\rho(t) = C(t)/C(0)$ at the antiferromagnetic critical point $J \approx +1$ for every system size. The curves fall to zero within a handful of MC steps, consistent with the small $\tau_\text{int}$ values reported in the left panel. For a qualitatively stronger signature of critical slowing down at larger $N$ one would need to approach significantly larger system sizes (which is beyond the scope of this project), where the local sampler is expected to struggle more noticeably with the long-range correlations of the critical state.

Taken together, the three panels in the autocorrelation analysis confirm that `MetropolisLocal` mixes very efficiently for the 1D TFIM at the sizes considered here, with $\tau_\text{int}$ never exceeding order-one MC steps except in narrow windows around the critical couplings. This justifies the use of the reported NQS estimates as essentially independent Monte Carlo averages.
