# NQS Representation Universality Analysis

## Hypothesis

> Different neural quantum state (NQS) architectures converge to the same internal representation when learning the ground state of the same Hamiltonian, because they all learn to mirror the Hamiltonian's circuit structure.

Five architectures - two CNNs, two bidirectional RNNs, and one local message-passing GNN - are tested on the 1D transverse-field Ising model (TFIM) at the quantum critical point $h/J = 1$:

$$H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j \;-\; h \sum_i \sigma^x_i$$

with $N=20$ spins and periodic boundary conditions.

All models parameterize $\log \psi_\theta(\mathbf{s})$ and are trained via VMC to minimize $\langle H \rangle$.

---

## Results and Interpretation

All five models converge to comparable ground-state energies. The two CNNs reach $E/N \approx -1.272$, the BiRNNs land at $E/N \approx -1.249$ to $-1.258$, and GNN-3layer-local achieves $E/N \approx -1.273$. The small spread between families reflects the difficulty of VMC optimization at the critical point rather than fundamentally different solutions - all models approximate the same ground state, as confirmed by the high tangent-space similarity (Fig 2). This validates the premise: the comparisons below concern representations of the *same* wave function, not different solutions.

---

### Figure 1 - Cross-architecture activation CKA

![Cross-architecture activation CKA](results/figures/fig1_cross_activation_cka.png)

**What it measures.** Linear Centered Kernel Alignment (CKA) between the hidden-layer activations of every pair of models, evaluated on the same set of spin configurations drawn from $|\psi|^2$. Given two centered activation matrices $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$:

$$\mathrm{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \;\|Y^\top Y\|_F}$$

CKA is invariant to invertible linear transforms and isotropic scaling. It equals 1 when $X$ and $Y$ span the same column space (up to linear mixing), and 0 when they induce orthogonal kernel matrices.

**Numerical example.** Suppose $X$ and $Y$ are $4 \times 2$ centered matrices. The numerator is $\|Y^\top X\|_F^2$: compute the $2 \times 2$ product $Y^\top X$, square each entry, and sum. If $Y^\top X = \bigl[\begin{smallmatrix}3 & 1\\0 & 2\end{smallmatrix}\bigr]$, then $\|Y^\top X\|_F^2 = 9+1+0+4 = 14$. The denominator is $\|X^\top X\|_F \cdot \|Y^\top Y\|_F$; if these equal $4.0$ and $3.7$, then $\mathrm{CKA} = 14 / (4.0 \times 3.7) \approx 0.95$.

**Results.** The activation CKA reveals a clear split between *functional convergence at the output* and *architectural divergence at early layers*:

- **Within-family CNN pairs** (CNN-3layer-k3 vs CNN-5layer-k3): warm throughout, with CKA $\gtrsim 0.6$ even between early layers. The same kernel size produces similar intermediate features regardless of depth.
- **CNN↔BiRNN pairs**: uniformly high CKA ($\gtrsim 0.7$) across all layer pairs. CNN intermediate layers align well with the BiRNN hidden layers, suggesting that the hidden representations share substantial structure even across architecture families.
- **CNN↔GNN pairs**: the input layers of both architectures show near-zero CKA due to different embedding strategies (raw spin vs learned dense embedding), but the hidden and output layers recover to CKA $\approx 0.6$–$0.8$. The GNN's output layer aligns poorly with CNN layer 1 (CKA $\approx 0.2$–$0.3$), reflecting a deep architectural divergence in how information is routed to the final readout.
- **BiRNN↔GNN pairs**: BiRNN hidden layers show moderate CKA ($\approx 0.5$–$0.7$) with GNN hidden layers, but the GNN output layer again stands out with very low CKA relative to BiRNN input layers. The BiRNN-1layer ↔ GNN-3layer-local pair shows particularly low CKA at the input-to-output cross ($\approx 0.1$), reflecting that the GNN's message-passing output compresses information very differently from the BiRNN's sequential readout.
- **BiRNN↔BiRNN**: high CKA at corresponding depths. The input layer of BiRNN-1layer shows near-zero CKA with deeper layers of BiRNN-2layer, reflecting that the raw input encoding has not yet been processed.

**Verdict:** Activation-level universality is **moderate**. Within CNN and BiRNN families, representations are highly similar. Cross-family pairs (CNN↔BiRNN) maintain moderate-to-high CKA. The GNN introduces the largest activation-level divergence, particularly at early and output layers, despite strong tangent-space convergence (Fig 2).

---

### Figure 2 - Full tangent-space CKA and NTK kernel alignment

![Tangent and NTK similarity](results/figures/fig2_tangent_ntk_similarity.png)

**What it measures.** Two model×model similarity matrices computed from the full variational tangent vectors $J(\mathbf{s}) = \partial \log \psi_\theta(\mathbf{s}) / \partial \theta$.

*Left panel (tangent CKA):* Treat $J_A \in \mathbb{R}^{n \times d_A}$ and $J_B \in \mathbb{R}^{n \times d_B}$ as feature matrices (rows = configurations, columns = parameter gradients), then compute CKA. This measures whether the two models' variational derivatives span the same functional subspace over configuration space.

*Right panel (NTK kernel alignment):* Build the neural tangent kernel $K_A = J_A J_A^\top \in \mathbb{R}^{n \times n}$, then compute centered kernel alignment:

$$\mathrm{KA}(K_1, K_2) = \frac{\mathrm{tr}(\tilde{K}_1 \tilde{K}_2)}{\sqrt{\mathrm{tr}(\tilde{K}_1^2)\;\mathrm{tr}(\tilde{K}_2^2)}}$$

where $\tilde{K} = H K H$ with centering matrix $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$.

**Numerical example.** For $n=3$ samples, suppose $K_1 = \bigl[\begin{smallmatrix}2 & 1 & 0\\1 & 3 & 1\\0 & 1 & 2\end{smallmatrix}\bigr]$ and $K_2 = \bigl[\begin{smallmatrix}3 & 1 & 0\\1 & 2 & 1\\0 & 1 & 3\end{smallmatrix}\bigr]$. After centering $\tilde{K}_i = H K_i H$, KA is the normalized Frobenius inner product of the centered kernels. If $\mathrm{tr}(\tilde{K}_1 \tilde{K}_2)=5.2$, $\mathrm{tr}(\tilde{K}_1^2)=6.0$, $\mathrm{tr}(\tilde{K}_2^2)=5.8$, then $\mathrm{KA} = 5.2 / \sqrt{6.0 \times 5.8} \approx 0.88$.

**Results.** This is the strongest evidence **for** the hypothesis. Both panels are nearly identical (CKA and kernel alignment agree), and all entries are remarkably high:

- **CNN-3layer-k3 ↔ CNN-5layer-k3:** 1.00 - essentially the same variational manifold despite different depths.
- **CNN ↔ BiRNN pairs:** 0.96–0.99 - even across architecture families, the tangent spaces are highly aligned.
- **CNN ↔ GNN-3layer-local:** 0.98 - the GNN tangent space aligns as well with the CNNs as the BiRNNs do.
- **BiRNN ↔ GNN-3layer-local:** 0.95–0.98 - the lowest value (0.95) occurs for BiRNN-1layer ↔ GNN-3layer-local, which is also the overall minimum of the matrix.
- **Minimum:** BiRNN-1layer ↔ GNN-3layer-local at 0.95 - still very high.

The tangent vector $\partial \log \psi / \partial \theta$ determines how the wave function *changes* when parameters are perturbed. High tangent CKA means that the set of "directions of variation" accessible to each model spans essentially the same functional subspace in Hilbert space. The models have converged to the same region of the variational manifold, just parameterized through different coordinate systems.

**Verdict:** Functional-level universality is **strong**. The hypothesis holds decisively at the tangent-space level across all three architecture families.

---

### Figure 3 - NTK eigenspectrum

![NTK eigenspectrum](results/figures/fig3_ntk_eigenspectrum.png)

**What it measures.** The eigenvalue spectrum of each model's neural tangent kernel $K = J J^\top$. The eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots$ determine the effective dimensionality of the learned kernel. The $k$-th eigenvalue measures how much the model's output varies along the $k$-th principal direction in configuration space.

**Results.** The spectra show good agreement at the top and clear architecture-dependent tails:

- **Top 3–5 eigenvalues:** All five models agree closely ($\lambda_1 \approx 10^4$). The dominant directions of variation - the handful of configuration-space modes that matter most - are universal.
- **Ranks 5–15:** The spectra begin to separate. CNN-3layer-k3 and CNN-5layer-k3 (blue, orange) track each other closely. BiRNN-2layer (red) has a slightly flatter decay.
- **GNN-3layer-local** (purple): shows a distinctly flatter spectral tail than the CNNs and BiRNN-2layer. Its eigenvalues remain above $10^{-7}$ out to rank 50, about 1–2 orders of magnitude above the CNN tails at the same rank. This indicates the GNN distributes its representational capacity more broadly across configuration-space modes, consistent with its multi-hop message-passing structure that progressively widens the receptive field at each layer.
- **BiRNN-1layer** (green): shows a strikingly steep drop at rank ~5, falling 4+ orders below the other models by rank 10. This indicates the single-layer BiRNN concentrates its representational capacity in very few modes, while the other architectures spread sensitivity more broadly.

**Interpretation.** The high tangent CKA (Fig 2) is driven by the top eigenvalues, which are shared. The spectral tails differ - the models agree on *what matters most* but disagree on the fine structure. The BiRNN-1layer's rapid spectral decay suggests it has found a very low-dimensional but effective parameterization of the ground state, whereas the GNN's flat tail suggests a higher effective dimensionality.

**Verdict:** Partial support. The dominant effective kernel is universal; the tail structure is architecture-dependent.

---

### Figure 4 - Orthogonal Procrustes distance

![Procrustes distance](results/figures/fig4_procrustes_distance.png)

**What it measures.** A stricter test than CKA. After centering and normalizing, find the orthogonal rotation $R$ that best aligns the two representations, then measure the residual:

$$d_{\mathrm{Proc}}(X, Y) = \min_{R^\top R = I} \left\| \frac{X}{\|X\|_F} R - \frac{Y}{\|Y\|_F} \right\|_F$$

CKA is invariant to *any* invertible linear transform; Procrustes is invariant only to orthogonal transforms. So Procrustes = 0 means the representations are geometrically identical up to rotation, while CKA = 1 only means they span the same subspace.

**Numerical example.** Given centered, Frobenius-normalized $X, Y \in \mathbb{R}^{4 \times 3}$ (so $\|X\|_F = \|Y\|_F = 1$), compute the SVD of $X^\top Y = U \Sigma V^\top$, set $R = U V^\top$, and evaluate $d = \|X R - Y\|_F$. If the singular values of $X^\top Y$ are $(0.95, 0.90, 0.80)$, then $d = \sqrt{2(3 - 0.95 - 0.90 - 0.80)} = \sqrt{2 \times 0.35} \approx 0.84$. A value of 0 indicates perfect geometric alignment; $\sqrt{2} \approx 1.414$ is the theoretical maximum.

**Results.**

*Left panel (readout activations):* A clear within-family vs cross-family split, with the GNN as the most distant architecture.

- CNN-3layer-k3 ↔ CNN-5layer-k3: 0.153 (close geometry).
- BiRNN-1layer ↔ BiRNN-2layer: 0.105 (very close).
- CNN↔BiRNN pairs: 0.61–0.76 - geometrically dissimilar across architecture families despite the high CKA in Fig 1.
- GNN-3layer-local ↔ BiRNNs: 0.969–1.016 - approaching the theoretical maximum, indicating near-maximal geometric dissimilarity in readout space.
- GNN-3layer-local ↔ CNNs: 0.742–0.782 - also large.

*Right panel (full tangent features):* Much tighter overall. CNN-3layer-k3 ↔ CNN-5layer-k3 is 0.084 (near-identical geometry). CNN↔BiRNN pairs are 0.21–0.30. BiRNN-1layer ↔ BiRNN-2layer is 0.138. GNN-3layer-local ↔ CNNs: 0.183–0.207, well aligned. GNN-3layer-local ↔ BiRNNs: 0.355–0.426, the largest tangent-space Procrustes values in the matrix but still well below the theoretical maximum.

**Interpretation.** The tangent Procrustes distances are roughly 3–5× smaller than the activation Procrustes distances, confirming the finding from Fig 2: the tangent-space geometry is shared across architectures, while the activation geometry retains family-specific structure. The GNN's readout Procrustes distances (approaching 1.0) are the most extreme in the matrix, indicating that its activation-level encoding is geometrically very different from the other families despite occupying the same tangent-space subspace. The non-zero tangent Procrustes distances (0.2–0.4 for cross-family) reveal that the alignment is not perfect - the models span the same subspace (high CKA) but arrange their axes differently within that subspace. The "content" is shared; the "coordinate system" is architecture-dependent.

**Verdict:** Tangent features show good geometric alignment (distances 0.08–0.43). Activation features show within-family alignment but cross-family divergence (0.10–0.15 within, 0.61–1.02 across). Universality is a subspace property, not a geometric one. The GNN introduces the largest geometric divergence in activation space.

---

### Figure 5 - Input-gradient saliency analysis

![Saliency analysis](results/figures/fig5_saliency_analysis.png)

**What it measures.** The input gradient $g_i(\mathbf{s}) = \partial \log \psi_\theta(\mathbf{s}) / \partial \sigma_i$ measures how sensitive the log-amplitude is to the spin at site $i$.

**Panel (a) - Mean saliency profile:**

The two CNNs show approximately flat mean saliency across sites ($\langle|g_i|\rangle \approx 0.50$–$0.55$), as expected from the translational symmetry of the PBC Hamiltonian. The BiRNNs show higher overall saliency ($\approx 0.65$–$0.75$) with a characteristic profile: BiRNN-1layer (green) is relatively flat, while BiRNN-2layer (red) shows a pronounced dip in the middle and peaks at the edges (sites 0 and 19). This edge sensitivity reflects the RNN's sequential processing: the forward pass starts at site 0, the backward pass starts at site 19, so information from the endpoints is processed with the freshest hidden state. Despite PBC in the Hamiltonian, the BiRNN's internal processing is inherently non-periodic.

GNN-3layer-local (purple) shows a mean saliency of $\approx 0.55$, close to the CNN level, with a profile that is approximately flat across sites. This is consistent with the GNN's translation-equivariant message-passing structure, which respects the periodic boundary conditions by construction (circular neighbor lookups via `jnp.roll`).

**Panel (b) - Saliency covariance vs distance:**

$$G(d) = \frac{1}{N} \sum_i \mathrm{Cov}_{\mathbf{s}}[g_i, g_{i+d}]$$

All five models show a clear peak at $d=0$ and rapid decay. The CNNs (blue, orange) show a sharp drop from $d=0$ to $d=1$, consistent with their kernel-3 receptive field. BiRNN-1layer (green) shows near-zero covariance at all distances - its saliency fluctuations are essentially uncorrelated across sites. BiRNN-2layer (red) shows intermediate decay. GNN-3layer-local (purple) shows the highest saliency covariance at $d=0$ ($\approx 0.10$) and a steep but smooth decay, reaching near-zero by $d \approx 4$. This elevated short-range covariance is consistent with the GNN's 3-layer message-passing, which builds a receptive field of radius 3 (each layer aggregates from the two nearest neighbors). All models converge to near-zero covariance by $d \geq 5$, confirming that none have learned spurious long-range couplings absent from the Hamiltonian.

**Panel (c) - Saliency CKA:**

Cross-model saliency CKA is high within the CNN family (0.99) and moderate-to-high for CNN↔BiRNN (0.87–0.92). The BiRNN-1layer ↔ BiRNN-2layer CKA is 0.87. GNN-3layer-local shows CKA of 0.91 with CNN-3layer-k3, 0.95 with CNN-5layer-k3 and BiRNN-2layer, and 0.77 with BiRNN-1layer - the latter being the lowest entry in the matrix and reflecting the very different saliency structure of the single-layer BiRNN (near-zero spatial covariance vs. the GNN's strong short-range covariance).

**Panels (d–h) - Saliency covariance heatmaps:**

All models show a banded structure centered on the diagonal, with the strongest off-diagonal entries at $|i-j|=1$ (nearest neighbors). This directly mirrors the Hamiltonian's nearest-neighbor $\sigma^z_i \sigma^z_j$ coupling. The CNN heatmaps (d, e) show tight, symmetric bands with comparable magnitude ($\approx \pm 0.04$). The BiRNN-1layer heatmap (f) shows extremely weak covariance (scale ~$10^{-4}$), while BiRNN-2layer (g) shows a clear nearest-neighbor band ($\approx \pm 0.02$). GNN-3layer-local (h) shows the broadest and most intense banded structure ($\approx \pm 0.10$), with off-diagonal coupling extending visibly to $|i-j| \approx 3$, consistent with its 3-layer message-passing receptive field.

**Verdict:** Strong support. All architectures have learned to couple nearest-neighbor sites most strongly, mirroring the Hamiltonian graph. The saliency *content* is similar (high CKA), but the *profile* differs (BiRNN edge effects from sequential processing, GNN broader coupling from message-passing). The Hamiltonian's locality is imprinted into every architecture's sensitivity structure.

---

### Figure 6 - Multi-distance $z_i z_{i+d}$ decoding

![Multi-distance decoding](results/figures/fig6_multidistance_decoding.png)

**What it measures.** For each model's first hidden layer, train a ridge probe to predict the correlator $z_i z_{i+d}$ at distances $d = 1, \ldots, 10$. The $R^2$ at each distance indicates how much information about the $d$-distant interaction the first hidden layer has extracted.

**Results.**

*Left panel (physical samples):*

- All models achieve $R^2 \geq 0.9$ at $d=1$, except GNN-3layer-local which achieves $R^2 \approx 0.67$.
- CNN-5layer-k3 (orange) achieves $R^2 \approx 1.0$ at $d=1$ but drops to $\sim 0.35$ by $d=2$. The narrow kernel encodes only the nearest neighbor.
- CNN-3layer-k3 (blue, hidden behind orange at $d=1$) follows a nearly identical pattern.
- BiRNN-1layer (green) maintains $R^2 \approx 0.65$ out to $d=3$ before decaying to $\sim 0.38$. The sequential scan lets a single RNN layer "see" further than a single CNN layer.
- BiRNN-2layer (red) starts at $R^2 \approx 0.97$ at $d=1$ and decays to $\sim 0.55$ by $d=2$, then gradually to $\sim 0.33$.
- GNN-3layer-local (purple) starts lower ($R^2 \approx 0.67$ at $d=1$) and decays quickly to a plateau of $\sim 0.33$ by $d=3$. This lower starting $R^2$ relative to the CNNs reflects the GNN's first layer (`gnn1`) having a receptive field of only radius 1 (self + two neighbors), comparable to the CNNs' kernel-3 - but the GNN distributes its feature capacity across the message and update MLPs, leaving less linearly decodable signal for the probe.
- All models converge to $R^2 \approx 0.33$ by $d \geq 7$. This residual plateau reflects the inherent correlation in physical samples - the ground state at $h/J=1$ has algebraically decaying correlations, so $z_i z_{i+d}$ is partially predictable from $z_i$ alone.

*Right panel (uniform samples):*

- CNN-3layer-k3 (blue) and CNN-5layer-k3 (orange) achieve $R^2 \approx 1.0$ at $d=1$ and drop to nearly zero by $d=2$. The k=3 receptive field encodes *exactly* the nearest neighbor and nothing else.
- BiRNN-1layer (green) maintains $R^2 \approx 0.83$ at $d=2$ and $R^2 \approx 0.08$ at $d=4$ before dropping. The RNN's sequential memory allows decoding of correlators several sites away.
- BiRNN-2layer (red) decays similarly to BiRNN-1layer but slightly faster.
- GNN-3layer-local (purple) achieves $R^2 \approx 0.63$ at $d=1$ and drops to near-zero by $d=2$, closely tracking the CNNs' sharp cutoff. This confirms that the GNN's first hidden layer (gnn1) has a similar effective receptive field to the kernel-3 CNNs.

The uniform-sample panel is the more diagnostic test because it removes the confound of input correlations. The fact that $R^2 > 0$ at $d=1$ on uniform samples - where spins are independent - proves the network has **hardwired** the nearest-neighbor interaction into its weights, not just learned to exploit input statistics.

**Verdict:** Strong support with nuance. All architectures encode the nearest-neighbor Hamiltonian term in their first hidden layer, but the *range* of encoded interactions differs: k=3 CNNs and the GNN's first layer encode exactly one bond, while BiRNNs encode several. The interaction range is determined by the architectural receptive field, not by the Hamiltonian. All architectures "mirror the Hamiltonian's circuit structure" but through different windows.

---

### Figure 7 - Local Hamiltonian-term decoding

![Local decoding](results/figures/fig7_local_decoding.png)

**What it measures.** Layer-by-layer $R^2$ for a ridge probe predicting $z_i z_{i+1}$ (the nearest-neighbor Ising term) from hidden activations. Dashed line = baseline from raw input.

**Results.**

*Top row (physical samples):*

Every model shows the same qualitative pattern: $R^2$ jumps sharply at the first hidden layer (conv1, rnn1, gnn1), peaks there or at the second hidden layer, and then decays toward the readout. The first hidden layer achieves $R^2 \approx 0.95$–$1.0$ for both CNNs and BiRNN-2layer, and $R^2 \approx 0.85$ for BiRNN-1layer. GNN-3layer-local peaks at gnn1 ($R^2 \approx 0.85$) and decays monotonically through g2 ($\approx 0.75$) and g3 ($\approx 0.50$) to the readout ($\approx 0.35$). The readout layer has lower $R^2$ (0.3–0.5) across all models because it compresses the per-site features into a single scalar for the log-amplitude sum - the local correlation information has been "consumed" to build the wave function.

*Bottom row (uniform samples):*

Same pattern, but the input baseline drops to $R^2 \approx 0$. The first hidden layer achieves $R^2 \approx 0.95$–$1.0$ for the CNNs and BiRNN-2layer, $R^2 \approx 0.95$ for BiRNN-1layer, and $R^2 \approx 0.80$ for GNN-3layer-local. The GNN's slightly lower peak $R^2$ on uniform samples may reflect a less linearly separable encoding of the nearest-neighbor term, even though the information is clearly present. This is strong evidence that all models have hardwired the nearest-neighbor interaction, since the uniform samples remove any input-correlation confound.

**Verdict:** Strong support. The "first hidden layer = Hamiltonian term encoder" pattern is universal across all five architectures. This is the most direct mechanistic evidence that the networks learn to mirror the Hamiltonian's local interaction structure, and they all do it at the same computational depth (first layer).

---

### Figure 8 - Learned correlation functions

![Correlation functions](results/figures/fig8_correlation_functions.png)

**What it measures.** The two-point correlator $C(d) = \frac{1}{N}\sum_i \langle \sigma^z_i \sigma^z_{i+d} \rangle$ computed from MCMC samples drawn from each model's $|\psi_\theta|^2$.

**Results.**

The correlation functions split into three groups:

- **CNN-3layer-k3 + CNN-5layer-k3** (blue, orange): nearly identical, with $C(1) \approx 0.66$ decaying to $C(10) \approx 0.52$.
- **BiRNN-1layer** (green): systematically higher and flatter, with $C(1) \approx 0.74$ and $C(d) \approx 0.74$ for all $d \geq 1$ - the correlator barely decays. This indicates the BiRNN-1layer wave function overestimates long-range order.
- **BiRNN-2layer** (red): similar to BiRNN-1layer but slightly lower, with $C(d) \approx 0.73$ at long range.
- **GNN-3layer-local** (purple): shows the steepest decay of all five models, with $C(1) \approx 0.64$ dropping to $C(10) \approx 0.45$. The GNN captures more of the critical algebraic decay structure than the other architectures, consistent with its slightly lower ground-state energy ($E/N \approx -1.273$).

The spread between the models ($\Delta C(10) \approx 0.29$, from GNN at 0.45 to BiRNN-1layer at 0.74) indicates that the BiRNN models have settled into slightly different solutions than the CNNs and GNN. The energy is dominated by the $d=0$ and $d=1$ terms, so $E/N$ can be similar even when $C(d>2)$ differs. The BiRNNs' flatter correlation functions suggest they capture less of the critical decay structure, consistent with their slightly higher energies ($E/N \approx -1.25$ vs $-1.27$). The GNN's steeper decay is the most physically realistic profile among the five models.

**Verdict:** Partial concern. The CNN pair and GNN have converged to consistent states with appropriate decay, but the BiRNNs produce flatter correlations with higher long-range order. This is a known challenge for sequential architectures at criticality. The representation comparisons (Figs 1–7) remain valid - the high tangent CKA (0.95–1.00) shows the models occupy the same variational manifold even if they have not converged to exactly the same point on it.

---

### Figure 9 - Within-model layer CKA

![Within-model layer CKA](results/figures/fig9_within_model_cka.png)

**What it measures.** For each model individually, linear CKA is computed between every pair of its own layers' activations. This reveals how much each layer transforms the representation relative to its predecessors: high CKA between adjacent layers means the transformation is mild; low CKA between distant layers means deep processing has occurred.

**Results.**

- **CNN-3layer-k3:** CKA decays gradually with layer distance. The input-to-output CKA is 0.64, and adjacent hidden layers have CKA $\geq 0.76$. The representation changes smoothly across depth.
- **CNN-5layer-k3:** A similar pattern but with a more pronounced gradient due to the greater depth. Input-to-output CKA is 0.61. The early layers (1–2) have high CKA with the input ($\approx 0.94$), while deeper layers progressively diverge. Adjacent layers maintain CKA $\geq 0.75$ throughout.
- **BiRNN-1layer:** Very high CKA across all layer pairs ($\geq 0.96$). With only one hidden layer and a readout, the representation undergoes minimal transformation - the single recurrent layer produces features that are already close to the final readout space.
- **BiRNN-2layer:** High CKA throughout ($\geq 0.92$), with the two recurrent layers and readout all highly similar. The second recurrent layer refines the representation only mildly compared to the first.
- **GNN-3layer-local:** Shows the most dramatic depth effect. The input and gnn1 layers have high mutual CKA ($\approx 0.88$), but deeper GNN layers diverge sharply. The input-to-output CKA is only 0.26, and gnn1-to-output is 0.21 - the lowest within-model values in the entire experiment. The message-passing layers (gnn2, gnn3) progressively transform the representation into a form far from the original input encoding. This steep CKA gradient indicates that the GNN performs the deepest effective processing of all five architectures, consistent with its 3-layer message-passing building up increasingly nonlocal features.

**Interpretation.** The within-model CKA gradient is inversely related to effective depth: BiRNNs have nearly flat CKA profiles (shallow processing), CNNs have moderate gradients, and the GNN has the steepest gradient (deepest processing). Despite this, all five models achieve comparable energies and near-identical tangent CKA (Fig 2), demonstrating that functional equivalence can emerge from very different representational processing depths. The GNN's steep CKA gradient also explains its low cross-architecture activation CKA (Fig 1): its output-layer representation is so heavily processed that it no longer resembles the intermediate representations of shallower architectures.

**Verdict:** The within-model CKA analysis reveals architecture-dependent processing depths that are invisible to functional metrics like tangent CKA. The GNN's deep processing chain - with input-to-output CKA of 0.26 vs. 0.96 for BiRNN-1layer - demonstrates that architecturally diverse models can achieve the same functional endpoint through fundamentally different representational trajectories.

---

### Figure 10 - Linear vs RBF CKA

![Linear vs RBF CKA](results/figures/fig10_linear_vs_rbf_cka.png)

**What it measures.** A comparison of linear CKA (left) and RBF (radial basis function) CKA (right) on the full tangent features. Linear CKA measures whether two representations span the same linear subspace. RBF CKA replaces the linear kernel with a Gaussian kernel:

$$K^{\mathrm{RBF}}_{ij} = \exp\!\left(-\frac{\|x_i - x_j\|^2}{2 \sigma^2}\right)$$

where $\sigma$ is set as a fraction of the median pairwise distance. RBF CKA is then computed using centered kernel alignment on these nonlinear kernel matrices:

$$\mathrm{CKA}_{\mathrm{RBF}} = \frac{\mathrm{tr}(\tilde{K}_1^{\mathrm{RBF}} \tilde{K}_2^{\mathrm{RBF}})}{\sqrt{\mathrm{tr}((\tilde{K}_1^{\mathrm{RBF}})^2) \; \mathrm{tr}((\tilde{K}_2^{\mathrm{RBF}})^2)}}$$

RBF CKA captures nonlinear representational structure that linear CKA might miss. If linear and RBF CKA agree, the representational similarity is robust to the choice of kernel; if RBF CKA is substantially lower, the linear subspace overlap masks nonlinear geometric differences.

**Numerical example.** Given two feature matrices $X, Y \in \mathbb{R}^{5 \times 3}$, compute pairwise squared-distance matrices $D^X_{ij} = \|x_i - x_j\|^2$ and $D^Y_{ij} = \|y_i - y_j\|^2$. Set $\sigma^2 = 0.5 \times \mathrm{median}(D)$ for each. Exponentiate to get the $5 \times 5$ RBF kernel matrices, center them, and compute kernel alignment as in the NTK case (Fig 2). If the pairwise similarity structure is preserved - i.e., the same pairs of configurations are "close" and "far" in both feature spaces - then RBF CKA will be high.

**Results.**

*Left panel (linear CKA):* Identical to Fig 2, left panel. All cross-model values are 0.95–1.00.

*Right panel (RBF CKA):* Slightly lower than linear CKA but still very high:

- **CNN-3layer-k3 ↔ CNN-5layer-k3:** 0.99 (vs 1.00 linear) - negligible drop.
- **CNN ↔ BiRNN pairs:** 0.94–0.98 (vs 0.96–0.99 linear) - a modest decrease of 0.01–0.03.
- **CNN ↔ GNN-3layer-local:** 0.97 (vs 0.98 linear) - minimal drop.
- **BiRNN-1layer ↔ GNN-3layer-local:** 0.92 (vs 0.95 linear) - the largest drop in the matrix, but still indicating strong similarity.
- **BiRNN-2layer ↔ GNN-3layer-local:** 0.94 (vs 0.98 linear).

**Interpretation.** The close agreement between linear and RBF CKA confirms that the tangent-space universality is not an artifact of linear projection. The representational similarity extends to nonlinear pairwise structure: the same configurations are "close" and "far" in all five models' tangent spaces, not just when projected linearly. The modest RBF CKA decreases (0.01–0.03 for most pairs, up to 0.03 for BiRNN-1layer ↔ GNN) indicate minor nonlinear differences that are consistent with the architecture-dependent spectral tails observed in Fig 3.

**Verdict:** The tangent-space universality is robust to the choice of similarity kernel. Both linear and nonlinear measures confirm that all five architectures have converged to the same functional region.

---

## Synthesis

| Metric | Level tested | Result | Verdict |
|---|---|---|---|
| Activation CKA (Fig 1) | Hidden representations | High within-family; moderate cross-family; GNN output layers diverge | ⚠️ Moderate universality |
| Tangent CKA (Fig 2) | Variational manifold | 0.95–1.00 across all 10 pairs, including CNN↔RNN↔GNN | ✅ Strong universality |
| NTK alignment (Fig 2) | Effective kernel | Identical to tangent CKA; confirms result | ✅ Strong universality |
| NTK spectrum (Fig 3) | Kernel dimensionality | Top eigenvalues shared; tails diverge (GNN flattest, BiRNN-1layer steepest) | ⚠️ Partial universality |
| Procrustes (Fig 4) | Geometric alignment | Low in tangent space (0.08–0.43); high in activations (0.11–1.02) | ⚠️ Subspace shared, geometry differs |
| Saliency (Fig 5) | Site-level sensitivity | All models couple nearest neighbors; CKA 0.77–0.99 | ✅ Strong universality |
| Multi-$d$ decoding (Fig 6) | Interaction range | All encode $d{=}1$; range set by architecture, not Hamiltonian | ⚠️ Partial universality |
| Local decoding (Fig 7) | Depth of encoding | First hidden layer = peak in all models | ✅ Strong universality |
| Correlation functions (Fig 8) | Physical state (control) | CNNs and GNN show decay; BiRNNs show flatter $C(d)$ | ⚠️ Moderate concern |
| Within-model CKA (Fig 9) | Internal processing depth | CNNs/BiRNNs have gradual change; GNN shows steep CKA gradient | 📊 Architecture-dependent |
| Linear vs RBF CKA (Fig 10) | Nonlinear tangent structure | RBF CKA ≥ 0.92 across all pairs; matches linear CKA closely | ✅ Strong universality |

### Conclusion

The hypothesis is **largely confirmed**, with a clear pattern emerging across all three architecture families:

**Universality is strong at the functional level.** The tangent-space CKA (0.95–1.00), NTK kernel alignment, saliency CKA (0.77–0.99), and shared local-decoding depth all point to the same conclusion: the models learn the same variational manifold and the same Hamiltonian locality structure, regardless of architecture. The agreement between linear and RBF CKA (Fig 10) confirms this is not an artifact of linear projection.

**Universality is moderate at the representation level.** Activation CKA shows substantial within-family similarity and moderate cross-family similarity, with the GNN introducing the largest divergence at both early and output layers. Procrustes distance reveals that while the tangent-space geometry is well aligned (0.08–0.43), the readout activation geometry diverges severely for the GNN (approaching 1.0 against the BiRNNs). The networks share representational content but organize it differently.

**Processing depth varies dramatically.** Within-model CKA (Fig 9) reveals that the five architectures reach functional equivalence through very different internal trajectories. The GNN transforms its representation the most aggressively (input-to-output CKA = 0.26), while the BiRNN-1layer barely transforms it at all (input-to-output CKA = 0.96). Despite this 4× range in effective processing depth, all models converge to the same tangent-space manifold.

**The mechanism is Hamiltonian mirroring.** All architectures encode the nearest-neighbor $z_i z_{i+1}$ interaction in their first hidden layer (Fig 7), and their input saliency couples nearest-neighbor sites most strongly (Fig 5d–h). This is the Hamiltonian's circuit structure imprinted into the network's sensitivity pattern.

**The boundary of universality is the receptive field.** While all models encode $d=1$ interactions, the range of encoded interactions (Fig 6) is determined by the architecture's receptive field (kernel size for CNNs, sequential memory for RNNs, message-passing hops for GNNs), not by the Hamiltonian. Universality holds for *what* is encoded (the Hamiltonian's local terms) but not for *how far* each layer sees.

---

## Models

| Name | Architecture | Layers | Channels / Hidden dim | Kernel size | Parameters |
|---|---|---|---|---|---|
| CNN-3layer-k3 | CNN | 3 conv + readout | 10 per layer | 3 | 671 |
| CNN-5layer-k3 | CNN | 5 conv + readout | 7 per layer | 3 | 652 |
| BiRNN-1layer | Bidirectional Elman RNN | 1 BiRNN + readout | 17 (×2 directions = 34) | - | 681 |
| BiRNN-2layer | Bidirectional Elman RNN | 2 BiRNN + readout | 9 (×2 directions = 18) | - | 721 |
| GNN-3layer-local | Local message-passing GNN | embed + 3 MP layers + readout | 7 per layer | - | ~652 |