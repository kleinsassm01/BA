# NQS Representation Universality Analysis

> Different NQS nn architectures converge to the same internal representation when learning the ground state of the same Hamiltonian, because they all learn to mirror the Hamiltonian's circuit structure.

In the first version, two CNN, two bidirectional RNN and one local-message-passing GNN are tested on the 1D TIFM at critical point $h/J = 1$ with $N=20$ spins and periodic boundary conditions:

$$H = -J \sum_{\langle i,j \rangle} \sigma^z_i \sigma^z_j \;-\; h \sum_i \sigma^x_i$$

All models parameterize $\log \psi_\theta(\mathbf{s})$ and are trained with VMC to minimize $\langle H \rangle$, where $\mathbf{s}=(s_1,\dots,s_N)$ denotes a spin configuration in the $\sigma^z$-basis with $s_i\in\{+1,-1\}$.

---

All five models converge to comparable ground-state energies. The two CNNs reach $E/N \approx -1.272$, the BiRNNs land at $E/N \approx -1.249$ to $-1.258$, and GNN-3layer-local achieves $E/N \approx -1.273$. All models approximate the same ground state.

---

### Figure 1 - Cross-architecture activation CKA

![Cross-architecture activation CKA](results/figures/fig1_cross_activation_cka.png)

**What it measures.** Linear Centered Kernel Alignment (CKA, https://arxiv.org/pdf/1905.00414) between the hidden-layer activations of every pair of models evaluated on the same set of spin configurations drawn from $|\psi|^2$. 

---
All models parameterize $\log \psi_\theta(\mathbf{s})$ and are evaluated on the **same spin configurations** $\mathbf{s}=(s_1,\dots,s_N)$ with $s_i\in\{+1,-1\}.$ Procedure:
1. Draw configurations from the Born distribution: $\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2$, for example for a 6-spin system:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1) \quad \mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)$$
2. Feed the SAME samples into two models, for example Model A = CNN, Model B = RNN, receiving exactly the same spin configurations.
3. Suppose the RNN processes spins sequentially $s_1 \to s_2 \to s_3 \to \cdots \to s_N$, the hidden state evolves as $h_t = f(h_{t-1}, s_t)$ where:
    - $s_t$ = input spin at step $t$
    - $h_t$ = hidden activation
    - $f$ = recurrent update
4. Example forward pass: Input configuration: $\mathbf{s}^{(1)}=
(+1,-1,+1,+1,-1,+1)$, the RNN computes:
$$
h_1 = f(h_0,+1), \quad h_2 = f(h_1,-1), \quad h_3 = f(h_2,+1) \cdots
$$
Suppose the hidden dimension is 3, the activations might become:
$$
h_1 =
\begin{bmatrix}
0.2\\
-0.1\\
0.7
\end{bmatrix}
\quad 
h_2 =
\begin{bmatrix}
0.5\\
0.3\\
-0.2
\end{bmatrix}
\quad
h_3 =
\begin{bmatrix}
0.9\\
0.1\\
0.4
\end{bmatrix}
$$ These vectors are the **hidden-layer activations**. They encode information about: previous spins, correlations, entanglement structure, ...
5. Build activation matrices. For Model A, $H_A$ and Model B, $H_B$ each row corresponds to the hidden representation of one sampled spin configuration.:
$$H_A=
\begin{bmatrix}
h_A(\mathbf{s}^{(1)}) \\
h_A(\mathbf{s}^{(2)}) \\
\vdots \\
h_A(\mathbf{s}^{(M)})
\end{bmatrix} \quad
H_B=
\begin{bmatrix}
h_B(\mathbf{s}^{(1)}) \\
h_B(\mathbf{s}^{(2)}) \\
\vdots \\
h_B(\mathbf{s}^{(M)})
\end{bmatrix}
$$
6. Compute the linear CKA: Linear CKA compares the similarity between the two representation matrices:
$$
\mathrm{CKA}(H_A,H_B)
=
\frac{
\|H_A^\top H_B\|_F^2
}{
\|H_A^\top H_A\|_F
\,
\|H_B^\top H_B\|_F
}
$$

A high CKA value means:
- the two models organize spin configurations similarly
- they learned similar many-body correlations
- their internal representations are aligned
  
---
CKA measures **representation similarity**, not just final energy accuracy.
In general, given two centered activation matrices $X \in \mathbb{R}^{n \times p}$ and $Y \in \mathbb{R}^{n \times q}$:

$$\mathrm{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \;\|Y^\top Y\|_F}$$

CKA is invariant to invertible linear transforms and isotropic scaling: Suppose all activations are multiplied by a constant:$\tilde H = 5H$, then every neuron output becomes 5 times larger f.ex.:
$$
h =
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}
\quad\to\quad
\tilde h =
\begin{bmatrix}
5\\
10\\
15
\end{bmatrix}
$$

The representation is fundamentally the same — only the scale changed. CKA gives the same similarity score, so CKA ignores global magnitude differences.

$$\mathrm{CKA}(H_A,H_B)=
\mathrm{CKA}(5H_A,H_B)
$$
It equals 1 when $X$ and $Y$ span the same column space and 0 when they induce orthogonal kernel matrices.

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

**What it measures.** Two model×model similarity matrices computed from the full variational tangent vectors 

These metrics compare models using their **variational tangent vectors** $J(\mathbf{s}) = \partial \log \psi_\theta(\mathbf{s}) / \partial \theta$ instead of hidden activations. For each sampled spin configuration $\mathbf{s}$, the tangent vector measures, ow does the wavefunction change if the model parameters are infinitesimally perturbed?. Thus, these methods compare the **local geometry of the variational manifold** learned by different neural quantum states. All models parameterize $\log \psi_\theta(\mathbf{s})$ and are evaluated on the SAME sampled spin configurations $\mathbf{s}=(s_1,\dots,s_N)$ with $s_i\in\{+1,-1\}$. Procedure:
1. Draw configurations from the Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
for example for a 6-spin system:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)
$$

2. Feed the SAME samples into two models, for example Model A = CNN and Model B = RNN, receiving exactly the same spin configurations.

3. For each sampled configuration $\mathbf{s}^{(i)}$, compute the variational tangent vector:
$$
J(\mathbf{s}^{(i)})
=
\frac{\partial \log \psi_\theta(\mathbf{s}^{(i)})}{\partial \theta}
$$
- $\theta$ = all trainable parameters
- $J(\mathbf{s})$ = gradient of the wavefunction with respect to parameters

4. Concrete RNN example. Suppose the RNN outputs:
$
\log \psi_\theta(\mathbf{s}) = f_\theta(\mathbf{s})
$
for the sampled configuration
$
\mathbf{s}^{(1)}
=
(+1,-1,+1,+1,-1,+1).
$
Assume the RNN has 3 trainable parameters:
$
\theta=(\theta_1,\theta_2,\theta_3).
$ After backpropagation, the tangent vector may become:
$$
J(\mathbf{s}^{(1)})
=
\begin{bmatrix}
0.8 \\
-0.2 \\
0.5
\end{bmatrix}
$$
- increasing $\theta_1$ strongly increases $\log\psi$
- $\theta_2$ decreases it
- $\theta_3$ moderately increases it

5. Build tangent matrices. For Model A and Model B each row corresponds to one sampled spin configuration and each column corresponds to one trainable parameter.
$$
J_A=
\begin{bmatrix}
J_A(\mathbf{s}^{(1)}) \\
J_A(\mathbf{s}^{(2)}) \\
\vdots \\
J_A(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{n\times d_A}
\qquad 
J_B=
\begin{bmatrix}
J_B(\mathbf{s}^{(1)}) \\
J_B(\mathbf{s}^{(2)}) \\
\vdots \\
J_B(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{n\times d_B}
$$ 

6. Compute tangent-space CKA:
$$
\mathrm{CKA}(J_A,J_B)
=
\frac{
\|J_B^\top J_A\|_F^2
}{
\|J_A^\top J_A\|_F
\,
\|J_B^\top J_B\|_F
}
$$

7. A high tangent-space CKA means:
- the two models respond similarly to parameter perturbations
- they span similar variational subspaces
- they induce similar local wavefunction deformations

8. Construct the Neural Tangent Kernel (NTK, https://en.wikipedia.org/wiki/Neural_tangent_kernel) 
$
K_A = J_A J_A^\top,
$ and $
K_B = J_B J_B^\top
$
where the following expression measures how similarly the model responds to two spin configurations.
$$
(K_A)_{ij}
=
J(\mathbf{s}^{(i)})^\top
J(\mathbf{s}^{(j)})
$$
9. Center the kernels: $\tilde K = H K H$
with centering matrix
$$
H = I - \frac1n\mathbf1\mathbf1^\top.
$$

10. Compute kernel alignment:
$$
\mathrm{KA}(K_A,K_B)
=
\frac{
\mathrm{tr}(\tilde K_A \tilde K_B)
}{
\sqrt{
\mathrm{tr}(\tilde K_A^2)
\;
\mathrm{tr}(\tilde K_B^2)
}
}
$$

A high kernel alignment means:
- the two models induce similar similarity structure over configuration space
- optimization updates couple configurations similarly
- the models possess similar learning geometry

---
*Left panel (tangent CKA):* Treat $J_A \in \mathbb{R}^{n \times d_A}$ and $J_B \in \mathbb{R}^{n \times d_B}$ as feature matrices (rows = configurations, columns = parameter gradients), then compute CKA. This measures whether the two models' variational derivatives span the same functional subspace over configuration space.

**Numerical example.** For tangent-space CKA, suppose $J_A$ and $J_B$ are tangent matrices built from the same $n=3$ sampled spin configurations. Let
$
J_A \in \mathbb{R}^{3\times 2}
$
and
$
J_B \in \mathbb{R}^{3\times 2}
$.
The numerator is $\|J_B^\top J_A\|_F^2$. If

$$
J_B^\top J_A
=
\begin{bmatrix}
3 & 1\\
0 & 2
\end{bmatrix},
$$

then

$$
\|J_B^\top J_A\|_F^2
=
3^2+1^2+0^2+2^2
=
14.
$$

The denominator is
$
\|J_A^\top J_A\|_F \cdot \|J_B^\top J_B\|_F
$.
If these equal $4.0$ and $3.7$, then

$$
\mathrm{CKA}(J_A,J_B)
=
\frac{14}{4.0\times 3.7}
\approx 0.95.
$$

This indicates that the two models have highly similar tangent-space geometry, meaning their parameter gradients span very similar variational directions over the sampled spin configurations.

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

**What it measures.** The eigenvalue spectrum of each model's neural tangent kernel
$
K = JJ^\top
$.
The eigenvalues
$
\lambda_1 \geq \lambda_2 \geq \ldots
$
determine the effective dimensionality of the learned kernel. The $k$-th eigenvalue measures how much the model's output varies along the $k$-th principal direction in configuration space. Procedure:

---
1. Draw configurations from the Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
for example for a 6-spin system:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)
$$

2. Feed the same sampled spin configurations into a model, for example a CNN, RNN, or GNN neural quantum state.

3. For each sampled configuration $\mathbf{s}^{(i)}$, compute the variational tangent vector, each tangent vector measures how sensitively the wavefunction changes under infinitesimal parameter perturbations.
$$
J(\mathbf{s}^{(i)})
=
\frac{\partial \log \psi_\theta(\mathbf{s}^{(i)})}{\partial \theta}.
$$

4. Build the tangent matrix, where $n$ = number of sampled spin configurations and $d$ = number of trainable parameters. Each row corresponds to one sampled configuration and each column corresponds to one parameter derivative.
$$
J=
\begin{bmatrix}
J(\mathbf{s}^{(1)}) \\
J(\mathbf{s}^{(2)}) \\
\vdots \\
J(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{n\times d},
$$

5. Construct the Neural Tangent Kernel:
$
K = JJ^\top
\in \mathbb{R}^{n\times n}.
$. The following kernel entry measures how similarly the model responds to configurations $\mathbf{s}^{(i)}$ and $\mathbf{s}^{(j)}$.
$$
K_{ij}
=
J(\mathbf{s}^{(i)})^\top
J(\mathbf{s}^{(j)})
$$



6. Compute the eigenvalue decomposition, where $\Lambda =
\mathrm{diag}(\lambda_1,\lambda_2,\ldots,\lambda_n)$ contains the eigenvalues ordered as
$
\lambda_1 \geq \lambda_2 \geq \cdots
$
$$
K = U \Lambda U^\top,
$$
7. The eigenvalues determine how strongly the model varies along different directions in configuration space. Large leading eigenvalues indicate dominant collective deformation modes of the wavefunction.

---

**Numerical example.** Suppose for one model the NTK

$$
K = JJ^\top
=
\begin{bmatrix}
2 & 1 & 0\\
1 & 3 & 1\\
0 & 1 & 2
\end{bmatrix}.
$$

The eigenvalues of this kernel are $
\lambda_1 = 4,
\lambda_2 = 2,
\lambda_3 = 1.
$
Since $
\lambda_1 \geq \lambda_2 \geq \lambda_3,
$
the first direction explains the largest amount of variation in the model's tangent responses over configuration space. The total kernel variance is

$$
\sum_i \lambda_i = 4+2+1 = 7.
$$

So the fraction explained by each direction is

$$
\frac{\lambda_1}{\sum_i \lambda_i}
=
\frac{4}{7}
\approx 0.57,
$$

$$
\frac{\lambda_2}{\sum_i \lambda_i}
=
\frac{2}{7}
\approx 0.29,
$$

$$
\frac{\lambda_3}{\sum_i \lambda_i}
=
\frac{1}{7}
\approx 0.14.
$$

Therefore most variation is concentrated in the first principal kernel direction. This suggests that the model's tangent space is relatively low-dimensional: parameter updates mainly deform $\log\psi_\theta(\mathbf{s})$ along a few dominant directions in configuration space.

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

$$
d_{\mathrm{Proc}}(X, Y)
=
\min_{R^\top R = I}
\left\|
\frac{X}{\|X\|_F}R
-
\frac{Y}{\|Y\|_F}
\right\|_F
$$

CKA is invariant to *any* invertible linear transform; Procrustes is invariant only to orthogonal transforms. So:
- $d_{\mathrm{Proc}}=0$ means the representations are geometrically identical up to rotation
- $\mathrm{CKA}=1$ only means they span the same subspace

Therefore, Procrustes (https://arxiv.org/pdf/2305.06329) is a stricter notion of representational similarity.

---

1. Draw configurations from the Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
for example for a 6-spin system:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)
$$

2. Feed the SAME sampled spin configurations into two models, for example Model A = CNN and Model B = RNN.

3. Extract hidden-layer activations from both models. For Model A and Model B each row corresponds to one sampled spin configuration and each column corresponds to one hidden feature.
$$
X=
\begin{bmatrix}
x(\mathbf{s}^{(1)}) \\
x(\mathbf{s}^{(2)}) \\
\vdots \\
x(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{n\times p} \qquad 
Y=
\begin{bmatrix}
y(\mathbf{s}^{(1)}) \\
y(\mathbf{s}^{(2)}) \\
\vdots \\
y(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{n\times q}.
$$

4. Center the representations by subtracting the mean feature vector:
$
X_c = HX$ and $
Y_c = HY
$ with the following centering matrix removes global offsets in the activations.
$$
H = I - \frac1n\mathbf1\mathbf1^\top.
$$

5. Normalize the centered matrices using the Frobenius norm, which removes overall scale differences between representations
$$
\hat X = \frac{X_c}{\|X_c\|_F},
\qquad
\hat Y = \frac{Y_c}{\|Y_c\|_F}.
$$
6. Find the orthogonal rotation $R^\top R = I$ that best aligns the two normalized representations, typically $R^*$ is obtained from the singular value decomposition (SVD) of $\hat X^\top \hat Y.$:
$$
R^*
=
\arg\min_{R^\top R=I}
\|\hat X R - \hat Y\|_F.
$$

7. Compute the Procrustes distance:
$$
d_{\mathrm{Proc}}(X,Y)
=
\|\hat X R^* - \hat Y\|_F.
$$

A small Procrustes distance means the two models represent spin configurations almost identically after a pure rotation.

---

**Numerical example.** Suppose after centering and normalization following matrices are obtained:
$$
\hat X =
\begin{bmatrix}
0.6 & 0.1\\
0.2 & 0.7\\
-0.5 & -0.4
\end{bmatrix},
\qquad
\hat Y =
\begin{bmatrix}
0.58 & 0.12\\
0.18 & 0.69\\
-0.52 & -0.41
\end{bmatrix}.
$$

Suppose the optimal orthogonal alignment matrix is

$$
R^*
=
\begin{bmatrix}
0.99 & -0.05\\
0.05 & 0.99
\end{bmatrix}.
$$

Applying the rotation gives:

$$
\hat X R^*
\approx
\hat Y.
$$

If

$$
\|\hat X R^* - \hat Y\|_F
=
0.06,
$$

then

$$
d_{\mathrm{Proc}}(X,Y)=0.06.
$$

Since the distance is very small, the two representations are nearly geometrically identical up to rotation. This means the two neural quantum states organize spin configurations almost identically in representation space, not merely within the same subspace as measured by CKA. A value of 0 indicates perfect geometric alignment; $\sqrt{2} \approx 1.414$ is the theoretical maximum and the worst result.

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


**What it measures.** The input gradient

$$
g_i(\mathbf{s})
=
\frac{\partial \log \psi_\theta(\mathbf{s})}{\partial \sigma_i}
$$

measures how sensitive the log-wavefunction amplitude is to the spin at site $i$.

Large $|g_i|$ means:
- changing spin $i$ strongly affects $\log\psi_\theta$
- the model considers that site important
- the wavefunction is locally sensitive there

Therefore, saliency (https://arxiv.org/pdf/1711.00867) probes which spatial structures the NQS relies on. procedure:

1. Draw configurations from the Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
for example for a 6-spin system:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)
$$

2. Feed the same sampled spin configurations into all models (CNN, BiRNN, GNN).

3. For each sampled configuration $\mathbf{s}^{(k)}$, compute the input gradients:
$$
g_i(\mathbf{s}^{(k)})
=
\frac{\partial \log \psi_\theta(\mathbf{s}^{(k)})}{\partial \sigma_i}.
$$

4. Collect the saliency vector:
$$
g(\mathbf{s}^{(k)})
=
\begin{bmatrix}
g_1(\mathbf{s}^{(k)}) \\
g_2(\mathbf{s}^{(k)}) \\
\vdots \\
g_N(\mathbf{s}^{(k)})
\end{bmatrix}.
$$

Each entry measures sensitivity to one lattice site.

**Numerical example.** Suppose for a 6-spin configuration:

$$
\mathbf{s}^{(1)}
=
(+1,-1,+1,+1,-1,+1)
$$

the model produces:

$$
g(\mathbf{s}^{(1)})
=
\begin{bmatrix}
0.72\\
0.65\\
0.61\\
0.58\\
0.63\\
0.71
\end{bmatrix}.
$$

Interpretation:
- sites 1 and 6 strongly affect $\log\psi_\theta$
- site 4 is less important
- the model is most sensitive near the edges

This type of edge enhancement is characteristic of sequential BiRNN processing.

---

**Panel a: Mean saliency profile** 

For each site:
$$
\langle |g_i| \rangle
=
\frac1M
\sum_{k=1}^M
|g_i(\mathbf{s}^{(k)})|.
$$

This measures the average importance of site $i$ across sampled configurations.

**numerical example** 

Suppose for site $i=3$ following saliencies obtained:

$$
|g_3(\mathbf{s}^{(1)})|=0.60,
\quad
|g_3(\mathbf{s}^{(2)})|=0.55,
\quad
|g_3(\mathbf{s}^{(3)})|=0.65.
$$

Then:

$$
\langle |g_3| \rangle
=
\frac{0.60+0.55+0.65}{3}
=
0.60.
$$

Plotting this for all sites produces the mean saliency profile. Interpretation:
- Flat profile $\rightarrow$ translation-equivariant sensitivity
- Edge peaks $\rightarrow$ sequential processing asymmetry
- Broad plateaus $\rightarrow$ spatially uniform correlation structure

---

**Panel b: Saliency covariance vs distance** 

Compute:

$$
G(d)
=
\frac1N
\sum_i
\mathrm{Cov}_{\mathbf{s}}
[g_i,g_{i+d}].
$$

This measures how correlated the saliency fluctuations are between sites separated by distance $d$.

**Numerical example** 

Suppose for nearest neighbors:

$$
(g_1,g_2)
=
(0.7,0.6),
(0.8,0.7),
(0.6,0.5).
$$

The covariance is:

$$
\mathrm{Cov}(g_1,g_2)
=
\langle g_1g_2\rangle
-
\langle g_1\rangle
\langle g_2\rangle.
$$

If:

$$
\langle g_1g_2\rangle = 0.44,
\qquad
\langle g_1\rangle = 0.70,
\qquad
\langle g_2\rangle = 0.60,
$$

then:

$$
\mathrm{Cov}(g_1,g_2)
=
0.44-(0.70)(0.60)
=
0.02.
$$

A large covariance at small $d$ means nearby sites influence the wavefunction together. Interpretation:

- Fast decay $\rightarrow$ local receptive field
- Long tails $\rightarrow$ long-range sensitivity
- Strong nearest-neighbor covariance $\rightarrow$ Hamiltonian locality learned correctly

---
**Panel c: Saliency CKA** 

Treat saliency vectors as feature representations:

$$
G_A=
\begin{bmatrix}
g_A(\mathbf{s}^{(1)}) \\
g_A(\mathbf{s}^{(2)}) \\
\vdots
\end{bmatrix},
\qquad
G_B=
\begin{bmatrix}
g_B(\mathbf{s}^{(1)}) \\
g_B(\mathbf{s}^{(2)}) \\
\vdots
\end{bmatrix}.
$$

Compute:

$$
\mathrm{CKA}(G_A,G_B)
=
\frac{
\|G_B^\top G_A\|_F^2
}{
\|G_A^\top G_A\|_F
\,
\|G_B^\top G_B\|_F
}.
$$

This measures whether two models organize saliency patterns similarly.

**numerical example:** 

Suppose:

$$
G_B^\top G_A
=
\begin{bmatrix}
2 & 1\\
1 & 2
\end{bmatrix}.
$$

Then:

$$
\|G_B^\top G_A\|_F^2
=
2^2+1^2+1^2+2^2
=
10.
$$

If:

$$
\|G_A^\top G_A\|_F=3.1,
\qquad
\|G_B^\top G_B\|_F=3.3,
$$

then:

$$
\mathrm{CKA}
=
\frac{10}{3.1\times3.3}
\approx0.98.
$$

This indicates highly similar saliency structure between the two models.

---
**Panels (d–h) — Saliency covariance heatmaps**

Construct the covariance matrix:

$$
C_{ij}
=
\mathrm{Cov}_{\mathbf{s}}
[g_i,g_j].
$$

Each entry measures how strongly saliency fluctuations at sites $i$ and $j$ are correlated.

**numerical example**

Suppose:

$$
C=
\begin{bmatrix}
0.05 & 0.03 & 0.01\\
0.03 & 0.05 & 0.03\\
0.01 & 0.03 & 0.05
\end{bmatrix}.
$$

Interpretation:
- strongest values occur on the diagonal
- nearest neighbors have strong covariance
- distant sites are weakly coupled

This produces a banded heatmap centered on the diagonal.

---
**Results.**

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

**What it measures.** For each model's first hidden layer, train a ridge-regression probe to predict the correlator $z_i z_{i+d}$ at distances $d=1,\dots,10$. The probe's coefficient of determination $R^2$ measures how much information about the distance-$d$ interaction is linearly decodable from the first hidden layer.

Large $R^2$ means:
- the hidden layer explicitly encodes the correlator
- the architecture can represent interactions at distance $d$
- the receptive field reaches that separation

---
1. Draw spin configurations:
- either from the physical Born distribution
$$
\mathbf{s}^{(1)},\dots,\mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$$
- or from a uniform random distribution over spins

For example:
$$
\mathbf{s}^{(1)}
=
(+1,-1,+1,+1,-1,+1)
$$

$$
\mathbf{s}^{(2)}
=
(-1,-1,+1,-1,+1,+1).
$$

2. Feed the SAME sampled configurations into a trained model (CNN, BiRNN, or GNN). Extract the first hidden-layer activations $
h^{(1)}(\mathbf{s}^{(k)})
\in
\mathbb{R}^p.
$ For all samples, build the hidden-feature matrix, where each row corresponds to one sampled configuration.

$$
H=
\begin{bmatrix}
h^{(1)}(\mathbf{s}^{(1)}) \\
h^{(1)}(\mathbf{s}^{(2)}) \\
\vdots \\
h^{(1)}(\mathbf{s}^{(M)})
\end{bmatrix}
\in \mathbb{R}^{M\times p}.
$$

3. For a chosen distance $d$, compute the target correlator for periodic boundary conditions: $i+d \mod N.$

$$
y^{(k)}_{(d)}
=
z_i^{(k)}z_{i+d}^{(k)}.
$$

**numerical example**

Suppose $\mathbf{s}^{(1)}=(+1,-1,+1,+1,-1,+1)$ for distance $d=1$
the nearest-neighbor correlators are:

$$
(+1)(-1)=-1,
$$

$$
(-1)(+1)=-1,
$$

$$
(+1)(+1)=+1,
$$

etc.

Suppose for one site $
y^{(1)}_{(1)} = z_2 z_3 = (-1)(+1)=-1.
$ for distance: $d=3$ to obtain:
$$
y^{(1)}_{(3)}
=
z_1 z_4
=
(+1)(+1)
=
+1.
$$

These correlators become the regression targets.

4. Train a ridge-regression probe: $\hat y = HW+b$ to predict the correlators from the hidden activations. If the probe is linear:
- it cannot invent information
- it only tests what is already encoded in the hidden layer

5. Evaluate decoding performance using:

$$
R^2
=
1-
\frac{
\sum_k (y_k-\hat y_k)^2
}{
\sum_k (y_k-\bar y)^2
}.
$$

Interpretation:
- $R^2=1$ → perfect decoding
- $R^2=0$ → no predictive information
- larger $R^2$ → stronger encoded correlator signal

**Numerical example for $R^2$**

Suppose the true correlators are:

$$
y=
\begin{bmatrix}
1\\
-1\\
1\\
1
\end{bmatrix}
$$

and the probe predicts:

$$
\hat y=
\begin{bmatrix}
0.9\\
-0.8\\
0.7\\
0.95
\end{bmatrix}.
$$

Suppose:

$$
\sum_k (y_k-\hat y_k)^2 = 0.10,
$$

and

$$
\sum_k (y_k-\bar y)^2 = 1.00.
$$

Then:

$$
R^2
=
1-\frac{0.10}{1.00}
=
0.90.
$$

This means the hidden layer explains 90% of the variance in the correlator.

---

**Left panel — Physical samples**

The probe is trained using configurations sampled from:

$$
|\psi(\mathbf{s})|^2.
$$

These samples already contain physical correlations from the quantum state.

Therefore:
- some long-distance predictability may come from the data distribution itself
- even architectures with local receptive fields can partially predict distant correlators

**Numerical interpretation**

Suppose:
- CNN first-layer features only encode nearest neighbors
- but the physical samples themselves contain long-range correlations

Then decoding at $d=5$ may still give $R^2\approx0.35$ because:
$
z_i
$
already partially predicts
$
z_{i+5}.
$ Therefore the residual plateau at large distance reflects physical correlations in the data.

---

**Right panel — Uniform samples**

Now spins are sampled independently:

$$
P(\mathbf{s})=
2^{-N}.
$$

There are NO physical correlations.

Therefore:
- any nonzero decoding performance must come entirely from the architecture itself
- this isolates the model's receptive field

**Numerical interpretation**

Suppose a kernel-3 CNN sees only nearest neighbors. Then: 
- decoding at $d=1$ may give $R^2\approx1$
- decoding at $d=2$ collapses to $R^2\approx0$


This proves:
- the first hidden layer encodes exactly one bond distance
- no hidden long-range information exists

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

*Conclusion:* 

**Kernel-3 CNNs:** The results reflects the convolutional receptive field:
- encode nearest neighbors extremely well
- sharply lose information beyond the kernel radius

**BiRNNs:** even the first hidden layer contains multi-distance information:
- maintain information across several sites
- sequential hidden-state memory propagates correlations farther

**GNNs:** the broader behavior emerges only after multiple message-passing layers:
- The first GNN layer aggregates self node and nearest neighbors
- Therefore effective radius is approximately 1, similar to a kernel-3 CNN

The architectural receptive field determines encoded interaction range, not the Hamiltonian alone. All architectures mirror the Hamiltonian locality, but through fundamentally different computational mechanisms.

**Verdict:** Strong support with nuance. All architectures encode the nearest-neighbor Hamiltonian term in their first hidden layer, but the *range* of encoded interactions differs: k=3 CNNs and the GNN's first layer encode exactly one bond, while BiRNNs encode several. The interaction range is determined by the architectural receptive field, not by the Hamiltonian. All architectures "mirror the Hamiltonian's circuit structure" but through different windows.

---

### Figure 7 - Local Hamiltonian-term decoding

![Local decoding](results/figures/fig7_local_decoding.png)

**What it measures.** Layer-by-layer $R^2$ for a ridge probe predicting the nearest-neighbor Ising term $z_i z_{i+1}$ from hidden activations. The dashed line is the baseline obtained from the raw input. If a hidden layer has $R^2$ above the dashed line, then that layer encodes the local Hamiltonian term more explicitly than the input representation alone.

---

1. Draw spin configurations from either the physical Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
or from the uniform distribution over spin configurations. f.ex: $\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1)
$

2. Feed the same sampled configurations into *one* trained model. f.ex. CNN-3layer-k3

3. Extract hidden activations layer by layer: $h^{(\ell)}(\mathbf{s}^{(k)})$, where $\ell$ = layer index and $k$ = sampled configuration index. For each layer, build the activation matrix:
$$
H^{(\ell)}
=
\begin{bmatrix}
h^{(\ell)}(\mathbf{s}^{(1)}) \\
h^{(\ell)}(\mathbf{s}^{(2)}) \\
\vdots \\
h^{(\ell)}(\mathbf{s}^{(M)})
\end{bmatrix}.
$$

4. For each sampled configuration, compute the nearest-neighbor target. With periodic boundary conditions, the final bond is: $z_N z_1.$
$$
y^{(k)}
=
z_i^{(k)}z_{i+1}^{(k)}.
$$

5. Train a ridge-regression probe at each layer. The probe is linear, so it tests whether the nearest-neighbor term is already linearly encoded in that layer.
$$
\hat y^{(k)}
=
W^\top h^{(\ell)}(\mathbf{s}^{(k)}) + b.
$$

6. Evaluate the probe using:
$$
R^2
=
1-
\frac{
\sum_k (y^{(k)}-\hat y^{(k)})^2
}{
\sum_k (y^{(k)}-\bar y)^2
}.
$$

Interpretation:
- $R^2=1$ means perfect decoding
- $R^2=0$ means no better than predicting the mean
- larger $R^2$ means the layer encodes $z_i z_{i+1}$ more clearly
- High $R^2$ in early layers means the architecture rapidly extracts the nearest-neighbor interaction.

---

**Numerical example**

Suppose for four sampled configurations, the true nearest-neighbor targets are:

$$
y=
\begin{bmatrix}
1\\
-1\\
1\\
1
\end{bmatrix}.
$$

At the raw input layer, the ridge probe predicts:

$$
\hat y_{\mathrm{in}}
=
\begin{bmatrix}
0.2\\
-0.1\\
0.3\\
0.1
\end{bmatrix}.
$$

Suppose this gives:

$$
R^2_{\mathrm{in}}=0.23.
$$

This corresponds to the dashed baseline in the figure.

Now suppose at the first hidden layer, the probe predicts:

$$
\hat y_{1}
=
\begin{bmatrix}
0.95\\
-0.90\\
0.98\\
0.88
\end{bmatrix}.
$$

If:

$$
\sum_k (y_k-\hat y_{1,k})^2 = 0.03
$$

and

$$
\sum_k (y_k-\bar y)^2 = 3.00,
$$

then:

$$
R^2_1
=
1-\frac{0.03}{3.00}
=
0.99.
$$

So the first hidden layer almost perfectly encodes the nearest-neighbor Hamiltonian term.

At a later output layer, suppose:

$$
R^2_{\mathrm{out}}=0.35.
$$

Then the local bond information is still present, but less explicitly linearly decodable after deeper nonlinear processing.

---
Each small plot corresponds to one model. The x-axis is the layer: $\text{in}, 1,2,\dots,\text{out}$ or for RNN/GNN: $\text{in}, r1,r2,\text{out}$ or $\text{in}, g1,g2,g3,\text{out}$. The y-axis is the decoding score: $R^2$ for predicting $z_i z_{i+1}.$ The top row uses physical samples from $|\psi|^2$. The bottom row uses uniform samples.

---

**Results.**

*Top row (physical samples):*

Every model shows the same qualitative pattern: $R^2$ jumps sharply at the first hidden layer (conv1, rnn1, gnn1), peaks there or at the second hidden layer, and then decays toward the readout. The first hidden layer achieves $R^2 \approx 0.95$–$1.0$ for both CNNs and BiRNN-2layer, and $R^2 \approx 0.85$ for BiRNN-1layer. GNN-3layer-local peaks at gnn1 ($R^2 \approx 0.85$) and decays monotonically through g2 ($\approx 0.75$) and g3 ($\approx 0.50$) to the readout ($\approx 0.35$). The readout layer has lower $R^2$ (0.3–0.5) across all models because it compresses the per-site features into a single scalar for the log-amplitude sum - the local correlation information has been "consumed" to build the wave function.

*Bottom row (uniform samples):*

Same pattern, but the input baseline drops to $R^2 \approx 0$. The first hidden layer achieves $R^2 \approx 0.95$–$1.0$ for the CNNs and BiRNN-2layer, $R^2 \approx 0.95$ for BiRNN-1layer, and $R^2 \approx 0.80$ for GNN-3layer-local. The GNN's slightly lower peak $R^2$ on uniform samples may reflect a less linearly separable encoding of the nearest-neighbor term, even though the information is clearly present. This is strong evidence that all models have hardwired the nearest-neighbor interaction, since the uniform samples remove any input-correlation confound.

**Verdict:** Strong support. The "first hidden layer = Hamiltonian term encoder" pattern is universal across all five architectures. This is the most direct mechanistic evidence that the networks learn to mirror the Hamiltonian's local interaction structure, and they all do it at the same computational depth (first layer).

---

### Figure 8 - Learned correlation functions

![Correlation functions](results/figures/fig8_correlation_functions.png)

**What it measures.** The two-point correlator $C(d)=\frac{1}{N}\sum_i\langle \sigma_i^z \sigma_{i+d}^z \rangle$ computed from MCMC samples drawn from each model's probability distribution $|\psi_\theta(\mathbf{s})|^2$. It measures how strongly spins separated by distance $d$ are correlated.

---

1. Draw spin configurations from the trained model: $\mathbf{s}^{(1)},\dots,\mathbf{s}^{(M)}\sim |\psi_\theta(\mathbf{s})|^2$ f.ex, for a 6-spin system: $\mathbf{s}^{(1)}=(+1,-1,+1,+1,-1,+1)$ and $\mathbf{s}^{(2)}=(-1,-1,+1,-1,+1,+1)$.
2. Choose a distance $d$. f.ex. $d=1$ measures nearest-neighbor correlations, while $d=5$ measures longer-range correlations.
3. For each sampled configuration, compute the site-averaged correlator With periodic boundary conditions $i+d \equiv i+d \mod N.$

$$
C^{(k)}(d)
=
\frac{1}{N}
\sum_i
s_i^{(k)}s_{i+d}^{(k)}.
$$

4. Average over all sampled configurations:

$$
C(d)
=
\frac{1}{MN}
\sum_{k=1}^M
\sum_i
s_i^{(k)}s_{i+d}^{(k)}.
$$

5. Repeat this for all distances:

$$
d=0,1,2,\dots,10.
$$

6. Plot $C(d)$ versus $d$ for each model.

Interpretation:
- slow decay means long-range order
- fast decay means weaker long-range correlations
- $C(0)=1$ because $s_i^2=1$
- differences at large $d$ reveal whether models learned the same physical state

**numerical example:**

Suppose a 6-spin configuration:

$$
\mathbf{s}^{(1)}
=
(+1,-1,+1,+1,-1,+1).
$$

For distance $d=1$, compute nearest-neighbor products:

$$
s_1s_2=(+1)(-1)=-1,
$$

$$
s_2s_3=(-1)(+1)=-1,
$$

$$
s_3s_4=(+1)(+1)=+1,
$$

$$
s_4s_5=(+1)(-1)=-1,
$$

$$
s_5s_6=(-1)(+1)=-1,
$$

$$
s_6s_1=(+1)(+1)=+1.
$$

Therefore:

$$
C^{(1)}(1)
=
\frac{1}{6}
(-1-1+1-1-1+1)
=
-\frac{2}{6}
=
-0.33.
$$

Now suppose a second sample gives:

$$
C^{(2)}(1)=0.67.
$$

Then the Monte Carlo estimate is:

$$
C(1)
=
\frac{C^{(1)}(1)+C^{(2)}(1)}{2}
=
\frac{-0.33+0.67}{2}
=
0.17.
$$

With many MCMC samples, this estimate converges to the model's learned nearest-neighbor correlation.

---

The x-axis is the distance $d$. The y-axis is the learned correlation $C(d)$. At distance $d=0$ all models have $C(0)=1$ because $\sigma_i^z\sigma_i^z=1$. For $d>0$, the decay of $C(d)$ shows how much long-range spin order each model has learned.

---

**Results.**

The correlation functions split into three groups:

- **CNN-3layer-k3 + CNN-5layer-k3** (blue, orange): nearly identical, with $C(1) \approx 0.66$ decaying to $C(10) \approx 0.52$.
- **BiRNN-1layer** (green): systematically higher and flatter, with $C(1) \approx 0.74$ and $C(d) \approx 0.74$ for all $d \geq 1$ - the correlator barely decays. This indicates the BiRNN-1layer wave function overestimates long-range order.
- **BiRNN-2layer** (red): similar to BiRNN-1layer but slightly lower, with $C(d) \approx 0.73$ at long range.
- **GNN-3layer-local** (purple): shows the steepest decay of all five models, with $C(1) \approx 0.64$ dropping to $C(10) \approx 0.45$. The GNN captures more of the critical algebraic decay structure than the other architectures, consistent with its slightly lower ground-state energy ($E/N \approx -1.273$).

The spread between the models ($\Delta C(10) \approx 0.29$, from GNN at 0.45 to BiRNN-1layer at 0.74) indicates that the BiRNN models have settled into slightly different solutions than the CNNs and GNN. The energy is dominated by the $d=0$ and $d=1$ terms, so $E/N$ can be similar even when $C(d>2)$ differs. The BiRNNs' flatter correlation functions suggest they capture less of the critical decay structure, consistent with their slightly higher energies ($E/N \approx -1.25$ vs $-1.27$). The GNN's steeper decay is the most physically realistic profile among the five models.

**Verdict:** concern, but may full under the category optimization. The CNN pair and GNN have converged to consistent states with appropriate decay, but the BiRNNs produce flatter correlations with higher long-range order. This is a known challenge for sequential architectures at criticality. The representation comparisons (Figs 1–7) remain valid - the high tangent CKA (0.95–1.00) shows the models occupy the same variational manifold even if they have not converged to exactly the same point on it.

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

The Gaussian (RBF) kernel is:

$$
K^{\mathrm{RBF}}_{ij}
=
\exp\!\left(
-\frac{\|x_i-x_j\|^2}{2\sigma^2}
\right)
$$

where:
- $x_i,x_j$ are feature vectors
- $\|x_i-x_j\|^2$ is the squared Euclidean distance
- $\sigma$ controls the neighborhood scale

Nearby points produce kernel values near 1, while distant points produce values near 0. Therefore, the RBF kernel converts distances into similarity scores.

---

**Gaussian kernel**

Suppose two tangent vectors are:

$$
x_i=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
x_j=
\begin{bmatrix}
1.1\\
2.2
\end{bmatrix}.
$$

Their squared distance is small:

$$
\|x_i-x_j\|^2
=
(1-1.1)^2+(2-2.2)^2
=
0.05.
$$

Then:

$$
K^{\mathrm{RBF}}_{ij}
=
\exp(-0.05/(2\sigma^2))
\approx 1.
$$

So for the kernel, these two configurations are represented similarly. Now consider a distant point:

$$
x_k=
\begin{bmatrix}
10\\
-5
\end{bmatrix}.
$$

Then:

$$
\|x_i-x_k\|^2
\gg 1,
$$

so:

$$
K^{\mathrm{RBF}}_{ik}
\approx 0.
$$

Therefore:
- nearby feature vectors become highly connected
- distant vectors become weakly connected

The Gaussian kernel therefore captures nonlinear neighborhood geometry.

---

1. Draw configurations from the Born distribution:
$
\mathbf{s}^{(1)}, \dots, \mathbf{s}^{(M)}
\sim |\psi(\mathbf{s})|^2
$,
for example:
$$
\mathbf{s}^{(1)} = (+1,-1,+1,+1,-1,+1)
\quad
\mathbf{s}^{(2)} = (-1,-1,+1,-1,+1,+1).
$$

2. Feed the same sampled configurations into two models.

3. Compute tangent vectors for each sampled configuration:
$$
J(\mathbf{s}^{(i)})
=
\frac{\partial \log\psi_\theta(\mathbf{s}^{(i)})}{\partial\theta}.
$$

4. Build tangent feature matrices for Model A and B, each row corresponds to one sampled spin configuration.
$$
X=
\begin{bmatrix}
J_A(\mathbf{s}^{(1)})\\
J_A(\mathbf{s}^{(2)})\\
\vdots
\end{bmatrix}
\in\mathbb{R}^{n\times d_A} \qquad
Y=
\begin{bmatrix}
J_B(\mathbf{s}^{(1)})\\
J_B(\mathbf{s}^{(2)})\\
\vdots
\end{bmatrix}
\in\mathbb{R}^{n\times d_B}
$$

5. Compute pairwise squared-distance matrices: $D^X_{ij}=\|x_i-x_j\|^2$ and $D^Y_{ij}=\|y_i-y_j\|^2$. These matrices measure how far apart configurations are in tangent-feature space.

**Numerical example for distances**

Suppose:

$$
x_1=
\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
x_2=
\begin{bmatrix}
2\\
3
\end{bmatrix}.
$$

Then:

$$
D^X_{12}
=
(1-2)^2+(2-3)^2
=
2.
$$

Repeating this for all pairs produces the full distance matrix:

$$
D^X=
\begin{bmatrix}
0 & 2 & 5\\
2 & 0 & 1\\
5 & 1 & 0
\end{bmatrix}.
$$

6. Choose the Gaussian bandwidth:

$$
\sigma^2
=
0.5\times\mathrm{median}(D).
$$

The median distance sets the characteristic neighborhood scale.

Interpretation:
- small $\sigma$ → only very close points are similar
- large $\sigma$ → many points become similar

7. Construct the RBF kernels:

$$
K^{\mathrm{RBF}}_{ij}
=
\exp\!\left(
-\frac{D_{ij}}{2\sigma^2}
\right).
$$

**Numerical example for the RBF kernel**

Suppose:

$$
D=
\begin{bmatrix}
0 & 2\\
2 & 0
\end{bmatrix},
\qquad
\sigma^2=1.
$$

Then:

$$
K^{\mathrm{RBF}}
=
\begin{bmatrix}
1 & e^{-1}\\
e^{-1} & 1
\end{bmatrix}
\approx
\begin{bmatrix}
1 & 0.37\\
0.37 & 1
\end{bmatrix}.
$$

Interpretation:
- diagonal entries are always 1
- off-diagonal entries measure nonlinear similarity

8. Center the kernels:

$$
\tilde K
=
HKH
$$

with centering matrix

$$
H
=
I-\frac1n\mathbf1\mathbf1^\top.
$$

This removes global offsets in similarity.

9. Compute RBF CKA:

$$
\mathrm{CKA}_{\mathrm{RBF}}
=
\frac{
\mathrm{tr}(\tilde K_1^{\mathrm{RBF}}
\tilde K_2^{\mathrm{RBF}})
}{
\sqrt{
\mathrm{tr}\!\left(
(\tilde K_1^{\mathrm{RBF}})^2
\right)
\;
\mathrm{tr}\!\left(
(\tilde K_2^{\mathrm{RBF}})^2
\right)
}
}.
$$

**Numerical example for RBF CKA**

Suppose after centering:

$$
\mathrm{tr}(\tilde K_1\tilde K_2)=5.2,
$$

$$
\mathrm{tr}(\tilde K_1^2)=6.0,
\qquad
\mathrm{tr}(\tilde K_2^2)=5.8.
$$

Then:

$$
\mathrm{CKA}_{\mathrm{RBF}}
=
\frac{5.2}{\sqrt{6.0\times5.8}}
\approx0.88.
$$

This means the two models preserve very similar nonlinear neighborhood structure in tangent space.

--- 

A comparison of linear CKA (left) and RBF (radial basis function) CKA (right) on the full tangent features. Linear CKA measures whether two representations span the same linear subspace. RBF CKA replaces the linear kernel with a Gaussian kernel:

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