## 2. The Jordan-Wigner Transformation

### 2.1 Motivation: Spins vs. Fermions

The key insight of Jordan and Wigner (1928) is that a single spin-$\frac{1}{2}$ is
algebraically identical to a single fermion mode. For a single site, define the raising
and lowering operators:

$$
\sigma_j^+ = \frac{\sigma_j^x + i\sigma_j^y}{2}, \qquad \sigma_j^- = \frac{\sigma_j^x - i\sigma_j^y}{2}
$$

These satisfy $\{\sigma_j^+, \sigma_j^-\} = \mathbb{1}$ and $(\sigma_j^+)^2 = (\sigma_j^-)^2 = 0$ — like fermionic creation and annihilation operators on a single site. Identify:

$$
|{\downarrow}\rangle \leftrightarrow |0\rangle, \qquad |{\uparrow}\rangle \leftrightarrow |1\rangle = f^\dagger |0\rangle
$$

However, there is a fundamental problem for *multiple* sites. Spin operators on different
sites **commute**:

$$
[\sigma_i^+, \sigma_j^-] = 0 \quad \text{for } i \neq j
$$

whereas fermionic operators must **anticommute**:

$$
\{c_i, c_j^\dagger\} = \delta_{ij}, \qquad \{c_i, c_j\} = \{c_i^\dagger, c_j^\dagger\} = 0
$$

Simply setting $c_j = \sigma_j^-$ fails because it gives the wrong statistics between
different sites.

### 2.2 The Transformation

Jordan and Wigner solved this by attaching a nonlocal "string" of operators — a phase
factor that tracks the parity of all sites to the left:

$$
\boxed{c_j = \left(\prod_{l < j} \sigma_l^z\right) \sigma_j^-, \qquad c_j^\dagger = \left(\prod_{l < j} \sigma_l^z\right) \sigma_j^+}
$$

The product $\prod_{l<j} \sigma_l^z$ is called the **Jordan-Wigner string**. Since
$\sigma_l^z$ has eigenvalues $\pm 1$, this string accumulates a factor of $(-1)$ for
each occupied (spin-up) site to the left of site $j$.

Equivalently, using the occupation number operator $n_l = \frac{1 + \sigma_l^z}{2}$
(where $n_l = 1$ for spin-up and $n_l = 0$ for spin-down):

$$
c_j = \left(\prod_{l < j} e^{i\pi n_l}\right) \sigma_j^- = \left(\prod_{l<j}(-\sigma_l^z)\right) \sigma_j^-
$$

### 2.3 Verification of Anticommutation Relations

Let us verify that these operators satisfy the correct fermionic algebra. We need three
relations.

**Case 1: Same site ($i = j$).**

$$
\{c_j, c_j^\dagger\} = c_j c_j^\dagger + c_j^\dagger c_j
$$

The JW strings cancel (since $(\sigma_l^z)^2 = \mathbb{1}$), leaving:

$$
\{c_j, c_j^\dagger\} = \sigma_j^- \sigma_j^+ + \sigma_j^+ \sigma_j^- = \{\sigma_j^-, \sigma_j^+\} = \mathbb{1} \quad \checkmark
$$

**Case 2: Different sites ($i < j$).**

$$
c_i c_j^\dagger = \underbrace{\left(\prod_{l<i}\sigma_l^z\right)}_{\text{string}_i} \sigma_i^- \underbrace{\left(\prod_{l<j}\sigma_l^z\right)}_{\text{string}_j} \sigma_j^+
$$

The string for $j$ includes site $i$, so we need to commute $\sigma_i^-$ past $\sigma_i^z$.
Using $\sigma_i^z \sigma_i^- = -\sigma_i^-$ (since lowering a spin flips the $z$-eigenvalue),
we pick up a factor of $(-1)$:

$$
\sigma_i^- \sigma_i^z = -\sigma_i^z \sigma_i^- \quad \Longrightarrow \quad \sigma_i^- \cdot (\text{string}_j \text{ containing } \sigma_i^z) = -(\text{string}_j) \cdot \sigma_i^-
$$

This sign difference between $c_i c_j^\dagger$ and $c_j^\dagger c_i$ gives:

$$
\{c_i, c_j^\dagger\} = 0 \quad \text{for } i \neq j \quad \checkmark
$$

**Case 3:** $\{c_i, c_j\} = 0$ follows by the same argument.

### 2.4 Inverse Transformation

The transformation can be inverted. We need expressions for the spin operators appearing
in the Hamiltonian:

**The $\sigma_i^z$ operator:** From the number operator $c_j^\dagger c_j = \sigma_j^+ \sigma_j^- = \frac{1 + \sigma_j^z}{2}$ (string cancels), we get:

$$
\boxed{\sigma_j^z = 2c_j^\dagger c_j - 1}
$$

**The $\sigma_i^x$ operator:** Using $\sigma_j^x = \sigma_j^+ + \sigma_j^-$ and the inverse
JW relations:

$$
\boxed{\sigma_j^x = \left(\prod_{l<j}\sigma_l^z\right)(c_j^\dagger + c_j) = \left(\prod_{l<j}(1 - 2c_l^\dagger c_l)\right)(c_j^\dagger + c_j)}
$$

### 2.5 Transforming the Hamiltonian

Now we substitute into the Hamiltonian $H = -J\sum_i \sigma_i^z\sigma_{i+1}^z - h\sum_i \sigma_i^x$.

**The transverse field term** $\sigma_j^x$:

Using the inverse transformation and the fact that the JW string on $\sigma_j^x$
cancels with the string from the fermion definition when acting within the Hamiltonian
(because $\sigma_j^x$ appears alone, not in a product with other spin operators at
different sites that would require tracking the string), we get:

$$
-h\sum_j \sigma_j^x = -h\sum_j (c_j^\dagger + c_j)\prod_{l<j}(1-2c_l^\dagger c_l)
$$

However, this expression looks nonlocal. The crucial simplification comes from the fact
that $\sigma_j^x = \sigma_j^+ + \sigma_j^-$, and in the JW representation,
$\sigma_j^{\pm} = (\prod_{l<j}\sigma_l^z)\,c_j^{(\dagger)}$, so
$\sigma_j^x = (\prod_{l<j}\sigma_l^z)(c_j^\dagger + c_j)$. But we also know
$\sigma_j^z = 2c_j^\dagger c_j - 1$. Substituting:

$$
-h\sigma_j^x = -h(c_j^\dagger + c_j)\prod_{l<j}(2c_l^\dagger c_l - 1) \cdot (-1)^{j-1}
$$

Actually, the cleanest route uses the fact that $\sigma_j^z = 1 - 2c_j^\dagger c_j$ directly in the Ising term.

**The Ising interaction term** $\sigma_i^z\sigma_{i+1}^z$:

This is simpler because $\sigma_j^z$ involves no JW string:

$$
\sigma_i^z \sigma_{i+1}^z = (2c_i^\dagger c_i - 1)(2c_{i+1}^\dagger c_{i+1} - 1) = (1 - 2c_i^\dagger c_i)(1 - 2c_{i+1}^\dagger c_{i+1})
$$

Expanding:

$$
\sigma_i^z \sigma_{i+1}^z = 1 - 2c_i^\dagger c_i - 2c_{i+1}^\dagger c_{i+1} + 4c_i^\dagger c_i c_{i+1}^\dagger c_{i+1}
$$

**The transverse field term** (revisited more carefully):

For the transverse field, the standard approach rewrites the full Hamiltonian using
$\sigma_j^x = \sigma_j^+ + \sigma_j^-$ and the products $\sigma_j^z\sigma_{j+1}^z$.

Using the identity $\sigma_j^z = -(\sigma_j^+ \sigma_j^- - \sigma_j^- \sigma_j^+) = -(c_j^\dagger c_j - c_j c_j^\dagger)$
and noting that for the nearest-neighbor product $\sigma_j^z\sigma_{j+1}^z$, the JW
strings between sites $j$ and $j+1$ are trivial (they only differ by the inclusion of
site $j$ in the string for $j+1$), the full Hamiltonian becomes:

$$
\boxed{H = -J\sum_{j=1}^{N-1}\left(c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j + c_j^\dagger c_{j+1}^\dagger + c_{j+1}c_j\right) - h\sum_{j=1}^{N}\left(2c_j^\dagger c_j - 1\right)}
$$

where we have written $g = h/J$ and absorbed factors appropriately. More precisely,
writing out the algebra step by step for the nearest-neighbor Ising coupling:

$$
\sigma_j^z\sigma_{j+1}^z = (2c_j^\dagger c_j - 1)(2c_{j+1}^\dagger c_{j+1} - 1)
$$

and the transverse field, using $\sigma_j^x = (\prod_{l<j}\sigma_l^z)(c_j + c_j^\dagger)$,
the key observation is that in the TFIM only single-site $\sigma^x$ terms and
nearest-neighbor $\sigma^z\sigma^z$ terms appear, so the nonlocal JW strings either
cancel completely ($\sigma^z$ terms) or reduce to local expressions
($\sigma^x$ terms act on individual sites).

The final fermionic Hamiltonian (ignoring the boundary term for now) is:

$$
H = -J \sum_{j=1}^{N-1} \left( c_j^\dagger - c_j \right)\left( c_{j+1}^\dagger + c_{j+1} \right) + h\sum_{j=1}^{N} \left(1 - 2c_j^\dagger c_j\right)
$$

Expanding the product in the Ising term:

$$
(c_j^\dagger - c_j)(c_{j+1}^\dagger + c_{j+1}) = c_j^\dagger c_{j+1}^\dagger + c_j^\dagger c_{j+1} - c_j c_{j+1}^\dagger - c_j c_{j+1}
$$

Using the anticommutation relation $c_j c_{j+1}^\dagger = -c_{j+1}^\dagger c_j$ (for $j \neq j+1$):

$$
= c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j + c_j^\dagger c_{j+1}^\dagger + c_{j+1} c_j
$$

This is a **quadratic** (free-fermion) Hamiltonian. It contains:

- **Hopping terms**: $c_j^\dagger c_{j+1} + c_{j+1}^\dagger c_j$ — fermions hop between neighboring sites
- **Pairing terms**: $c_j^\dagger c_{j+1}^\dagger + c_{j+1}c_j$ — pairs of fermions are created/annihilated (like a $p$-wave superconductor)
- **Chemical potential**: $c_j^\dagger c_j$ — from the transverse field

The Hamiltonian does *not* conserve total fermion number (due to the pairing terms), but
it *does* conserve fermion **parity** $(-1)^{\hat{N}_f}$, where $\hat{N}_f = \sum_j c_j^\dagger c_j$.

> **Key result:** The Jordan-Wigner transformation maps the 1D TFIM — an interacting
> spin model — to a model of **non-interacting** (free) fermions. This is what makes the
> model exactly solvable.

---

## 3. Fourier Transform to Momentum Space

Since we assume periodic boundary conditions and the system is translationally invariant,
we can diagonalize the hopping and pairing terms by Fourier transforming to momentum space.

Define:

$$
c_j = \frac{1}{\sqrt{N}} \sum_k e^{ikj}\, \tilde{c}_k, \qquad \tilde{c}_k = \frac{1}{\sqrt{N}} \sum_j e^{-ikj}\, c_j
$$

where the allowed momenta $k$ depend on the boundary conditions (discussed in detail in
Section 9). For now, we take $k_n = \frac{2\pi n}{N}$ with $n = 0, 1, \ldots, N-1$.

**Substituting into the hopping terms:**

$$
\sum_j c_j^\dagger c_{j+1} = \frac{1}{N}\sum_j \sum_{k,k'} e^{-ikj} e^{ik'(j+1)} \tilde{c}_k^\dagger \tilde{c}_{k'} = \sum_k e^{ik}\, \tilde{c}_k^\dagger \tilde{c}_k
$$

where we used $\frac{1}{N}\sum_j e^{i(k'-k)j} = \delta_{k,k'}$.

Similarly, $\sum_j c_{j+1}^\dagger c_j = \sum_k e^{-ik}\,\tilde{c}_k^\dagger \tilde{c}_k$.

**Substituting into the pairing terms:**

$$
\sum_j c_j^\dagger c_{j+1}^\dagger = \frac{1}{N}\sum_{k,k'} e^{ik'} \left(\sum_j e^{i(k+k')j}\right) \tilde{c}_k^\dagger \tilde{c}_{k'}^\dagger = \sum_k e^{ik}\, \tilde{c}_{-k}^\dagger \tilde{c}_k^\dagger
$$

(using $\delta_{k+k',0}$, so $k' = -k$), and $\sum_j c_{j+1}c_j = \sum_k e^{-ik}\, \tilde{c}_k \tilde{c}_{-k}$.

**The full Hamiltonian in momentum space:**

Collecting all terms (dropping the tildes for clarity and writing $c_k$ for the momentum-space operators):

$$
H = \sum_k \Big[ -\underbrace{(J\cos k + h)}_{\xi_k}\, (2c_k^\dagger c_k - 1) \;+\; iJ\sin k\, (c_{-k}^\dagger c_k^\dagger - c_k c_{-k}) \Big]
$$

or equivalently, grouping the $k$ and $-k$ modes into a $2\times 2$ matrix form
(Bogoliubov-de Gennes form). For each pair $(k, -k)$, define the Nambu spinor
$\Psi_k = (c_k, c_{-k}^\dagger)^T$. The Hamiltonian becomes:

$$
H = \sum_{k>0} \Psi_k^\dagger
\begin{pmatrix} -(J\cos k + h) & iJ\sin k \\ -iJ\sin k & (J\cos k + h) \end{pmatrix}
\Psi_k + \text{const.}
$$

Or, defining $a_k = J\cos k + h$ and $b_k = J\sin k$:

$$
H = \sum_{k>0}
\begin{pmatrix} c_k^\dagger & c_{-k} \end{pmatrix}
\begin{pmatrix} -a_k & ib_k \\ -ib_k & a_k \end{pmatrix}
\begin{pmatrix} c_k \\ c_{-k}^\dagger \end{pmatrix}
+ \text{const.}
$$

This $2\times 2$ structure — coupling $c_k$ with $c_{-k}^\dagger$ — is exactly what the
Bogoliubov transformation is designed to diagonalize.

---

## 4. The Bogoliubov Transformation

### 4.1 The Idea

The Hamiltonian couples operators at momenta $k$ and $-k$ through the pairing terms. To
diagonalize it, we introduce new **quasiparticle** operators $\eta_k$ that are linear
combinations of $c_k$ and $c_{-k}^\dagger$:

$$
\boxed{\eta_k = u_k\, c_k + v_k\, c_{-k}^\dagger}
$$

$$
\boxed{\eta_k^\dagger = u_k^*\, c_k^\dagger + v_k^*\, c_{-k}}
$$

where $u_k$ and $v_k$ are complex coefficients (the Bogoliubov coefficients). For $\eta_k$
to be a proper fermion, we require:

$$
\{\eta_k, \eta_k^\dagger\} = 1 \quad \Longrightarrow \quad |u_k|^2 + |v_k|^2 = 1
$$

### 4.2 Determining the Coefficients

We demand that the Hamiltonian takes the diagonal form:

$$
H = \sum_k \varepsilon_k \left(\eta_k^\dagger \eta_k - \frac{1}{2}\right) + E_{\text{const}}
$$

This is equivalent to requiring $[H, \eta_k] = -\varepsilon_k\, \eta_k$, i.e. $\eta_k$
lowers the energy by $\varepsilon_k$.

Computing $[H, \eta_k]$ using the BdG matrix $\mathcal{H}_k$ and equating with
$-\varepsilon_k\, \eta_k$, we get the eigenvalue equation:

$$
\begin{pmatrix} -a_k & ib_k \\ -ib_k & a_k \end{pmatrix}
\begin{pmatrix} u_k \\ v_k \end{pmatrix}
= -\varepsilon_k
\begin{pmatrix} u_k \\ v_k \end{pmatrix}
$$

The eigenvalues of this $2\times 2$ matrix are $\pm\varepsilon_k$ where:

$$
\varepsilon_k^2 = a_k^2 + b_k^2 = (J\cos k + h)^2 + (J\sin k)^2
$$

Expanding:

$$
\varepsilon_k^2 = J^2\cos^2 k + 2Jh\cos k + h^2 + J^2\sin^2 k = J^2 + h^2 + 2Jh\cos k
$$

$$
\boxed{\varepsilon_k = \sqrt{J^2 + h^2 + 2Jh\cos k}}
$$

Note: some conventions include a factor of 2, giving $\varepsilon_k = 2\sqrt{J^2 + h^2 + 2Jh\cos k}$,
depending on whether one uses Pauli matrices ($\sigma$) or spin-$\frac{1}{2}$ operators ($S = \sigma/2$).
In our convention (Pauli matrices in the Hamiltonian), the factor of 2 is present.

### 4.3 The Bogoliubov Angle

The coefficients $u_k$ and $v_k$ can be parametrized by a single angle $\theta_k$:

$$
u_k = \cos\frac{\theta_k}{2}, \qquad v_k = i\sin\frac{\theta_k}{2}
$$

where:

$$
\tan\theta_k = \frac{J\sin k}{J\cos k + h}
$$

The factor of $i$ in $v_k$ reflects the imaginary pairing amplitude in the BdG matrix.

---

## 5. The Ground State Energy

### 5.1 Quasiparticle Vacuum

After the Bogoliubov transformation, the Hamiltonian is:

$$
H = \sum_k \varepsilon_k \left(\eta_k^\dagger \eta_k - \frac{1}{2}\right)
$$

The ground state $|GS\rangle$ is the **quasiparticle vacuum**, defined by $\eta_k |GS\rangle = 0$ for all $k$. This is *not* the fermion vacuum — it is a complicated
superposition of states with different fermion numbers, connected by the pairing terms.

The ground state energy is obtained by setting all occupation numbers to zero:

$$
\boxed{E_0 = -\frac{1}{2}\sum_k \varepsilon_k = -\frac{1}{2}\sum_k \sqrt{J^2 + h^2 + 2Jh\cos k}}
$$

(with the factor of 2 from Pauli matrices: $E_0 = -\sum_k \sqrt{J^2 + h^2 + 2Jh\cos k}$.)

The energy per site is:

$$
\frac{E_0}{N} = -\frac{1}{N}\sum_{n=0}^{N-1} \sqrt{J^2 + h^2 + 2Jh\cos k_n}
$$

---

## 6. Thermodynamic Limit

### 6.1 From Sum to Integral

In the thermodynamic limit $N \to \infty$, the discrete sum over momenta becomes an
integral:

$$
\frac{1}{N}\sum_k f(k) \;\longrightarrow\; \frac{1}{2\pi}\int_0^{2\pi} f(k)\, dk
$$

Therefore:

$$
\frac{E_0}{N} = -\frac{1}{2\pi}\int_0^{2\pi} \sqrt{J^2 + h^2 + 2Jh\cos k}\, dk
$$

### 6.2 Reduction to the Complete Elliptic Integral

This integral can be evaluated in closed form using the **complete elliptic integral of
the second kind**:

$$
E(m) = \int_0^{\pi/2} \sqrt{1 - m\sin^2\phi}\, d\phi
$$

To reduce our integral to this form, factor out the maximum of $|J|$ and $h$:

$$
\frac{E_0}{N} = -\frac{1}{2\pi}\int_0^{2\pi} \sqrt{J^2 + h^2 + 2Jh\cos k}\, dk
$$

Let $a = \max(|J|, h)$ and $b = \min(|J|, h)$. Then inside the square root:

$$
J^2 + h^2 + 2Jh\cos k = a^2\left(1 + \frac{b^2}{a^2} + 2\frac{b}{a}\cos k\right) = a^2\left(1 + r^2 + 2r\cos k\right)
$$

where $r = b/a \leq 1$.

Using the identity $1 + r^2 + 2r\cos k = (1+r)^2 - 4r\sin^2(k/2)$ and substituting $\phi = k/2$:

$$
\frac{E_0}{N} = -\frac{a(1+r)}{\pi}\int_0^{\pi} \sqrt{1 - \frac{4r}{(1+r)^2}\sin^2\phi}\, d\phi
$$

Recognizing the elliptic integral with parameter $m = \frac{4r}{(1+r)^2} = \frac{4ab}{(a+b)^2}$:

$$
\frac{E_0}{N} = -\frac{2(a+b)}{\pi}\, E\!\left(\frac{4ab}{(a+b)^2}\right)
$$

Or equivalently, using a different standard form of the elliptic integral:

$$
\boxed{\frac{E_0}{N} = -\frac{2}{\pi}\,\max(|J|, h)\cdot E\!\left(\frac{\min(|J|, h)^2}{\max(|J|, h)^2}\right)}
$$

This is the exact ground state energy per site in the thermodynamic limit, expressed
in terms of a single special function.

---

## 7. Quantum Phase Transition

### 7.1 The Critical Point

The energy gap — the energy cost to create a single quasiparticle excitation — is:

$$
\Delta = \min_k \varepsilon_k
$$

The minimum of $\varepsilon_k = \sqrt{J^2 + h^2 + 2Jh\cos k}$ occurs at $k = \pi$
(for $J, h > 0$):

$$
\Delta = \sqrt{J^2 + h^2 - 2Jh} = |J - h|
$$

**The gap closes when $|J| = h$.** This is the quantum critical point, separating:

- **Ordered phase** ($|J| > h$): the ground state spontaneously breaks the $\mathbb{Z}_2$
  spin-flip symmetry; spins align along $z$ (ferro) or anti-align (antiferro)
- **Disordered phase** ($|J| < h$): the transverse field dominates; spins polarize
  along $x$ (paramagnetic)

### 7.2 Critical Behavior

At the critical point, the dispersion becomes linear at low energies:

$$
\varepsilon_k \approx J|k - \pi| \quad \text{(near } k = \pi \text{ at } J = h\text{)}
$$

This is the hallmark of a conformal field theory (CFT) with central charge $c = 1/2$
(the Ising CFT). The quantum phase transition is second-order and belongs to the
**2D classical Ising universality class** (via the quantum-classical mapping).



## 8. Boundary Conditions and Fermion Parity

### 8.1 The Boundary Term Problem

With periodic boundary conditions, the last term in the Ising sum connects site $N$
to site $1$:

$$
\sigma_N^z \sigma_1^z
$$

Under the Jordan-Wigner transformation, $\sigma_N^z = 2c_N^\dagger c_N - 1$ and
$\sigma_1^z = 2c_1^\dagger c_1 - 1$, but the full boundary term acquires a parity
operator:

$$
\sigma_N^z \sigma_1^z \to (2c_N^\dagger c_N - 1)(2c_1^\dagger c_1 - 1) \cdot (-1)^{\hat{N}_f}
$$

where $(-1)^{\hat{N}_f} = \prod_j(1-2c_j^\dagger c_j)$ is the total fermion parity.

This means the fermionic Hamiltonian has different boundary conditions depending on
whether the total fermion number is even or odd.

### 8.2 The Two Sectors

The Hilbert space splits into two sectors:

**Ramond (R) sector** — Even fermion parity, periodic fermion BC:

$$
k_n^{(R)} = \frac{2\pi n}{N}, \quad n = 0, 1, \ldots, N-1
$$

**Neveu-Schwarz (NS) sector** — Odd fermion parity, antiperiodic fermion BC:

$$
k_n^{(NS)} = \frac{2\pi(n + \tfrac{1}{2})}{N}, \quad n = 0, 1, \ldots, N-1
$$

The ground state energy is the minimum over both sectors:

$$
E_0 = \min\left(E_0^{(R)},\; E_0^{(NS)}\right)
$$

where $E_0^{(s)} = -\frac{1}{2}\sum_k \varepsilon_{k}^{(s)}$ for each sector $s$.

For even $N$ in the ordered phase ($|J| > h$), the ground state typically lies in the
NS sector. In the disordered phase, it lies in the R sector.
