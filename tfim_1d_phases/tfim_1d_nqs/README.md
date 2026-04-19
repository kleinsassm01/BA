# TFIM NQS — multi-N extension

Extends the original 1D TFIM NQS app with:

1. **Multi-N sweep** — run the full J-scan for every N in
   `MultiNConfig.N_values` (default `(10, 20, 40, 60, 80)`) and overlay the
   resulting curves.
2. **Critical-region zoom** — a fine J-scan around the quantum critical
   points at `J = ± h`, rendered as a dedicated "critical zoom" plot plus a
   Binder-cumulant plot. Together these visualize the second-order nature
   of the transition.
3. **Autocorrelation analysis** — both the per-step `τ_corr` from NetKet's
   built-in `Stats` object during training *and* a dedicated post-training
   MC chain that estimates the integrated autocorrelation time `τ_int` of
   the energy samples via the Sokal automated-windowing method.

## Entry points

Original (single-N, unchanged behaviour, autocorr off by default):

```bash
python -m tfim_ext.train       # train + save + plot
python -m tfim_ext.plot        # plot only, from saved dataset
```

New multi-N:

```bash
python -m tfim_ext.train_multi_N   # train for all N, save, plot
python -m tfim_ext.plot_multi_N    # re-plot only, from saved dataset
```

## What the new plots show

- **`multi_N_overlay.png`** — `<m²>(J)`, `<n²>(J)`, `E₀/N(J)` and the
  relative NQS error overlaid for every N. As N grows, the order-parameter
  curves sharpen and `E₀/N` approaches the exact `N → ∞` curve (elliptic
  integral).

- **`critical_zoom.png`** — three panels zooming into `J ≈ ± h`: `<n²>`
  around `J = -h`, `<m²>` around `J = +h`, and `-d²E₀/dJ²` (numerical,
  finite-difference). The third panel acts as an "energy susceptibility":
  its peak sharpens with N, a second-order fingerprint. A first-order
  transition would instead show a *finite* jump in `dE₀/dJ` (i.e. a
  δ-like divergence in `d²E₀/dJ²`).

- **`binder_cumulant.png`** — the Binder cumulant
  `U₄ = 1 - <m⁴>/(3 <m²>²)`. Curves for different N *cross at J_c* for a
  second-order transition — this crossing is a size-independent estimator
  of the critical coupling.

- **`tau_corr_vs_step.png`** — NetKet's per-step `τ_corr` during training,
  one subplot per N, colored by J. The plateau height near `|J| = h`
  reveals MCMC critical slowing down.

- **`tau_int_vs_J.png`** — left: `τ_int(J)` from the dedicated post-training
  chain (log scale), one curve per N. Right: the autocorrelation function
  `ρ(t)` at `J ≈ +h` for each N. Both tell the same story: near the
  critical point, chains decorrelate slowly, and the problem worsens with
  system size.

## Key design choices / notes

- **`<m⁴>` is built as `(Σ σᶻᵢ)⁴ / N⁴`** (operator composition) rather than
  as a literal O(N⁴) quadruple sum, so that N = 80 remains tractable.
- **Binder cumulant only uses `<mᵏ>` of the uniform magnetization**
  (ferro order parameter). For the antiferro side the analogous quantity
  built from the staggered magnetization would be cleaner, but the ferro
  Binder still crosses at `J = +h` and that's the transition we want to
  diagnose in detail.
- **`AutocorrConfig.enabled = False` by default on the single-N app** to
  preserve the original app's runtime. The multi-N entry point enables it.
- **Checkpointing**: `TFIMMultiNApp` writes `dataset.json.partial` after
  every completed N, so a crash mid-sweep does not lose earlier work.
- **Runtime**: a full N ∈ {10, 20, 40, 60, 80} sweep with coarse + zoom
  scans and autocorr analysis is an overnight job on a laptop. You can
  reduce the scope by passing `include_coarse_scan=False` to the multi-N
  app or setting a smaller `MultiNConfig.N_values`.

## Data model

`TrainingResult` now carries `N` per-result, so one JSON file can hold all
N values in a single dataset. `ExperimentDataset` provides
`results_for_N(N)` and `N_values()` accessors for the plotting code.
Legacy single-N JSON files still load correctly — `N` is backfilled from
metadata.
