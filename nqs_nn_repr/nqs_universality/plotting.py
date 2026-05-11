from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .metrics import (
    cka_matrix,
    linear_cka,
    model_pair_cka_matrix,
    ntk_alignment_matrix,
    ntk_eigenspectrum,
    procrustes_matrix,
    rbf_cka,
    saliency_covariance_by_distance,
)


def apply_plot_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 14,
    })


def _cmap(cfg: ExperimentConfig):
    cm = plt.get_cmap(cfg.cka_cmap).copy()
    cm.set_bad("0.85")
    return cm


def short_labels(layers):
    out = []
    for l in layers:
        if l == "input":
            out.append("in")
        elif l in ("readout", "readout_pre_sum"):
            out.append("out")
        elif l.startswith("conv"):
            out.append(l.replace("conv", ""))
        elif l.startswith("rnn"):
            out.append(l.replace("rnn", "r"))
        elif l.startswith("gnn"):
            out.append(l.replace("gnn", "g"))
        elif l == "embed":
            out.append("emb")
        else:
            out.append(l)
    return out


def save_fig(fig, name: str, cfg: ExperimentConfig):
    d = cfg.out_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)

    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    if cfg.save_pdf:
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    print(f"  saved {name}")


def _energy_label(meta, name):
    try:
        E = meta["models"][name]["final_energy"]["mean"]
        return f"E/N={E / meta['hamiltonian']['N']:.4f}"
    except Exception:
        return ""


def _annotate_matrix(ax, M, vmin: float, vmax: float):
    threshold = vmin + 0.6 * (vmax - vmin)
    M_arr = np.asarray(M)
    for i in range(M_arr.shape[0]):
        for j in range(M_arr.shape[1]):
            v = float(M_arr[i, j])
            if not np.isfinite(v):
                continue
            c = "black" if v > threshold else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=c)


def plot_cross_activation_cka(act_cka, metadata, cfg: ExperimentConfig):
    pairs = list(itertools.combinations(metadata["model_order"], 2))
    nc = 3
    nr = int(np.ceil(len(pairs) / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(5.1 * nc + 0.8, 4.8 * nr), squeeze=False)
    flat = axes.flatten()
    last_im = None

    for ax, (a, b) in zip(flat, pairs):
        la = metadata["models"][a]["activation_layer_order"]
        lb = metadata["models"][b]["activation_layer_order"]
        M, ly, lx = cka_matrix(act_cka[a], act_cka[b], la, lb)
        M = np.ma.masked_invalid(M)

        last_im = ax.imshow(
            M,
            vmin=cfg.cka_vmin,
            vmax=cfg.cka_vmax,
            cmap=_cmap(cfg),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"{a}\nvs {b}", fontsize=8)
        ax.set_xlabel(b, fontsize=8)
        ax.set_ylabel(a, fontsize=8)
        ax.set_xticks(range(len(lx)))
        ax.set_xticklabels(short_labels(lx))
        ax.set_yticks(range(len(ly)))
        ax.set_yticklabels(short_labels(ly))

    for ax in flat[len(pairs):]:
        ax.axis("off")

    fig.subplots_adjust(hspace=0.75, wspace=0.45, right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label("Activation CKA")
    fig.suptitle("Fig 1 - Cross-architecture activation CKA", y=1.005)
    save_fig(fig, "fig1_cross_activation_cka", cfg)
    plt.close(fig)


def plot_tangent_and_ntk(full_tangents, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    M_cka = model_pair_cka_matrix(full_tangents, mo)
    M_ntk = ntk_alignment_matrix(full_tangents, mo)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, M, title, label in [
        (ax1, M_cka, "Full tangent-space CKA", "CKA"),
        (ax2, M_ntk, "NTK kernel alignment", "KA"),
    ]:
        M_plot = np.ma.masked_invalid(M)
        im = ax.imshow(
            M_plot,
            vmin=cfg.sim_vmin,
            vmax=cfg.sim_vmax,
            cmap=_cmap(cfg),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        ax.set_xticks(range(len(mo)))
        ax.set_xticklabels(mo, rotation=30, ha="right")
        ax.set_yticks(range(len(mo)))
        ax.set_yticklabels(mo)
        _annotate_matrix(ax, np.asarray(M_plot), cfg.sim_vmin, cfg.sim_vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)

    fig.suptitle("Fig 2 - Functional similarity: tangent space and NTK")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig2_tangent_ntk_similarity", cfg)
    plt.close(fig)


def plot_ntk_eigenspectrum(full_tangents, metadata, cfg: ExperimentConfig):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in metadata["model_order"]:
        evals = ntk_eigenspectrum(full_tangents[name], cfg.ntk_top_k)
        evals = evals[evals > 0]
        ax.semilogy(
            np.arange(1, len(evals) + 1),
            evals,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=name,
        )

    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("NTK eigenvalue")
    ax.set_title("Fig 3 - NTK eigenspectrum")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, "fig3_ntk_eigenspectrum", cfg)
    plt.close(fig)


def plot_procrustes(activations_physical, full_tangents, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    readout_feats = {}
    for name in mo:
        ro = activations_physical[name]["readout_pre_sum"]
        readout_feats[name] = ro.reshape(ro.shape[0], -1)

    M_act = procrustes_matrix(readout_feats, mo)
    M_tan = procrustes_matrix(full_tangents, mo)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, M, title in [
        (ax1, M_act, "Readout activations"),
        (ax2, M_tan, "Full tangent features"),
    ]:
        M_plot = np.ma.masked_invalid(M)
        valid_vals = M[np.isfinite(M)]
        vmax_val = valid_vals.max() * 1.05 + 1e-6 if len(valid_vals) > 0 else 1.0
        proc_cmap = plt.get_cmap("magma_r").copy()
        proc_cmap.set_bad("0.85")

        im = ax.imshow(M_plot, vmin=0, vmax=vmax_val, cmap=proc_cmap, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        ax.set_xticks(range(len(mo)))
        ax.set_xticklabels(mo, rotation=30, ha="right")
        ax.set_yticks(range(len(mo)))
        ax.set_yticklabels(mo)

        threshold = vmax_val * 0.4
        for i in range(len(mo)):
            for j in range(len(mo)):
                v = M[i, j]
                txt = f"{v:.3f}" if np.isfinite(v) else "nan"
                c = "white" if (np.isfinite(v) and v > threshold) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=c)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Procrustes dist.")

    fig.suptitle("Fig 4 - Orthogonal Procrustes distance")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig4_procrustes_distance", cfg)
    plt.close(fig)


def plot_saliency(saliency_all, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    n_m = len(mo)
    N_sites = metadata["hamiltonian"]["N"]
    hm_rows = int(np.ceil(n_m / 2))

    fig = plt.figure(figsize=(14, 4.5 * (1 + 0.7 + hm_rows)))
    gs = fig.add_gridspec(
        1 + 1 + hm_rows,
        4,
        height_ratios=[1.0, 0.9] + [1.0] * hm_rows,
        hspace=0.55,
        wspace=0.45,
    )

    ax_prof = fig.add_subplot(gs[0, 0:2])
    for name in mo:
        prof = np.mean(np.abs(saliency_all[name]), axis=0)
        ax_prof.plot(range(N_sites), prof, marker=".", markersize=3, label=name)
    ax_prof.set_xlabel("Site i")
    ax_prof.set_ylabel("mean absolute input saliency")
    ax_prof.set_title("(a) Mean saliency profile")
    ax_prof.legend(fontsize=7, frameon=False, ncol=2)

    ax_cov = fig.add_subplot(gs[0, 2:4])
    for name in mo:
        cv = saliency_covariance_by_distance(saliency_all[name], metadata["hamiltonian"]["pbc"])
        ax_cov.plot(range(len(cv)), cv, marker="o", markersize=3, label=name)
    ax_cov.axvline(1, color="grey", linestyle="--", linewidth=0.8)
    ax_cov.set_xlabel("Distance d")
    ax_cov.set_ylabel("saliency covariance")
    ax_cov.set_title("(b) Saliency covariance vs distance")
    ax_cov.legend(fontsize=7, frameon=False, ncol=2)

    ax_cka = fig.add_subplot(gs[1, 1:3])
    sal_flat = {n: saliency_all[n] for n in mo}
    M_sal = model_pair_cka_matrix(sal_flat, mo)
    M_sal = np.ma.masked_invalid(M_sal)
    im = ax_cka.imshow(
        M_sal,
        vmin=cfg.sim_vmin,
        vmax=cfg.sim_vmax,
        cmap=_cmap(cfg),
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    ax_cka.set_xticks(range(n_m))
    ax_cka.set_xticklabels(mo, rotation=35, ha="right", fontsize=7)
    ax_cka.set_yticks(range(n_m))
    ax_cka.set_yticklabels(mo, fontsize=7)
    ax_cka.set_title("(c) Saliency CKA", fontsize=10)
    _annotate_matrix(ax_cka, np.asarray(M_sal), cfg.sim_vmin, cfg.sim_vmax)
    fig.colorbar(im, ax=ax_cka, fraction=0.046, pad=0.04)

    panel_labels = [chr(ord("d") + i) for i in range(n_m)]
    for idx, name in enumerate(mo):
        row = idx // 2
        col = (idx % 2) * 2
        ax = fig.add_subplot(gs[2 + row, col:col + 2])
        sal = saliency_all[name].astype(np.float64)
        sal -= sal.mean(0, keepdims=True)
        C = (sal.T @ sal) / sal.shape[0]
        vabs = np.abs(C).max()
        im2 = ax.imshow(C, cmap="RdBu_r", origin="lower", aspect="equal", vmin=-vabs, vmax=vabs)
        ax.set_title(f"({panel_labels[idx]}) {name}", fontsize=9)
        ax.set_xlabel("Site")
        ax.set_ylabel("Site")
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Fig 5 - Input-gradient saliency analysis", y=1.01)
    save_fig(fig, "fig5_saliency_analysis", cfg)
    plt.close(fig)


def plot_multidistance_decoding(multidist, metadata, cfg: ExperimentConfig):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name in metadata["model_order"]:
        r2 = multidist["physical"][name]["r2_vs_d"]
        ds = np.arange(1, len(r2) + 1)
        ax1.plot(ds, r2, marker="o", markersize=4, linewidth=1.5, label=name)
    ax1.set_xlabel("Distance d")
    ax1.set_ylabel("R2 for z_i z_{i+d}")
    ax1.set_title("(a) Physical samples")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylim(-0.05, 1.05)

    for name in metadata["model_order"]:
        r2 = multidist["uniform"][name]["r2_vs_d"]
        ds = np.arange(1, len(r2) + 1)
        ax2.plot(ds, r2, marker="o", markersize=4, linewidth=1.5, label=name)
    ax2.set_xlabel("Distance d")
    ax2.set_ylabel("R2 for z_i z_{i+d}")
    ax2.set_title("(b) Uniform samples")
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Fig 6 - Multi-distance decoding from first hidden layer")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig6_multidistance_decoding", cfg)
    plt.close(fig)


def plot_local_decoding(local_all, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    n_m = len(mo)
    fig, axes = plt.subplots(2, n_m, figsize=(4 * n_m, 7), squeeze=False)

    for row, (skey, title_tag) in enumerate([("physical", "physical"), ("uniform", "uniform")]):
        dec = local_all[skey]
        baseline_r2 = dec["_input_baseline"]["r2"]

        for col, name in enumerate(mo):
            layers = metadata["models"][name]["activation_layer_order"]
            xs = np.arange(1, len(layers) + 1)
            r2 = np.array([dec[name][l]["r2"] for l in layers])
            ax = axes[row, col]
            valid = np.isfinite(r2)
            if valid.any():
                ax.plot(xs[valid], r2[valid], "o-", linewidth=1.8, markersize=3.5)
            ax.axhline(baseline_r2, ls="--", lw=1, color="steelblue", alpha=0.6)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(xs)
            ax.set_xticklabels(short_labels(layers))
            ax.set_xlabel("Layer")
            if col == 0:
                ax.set_ylabel(f"R2 ({title_tag})")
            if row == 0:
                lbl = name
                e = _energy_label(metadata, name)
                if e:
                    lbl += f"\n{e}"
                ax.set_title(lbl, fontsize=9)

    fig.suptitle(
        "Fig 7 - Local Hamiltonian-term decoding: R2 for z_i z_{i+1}\n"
        "top: physical samples, bottom: uniform samples",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig7_local_decoding", cfg)
    plt.close(fig)


def plot_correlation_functions(correlations, metadata, cfg: ExperimentConfig):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in metadata["model_order"]:
        C = correlations[name]
        ds = np.arange(len(C))
        ax.plot(ds, C, marker="o", markersize=4, linewidth=1.5, label=name)
    ax.set_xlabel("Distance d")
    ax.set_ylabel("C(d)")
    ax.set_title("Fig 8 - Learned two-point correlation functions")
    ax.legend(frameon=False, fontsize=9)
    save_fig(fig, "fig8_correlation_functions", cfg)
    plt.close(fig)


def plot_within_model_cka(activations_physical, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    n_m = len(mo)
    fig, axes = plt.subplots(1, n_m, figsize=(4.5 * n_m, 4.2), squeeze=False)
    axes = axes[0]

    for idx, name in enumerate(mo):
        layers = metadata["models"][name]["activation_layer_order"]
        n_l = len(layers)
        M = np.ones((n_l, n_l), dtype=np.float64)
        flat = {
            l: activations_physical[name][l].reshape(activations_physical[name][l].shape[0], -1)
            for l in layers
        }

        for i in range(n_l):
            for j in range(i + 1, n_l):
                v = linear_cka(flat[layers[i]], flat[layers[j]])
                M[i, j] = v
                M[j, i] = v

        ax = axes[idx]
        M_plot = np.ma.masked_invalid(M)
        im = ax.imshow(
            M_plot,
            vmin=cfg.cka_vmin,
            vmax=cfg.cka_vmax,
            cmap=_cmap(cfg),
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        sl = short_labels(layers)
        ax.set_xticks(range(n_l))
        ax.set_xticklabels(sl, fontsize=8)
        ax.set_yticks(range(n_l))
        ax.set_yticklabels(sl, fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")

        threshold = cfg.cka_vmin + 0.6 * (cfg.cka_vmax - cfg.cka_vmin)
        for i in range(n_l):
            for j in range(n_l):
                v = float(M[i, j])
                if np.isfinite(v):
                    c = "black" if v > threshold else "white"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=c)

    fig.subplots_adjust(right=0.88, wspace=0.4)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("CKA")
    fig.suptitle("Fig 9 - Within-model layer CKA")
    save_fig(fig, "fig9_within_model_cka", cfg)
    plt.close(fig)


def plot_rbf_cka(full_tangents, activations_physical, metadata, cfg: ExperimentConfig):
    mo = metadata["model_order"]
    n = len(mo)
    M_lin = np.ones((n, n))
    M_rbf = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            lv = linear_cka(full_tangents[mo[i]], full_tangents[mo[j]])
            rv = rbf_cka(full_tangents[mo[i]], full_tangents[mo[j]])
            M_lin[i, j] = lv
            M_lin[j, i] = lv
            M_rbf[i, j] = rv
            M_rbf[j, i] = rv

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, M, title, label in [
        (ax1, M_lin, "Linear CKA: tangent features", "CKA"),
        (ax2, M_rbf, "RBF CKA: tangent features", "CKA"),
    ]:
        M_plot = np.ma.masked_invalid(M)
        im = ax.imshow(
            M_plot,
            vmin=cfg.sim_vmin,
            vmax=cfg.sim_vmax,
            cmap=_cmap(cfg),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        ax.set_xticks(range(n))
        ax.set_xticklabels(mo, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(mo)
        _annotate_matrix(ax, np.asarray(M_plot), cfg.sim_vmin, cfg.sim_vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(label)

    fig.suptitle("Fig 10 - Linear vs RBF CKA")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "fig10_linear_vs_rbf_cka", cfg)
    plt.close(fig)


def make_all_plots(
    cfg: ExperimentConfig,
    metadata,
    act_cka,
    activations_physical,
    full_tangents,
    layer_tangents,
    saliency_all,
    local_all,
    multidist,
    correlations,
):
    apply_plot_style()
    print("\nGenerating figures ...")
    plot_cross_activation_cka(act_cka, metadata, cfg)
    plot_tangent_and_ntk(full_tangents, metadata, cfg)
    plot_ntk_eigenspectrum(full_tangents, metadata, cfg)
    plot_procrustes(activations_physical, full_tangents, metadata, cfg)
    plot_saliency(saliency_all, metadata, cfg)
    plot_multidistance_decoding(multidist, metadata, cfg)
    plot_local_decoding(local_all, metadata, cfg)
    plot_correlation_functions(correlations, metadata, cfg)
    plot_within_model_cka(activations_physical, metadata, cfg)
    plot_rbf_cka(full_tangents, activations_physical, metadata, cfg)
    print(f"\nAll figures -> {(cfg.out_dir / 'figures').resolve()}")
