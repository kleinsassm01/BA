from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import between_net_cka


def _ticks(names):
    n = len(names)
    idx = np.arange(0, n, max(1, n // 6))
    if len(idx) == 0 or idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    labels = []
    for name in np.asarray(names)[idx]:
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        labels.append(str(int(digits) + 1) if digits else str(name))
    return idx, labels


def _stats(mat):
    best = np.argmax(mat, axis=1)
    mono = float(np.mean(np.diff(best) >= 0)) if len(best) > 1 else 1.0
    mean_best = float(np.mean(np.max(mat, axis=1)))
    y = np.arange(mat.shape[0])
    ref = y * (mat.shape[1] - 1) / max(1, mat.shape[0] - 1)
    err = float(np.sqrt(np.mean((best - ref) ** 2))) if len(best) else 0.0
    return mean_best, mono, err


def make_figure5(
    archs: dict,
    pairs,
    out_dir: str,
    *,
    filename: str = "figure5-critical_2d_tfim.png",
    title: str = "Critical 2D TFIM",
):
    mats, stats = [], []
    for left, right in pairs:
        mat = between_net_cka(archs[left]["acts"], archs[left]["layers"], archs[right]["acts"], archs[right]["layers"])
        mats.append(mat)
        stats.append(_stats(mat))
    all_vals = np.concatenate([m.reshape(-1) for m in mats])
    floor = float(np.clip(np.percentile(all_vals, 3), 0, 0.97))
    ncol = 2
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.4 * nrow), squeeze=False)
    im = None
    for k, ((left, right), mat, st) in enumerate(zip(pairs, mats, stats)):
        r, c = divmod(k, ncol)
        ax = axes[r, c]
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap="inferno", vmin=floor, vmax=1.0)
        mean_best, mono, err = st
        ax.set_title(f"{left} vs {right}", fontsize=9)
        xt, xl = _ticks(archs[right]["layers"])
        yt, yl = _ticks(archs[left]["layers"])
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7)
        ax.set_xlabel(f"{right}\nlayer")
        ax.set_ylabel(f"{left}\nlayer")
        ax.text(
            0.02, 0.98,
            f"best={mean_best:.3f}\nmono={mono:.2f}\nerr={err:.1f}",
            transform=ax.transAxes, ha="left", va="top", color="white", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.38, edgecolor="none", pad=2),
        )
    for k in range(len(pairs), nrow * ncol):
        axes.flat[k].axis("off")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.90, hspace=0.42, wspace=0.34)
    if im is not None:
        cax = fig.add_axes([0.915, 0.14, 0.013, 0.72])
        fig.colorbar(im, cax=cax, label=f"linear CKA (floor {floor:.2f})")
    fig.suptitle(f"Cross-architecture CKA: {title}", fontsize=12)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)
