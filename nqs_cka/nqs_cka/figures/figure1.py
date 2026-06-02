from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from ..metrics import pca_scores, orthogonal_align
from ..physics import magnetization


def make_figure1(records, activations, samples, layer: str, out_dir: str):
    if len(records) < 2:
        return None
    scores_a = pca_scores(activations[0][layer], 2)
    scores_b = orthogonal_align(scores_a, pca_scores(activations[1][layer], 2))
    mags = magnetization(samples)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, scores, rec in zip(axes, [scores_a, scores_b], records[:2]):
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=mags, s=7, linewidths=0, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"seed {rec['seed']} · {layer}")
        ax.set_xlabel("aligned activation coordinate 1")
        ax.set_ylabel("aligned activation coordinate 2")
    fig.colorbar(sc, ax=axes, label="magnetization")
    fig.suptitle("Figure 1 (NQS): independent seeds align by the physical order parameter")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "figure1_pca.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)
