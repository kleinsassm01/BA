from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ticks(n: int):
    idx = np.arange(0, n, max(1, n // 6))
    if len(idx) == 0 or idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return idx, [str(i + 1) for i in idx]


def _cka_floor(items):
    vals = []
    for item in items:
        mat = np.asarray(item["cka"], dtype=float)
        if mat.size:
            vals.append(mat.reshape(-1))
    return float(np.clip(np.percentile(np.concatenate(vals), 2), 0.0, 0.96)) if vals else 0.0


def _long_range_curve(item, min_distance: int):
    probe = item["probe"]
    delta = np.asarray(probe["delta_r2"], dtype=float)
    distances = np.asarray(probe["distances"], dtype=int)
    if delta.size == 0:
        return np.zeros(len(item["probe_layers"]), dtype=float)
    mask = distances >= int(min_distance)
    if not np.any(mask):
        mask = np.ones(len(distances), dtype=bool)
    return np.clip(np.nanmean(delta[mask], axis=0), 0.0, 1.0)


def _draw_grid_edges(ax, radius: int, shape: tuple[int, ...]):
    if len(shape) == 1:
        xs = np.arange(-radius, radius + 1)
        ax.plot(xs, np.zeros_like(xs), color="0.87", lw=1.0, zorder=0)
        return
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            for dx, dy in ((1, 0), (0, 1)):
                if -radius <= x + dx <= radius and -radius <= y + dy <= radius:
                    ax.plot([x, x + dx], [y, y + dy], color="0.87", lw=0.8, zorder=0)


def _reach_coordinates(shape: tuple[int, ...], radius: int):
    coords, shells = [], []
    if len(shape) == 1:
        for x in range(-radius, radius + 1):
            coords.append((x, 0))
            shells.append(abs(x))
    else:
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                coords.append((x, y))
                shells.append(abs(x) + abs(y))
    return np.asarray(coords, dtype=float), np.asarray(shells, dtype=int)


def _draw_reach(ax, item, shape, *, radius=4):
    probe = item["probe"]
    distances = np.asarray(probe["distances"], dtype=int)
    first = np.asarray(probe["first_layer_by_distance"], dtype=float)
    first_by_distance = {int(r): int(layer) for r, layer in zip(distances, first) if np.isfinite(layer)}

    coords, shells = _reach_coordinates(shape, radius)
    _draw_grid_edges(ax, radius, shape)

    colors = []
    for shell in shells:
        if shell == 0:
            colors.append((0.05, 0.05, 0.15, 1.0))
        elif shell in first_by_distance:
            colors.append(plt.cm.viridis(0.18 + 0.74 * min(shell, radius) / max(1, radius)))
        else:
            colors.append((0.86, 0.86, 0.86, 1.0))

    ax.scatter(
        coords[:, 0], coords[:, 1], c=colors,
        s=np.where(shells == 0, 150, 115), edgecolors="white", linewidths=0.6, zorder=2,
    )
    ax.scatter([0], [0], c="black", s=42, zorder=3)

    # Label each reached graph shell by the first layer where its local-shell
    # ΔR² crosses the configured threshold. This is a shell-level summary, not a
    # claim that every node is learned at that layer.
    for graph_distance, first_layer in first_by_distance.items():
        pts = coords[shells == graph_distance]
        if len(pts) == 0:
            continue
        order = np.argsort(-(pts[:, 1] * 100 - np.abs(pts[:, 0])))
        for x, y in pts[order[: min(4, len(pts))]]:
            ax.text(x, y, str(first_layer), ha="center", va="center", color="white", fontsize=6.2, zorder=4)

    ax.set_title("effective graph reach", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-radius - 0.6, radius + 0.6)
    ax.set_ylim(-radius - 0.6, radius + 0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure3(
    items,
    shape,
    out_dir: str,
    *,
    min_distance: int = 3,
    filename: str = "figure3-critical_2d_tfim.png",
    title: str = "Critical 2D TFIM",
):
    n = len(items)
    fig, axes = plt.subplots(
        4, n, figsize=(4.2 * n, 9.8),
        gridspec_kw={"height_ratios": [2.35, 1.0, 1.0, 1.55]},
    )
    if n == 1:
        axes = axes.reshape(4, 1)

    floor = _cka_floor(items)
    curves = [_long_range_curve(item, min_distance) for item in items]
    ymax_delta = max(0.25, min(1.0, 1.08 * max(float(np.max(c)) for c in curves)))
    im = None
    for col, (item, curve) in enumerate(zip(items, curves)):
        mat = np.asarray(item["cka"], dtype=float)
        im = axes[0, col].imshow(mat, origin="lower", aspect="equal", cmap="inferno", vmin=floor, vmax=1.0)
        axes[0, col].set_title(f"{item['mult']}x depth  E/N={item['e_per_site']:.3f}", fontsize=8.5)
        axes[0, col].set_xlabel("layer")
        if col == 0:
            axes[0, col].set_ylabel("layer")
        tick, labels = _ticks(mat.shape[0])
        axes[0, col].set_xticks(tick); axes[0, col].set_xticklabels(labels, fontsize=7)
        axes[0, col].set_yticks(tick); axes[0, col].set_yticklabels(labels, fontsize=7)

        x_delta = np.arange(1, len(curve) + 1)
        axes[1, col].plot(x_delta, curve, "-o", ms=3)
        if len(curve):
            axes[1, col].axvline(int(np.argmax(curve) + 1), ls="--", color="0.55", lw=0.9)
        axes[1, col].set_ylim(0, ymax_delta)
        axes[1, col].grid(alpha=0.3)
        axes[1, col].set_xlabel("layer")
        if col == 0:
            axes[1, col].set_ylabel(r"long-range $\Delta R^2$")

        logpsi = np.asarray(item.get("logpsi_r2", []), dtype=float)
        x_log = np.arange(1, len(logpsi) + 1)
        axes[2, col].plot(x_log, logpsi, "-o", ms=3)
        axes[2, col].set_ylim(0, 1.03)
        axes[2, col].grid(alpha=0.3)
        axes[2, col].set_xlabel("layer")
        if col == 0:
            label = r"probe $R^2(\log |\psi|)$"
            if item.get("logpsi_target") == "reference":
                label = r"probe $R^2(\log |\psi_{ref}|)$"
            axes[2, col].set_ylabel(label)

        _draw_reach(axes[3, col], item, shape)
        if col == 0:
            axes[3, col].set_ylabel("interaction graph", fontsize=8)

    fig.subplots_adjust(top=0.90, bottom=0.055, left=0.06, right=0.91, hspace=0.46, wspace=0.28)
    if im is not None:
        cax = fig.add_axes([0.925, 0.70, 0.012, 0.20])
        fig.colorbar(im, cax=cax, label=f"linear CKA (floor {floor:.2f})")
    fig.suptitle(f"Depth analysis: {title}", fontsize=12)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)
