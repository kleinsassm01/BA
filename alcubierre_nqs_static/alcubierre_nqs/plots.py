from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from .config import Config
from .geometry import Problem, bubble_profile_surface_quantity
from .observables import ObservableDefinition


def cell_extent(a: np.ndarray, b: np.ndarray, da: float, db: float):
    return [a[0] - 0.5 * da, a[-1] + 0.5 * da, b[0] - 0.5 * db, b[-1] + 0.5 * db]


def symmetric_vlim(arrays, percentile: float = 100.0, floor: float = 1e-14) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(np.abs(a).reshape(-1))
    if not vals:
        return floor
    joined = np.concatenate(vals)
    lim = float(np.max(joined)) if percentile >= 100.0 else float(np.percentile(joined, percentile))
    return max(lim, floor)


def positive_limits(arrays, percentile: float = 100.0, floor: float = 1e-14):
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a.reshape(-1))
    if not vals:
        return 0.0, floor
    joined = np.concatenate(vals)
    vmin = float(np.min(joined))
    vmax = float(np.max(joined)) if percentile >= 100.0 else float(np.percentile(joined, percentile))
    if np.isclose(vmax, vmin):
        vmax = vmin + floor
    return vmin, vmax


def central_z_slice(vec: np.ndarray, cfg: Config):
    arr = np.asarray(vec).reshape(cfg.Nx, cfg.Ny, cfg.Nz)
    k = cfg.Nz // 2
    return arr[:, :, k], k


def choose_plot_halfwidth(cfg: Config) -> float:
    if cfg.plot_halfwidth is not None:
        return float(cfg.plot_halfwidth)
    auto = max(1.4 * cfg.R, cfg.R + 3.0 / max(cfg.sigma, 1e-12))
    return min(auto, 0.48 * min(cfg.Lx, cfg.Ly))


def crop_xy(x: np.ndarray, y: np.ndarray, Z: np.ndarray, halfwidth: float):
    mask_x = np.abs(x) <= halfwidth
    mask_y = np.abs(y) <= halfwidth
    if not np.any(mask_x) or not np.any(mask_y):
        return x, y, Z
    return x[mask_x], y[mask_y], Z[np.ix_(mask_x, mask_y)]


def interp2d_uniform(x: np.ndarray, y: np.ndarray, Z: np.ndarray, factor: int):
    factor = max(1, int(factor))
    if factor == 1:
        return x, y, Z

    nx_new = (len(x) - 1) * factor + 1
    ny_new = (len(y) - 1) * factor + 1
    xi = np.linspace(x[0], x[-1], nx_new)
    yi = np.linspace(y[0], y[-1], ny_new)

    return xi, yi, interp2d_to_grid(x, y, Z, xi, yi)


def interp2d_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
) -> np.ndarray:
    tmp = np.empty((len(xi), len(y)), dtype=float)
    for j in range(len(y)):
        tmp[:, j] = np.interp(xi, x, Z[:, j])

    Zi = np.empty((len(xi), len(yi)), dtype=float)
    for i in range(len(xi)):
        Zi[i, :] = np.interp(yi, y, tmp[i, :])

    return Zi


def wall_radius_in_central_slice(cfg: Config, z_value: float) -> Optional[float]:
    if abs(z_value) >= cfg.R:
        return None
    return float(np.sqrt(cfg.R ** 2 - z_value ** 2))


def overlay_wall_circle_2d(ax, radius: Optional[float]):
    if radius is None:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), "k--", linewidth=0.9)


def overlay_wall_circle_3d(ax, x: np.ndarray, y: np.ndarray, H: np.ndarray, radius: Optional[float]):
    if radius is None:
        return
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    xc = radius * np.cos(theta)
    yc = radius * np.sin(theta)
    ix = np.clip(np.searchsorted(x, xc), 1, len(x) - 1)
    iy = np.clip(np.searchsorted(y, yc), 1, len(y) - 1)
    ix = np.where(np.abs(x[ix] - xc) < np.abs(x[ix - 1] - xc), ix, ix - 1)
    iy = np.where(np.abs(y[iy] - yc) < np.abs(y[iy - 1] - yc), iy, iy - 1)
    zc = H[ix, iy]
    ax.plot(xc, yc, zc, color="black", linestyle="--", linewidth=1.0, alpha=0.85)


def two_line_error_title(title: str) -> str:
    """Split error-plot titles into two lines so they do not get clipped."""
    if ": " in title:
        head, tail = title.split(": ", 1)
        return f"{head}:\n{tail}"
    return title


def save_energy_plot(path: Path, hist: np.ndarray, E_analytic: float):
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", linewidth=2.2, label="NQS training")
    ax.axhline(E_analytic, color="black", linestyle="--", linewidth=1.0, label="analytic")
    ax.set_xlabel("training step")
    ax.set_ylabel("energy")
    ax.set_title("3+1D NQS energy convergence to analytic Gaussian energy")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_compare_heatmap(
    path: Path,
    cfg: Config,
    prob: Problem,
    obs: ObservableDefinition,
):
    Zn, k = central_z_slice(obs.nqs, cfg)
    Za, _ = central_z_slice(obs.analytic, cfg)

    halfwidth = choose_plot_halfwidth(cfg)
    x, y, Zn = crop_xy(prob.x, prob.y, Zn, halfwidth)
    _, _, Za = crop_xy(prob.x, prob.y, Za, halfwidth)
    dx, dy = prob.dx, prob.dy
    radius = wall_radius_in_central_slice(cfg, prob.z[k])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    if obs.signed:
        vlim = symmetric_vlim([Zn, Za], percentile=cfg.heatmap_percentile)
        kwargs = {"norm": TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim), "cmap": "RdBu_r"}
    else:
        vmin, vmax = positive_limits([Zn, Za], percentile=cfg.heatmap_percentile)
        kwargs = {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}

    ims = []
    for ax, Z, name in zip(axes, [Zn, Za], ["NQS", "analytic"]):
        im = ax.imshow(
            Z.T,
            origin="lower",
            extent=cell_extent(x, y, dx, dy),
            aspect="equal",
            interpolation="nearest",
            **kwargs,
        )
        ims.append(im)
        overlay_wall_circle_2d(ax, radius if cfg.wall_overlay else None)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label(obs.label)
    fig.suptitle(f"3+1D NQS vs analytic: {obs.title} (zoomed central z-slice)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_heatmap(
    path: Path,
    cfg: Config,
    prob: Problem,
    obs: ObservableDefinition,
):
    Zd, k = central_z_slice(obs.nqs - obs.analytic, cfg)
    halfwidth = choose_plot_halfwidth(cfg)
    x, y, Zd = crop_xy(prob.x, prob.y, Zd, halfwidth)
    dx, dy = prob.dx, prob.dy
    radius = wall_radius_in_central_slice(cfg, prob.z[k])
    vlim = symmetric_vlim([Zd], percentile=cfg.heatmap_percentile)

    fig, ax = plt.subplots(figsize=(6.3, 5.5), constrained_layout=True)

    im = ax.imshow(
        Zd.T,
        origin="lower",
        extent=cell_extent(x, y, dx, dy),
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
    )

    overlay_wall_circle_2d(ax, radius if cfg.wall_overlay else None)

    ax.set_title(two_line_error_title(f"NQS - analytic error: {obs.title}"), fontsize=14, pad=10)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.tick_params(labelsize=11)

    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(f"error in {obs.label}", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_surface_height_scale(cfg: Config, arrays: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    if cfg.manual_surface_height_scale is not None:
        return float(cfg.manual_surface_height_scale)
    amp = symmetric_vlim(arrays, percentile=cfg.heatmap_percentile)
    if amp <= 0.0 or not np.isfinite(amp):
        return 1.0
    xy_span = max(float(x[-1] - x[0]), float(y[-1] - y[0]))
    return cfg.surface_target_relief * xy_span / amp


def compute_bubble_height_scale(cfg: Config, bubble: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    amp = symmetric_vlim([bubble], percentile=100.0)
    if amp <= 0.0 or not np.isfinite(amp):
        return 1.0
    xy_span = max(float(x[-1] - x[0]), float(y[-1] - y[0]))
    return cfg.bubble_surface_target_relief * xy_span / amp


def save_compare_surface(
    path: Path,
    cfg: Config,
    prob: Problem,
    obs: ObservableDefinition,
):
    Zn, k = central_z_slice(obs.nqs, cfg)
    Za, _ = central_z_slice(obs.analytic, cfg)

    halfwidth = choose_plot_halfwidth(cfg)
    x, y, Zn = crop_xy(prob.x, prob.y, Zn, halfwidth)
    _, _, Za = crop_xy(prob.x, prob.y, Za, halfwidth)

    xi, yi, Zni = interp2d_uniform(x, y, Zn, cfg.surface_interp_factor)
    _, _, Zai = interp2d_uniform(x, y, Za, cfg.surface_interp_factor)
    XI, YI = np.meshgrid(xi, yi, indexing="ij")

    if obs.signed:
        vlim = symmetric_vlim([Zni, Zai], percentile=cfg.heatmap_percentile)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        cmap = plt.cm.RdBu_r
    else:
        vmin, vmax = positive_limits([Zni, Zai], percentile=cfg.heatmap_percentile)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

    height_scale = compute_surface_height_scale(cfg, [Zni, Zai], xi, yi)
    Hn = height_scale * Zni
    Ha = height_scale * Zai
    zlim = symmetric_vlim([Hn, Ha], percentile=100.0)
    z_offset = -1.12 * zlim
    radius = wall_radius_in_central_slice(cfg, prob.z[k]) if cfg.wall_overlay else None

    fig = plt.figure(figsize=(13.8, 6.3), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, Zraw, H, name in [(ax1, Zni, Hn, "NQS"), (ax2, Zai, Ha, "analytic")]:
        if obs.signed:
            ax.plot_surface(XI, YI, np.zeros_like(H), color="0.8", alpha=0.10, linewidth=0.0, shade=False)

        ax.plot_surface(
            XI,
            YI,
            H,
            facecolors=cmap(norm(Zraw)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            edgecolor="none",
            antialiased=True,
            shade=True,
            alpha=0.98,
        )

        ax.contour(
            XI,
            YI,
            H,
            zdir="z",
            offset=z_offset,
            levels=cfg.surface_levels,
            cmap=cmap,
            norm=norm,
            linewidths=1.0,
        )

        if obs.signed and np.nanmin(Zraw) < 0.0 < np.nanmax(Zraw):
            ax.contour(XI, YI, Zraw, zdir="z", offset=z_offset, levels=[0.0], colors="black", linewidths=1.2)

        overlay_wall_circle_3d(ax, xi, yi, H, radius)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(f"scaled {obs.label}")
        ax.set_xlim(xi[0], xi[-1])
        ax.set_ylim(yi[0], yi[-1])
        ax.set_zlim(z_offset, zlim)
        ax.view_init(elev=cfg.surface_elev, azim=cfg.surface_azim)
        try:
            ax.set_box_aspect((xi[-1] - xi[0], yi[-1] - yi[0], 0.55 * max(xi[-1] - xi[0], yi[-1] - yi[0])))
        except Exception:
            pass

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.concatenate([Zni.reshape(-1), Zai.reshape(-1)]))
    cbar = fig.colorbar(mappable, ax=[ax1, ax2], shrink=0.78, pad=0.04)
    cbar.set_label(obs.label)
    fig.suptitle(
        f"3+1D NQS vs analytic: {obs.title}: zoomed/interpolated central z-slice height surface; "
        f"interp factor = {cfg.surface_interp_factor}, height scale = {height_scale:.4g}"
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_bubble_profile_compare_surface(
    path: Path,
    cfg: Config,
    prob: Problem,
    obs: ObservableDefinition,
):
    """
    High-visibility bubble plot.

    The 3D height is computed directly from the analytic Alcubierre bubble
    profile on a fine plot-only grid. The NQS / analytic observable is shown
    through surface color and projected contours.
    """
    Zn, k = central_z_slice(obs.nqs, cfg)
    Za, _ = central_z_slice(obs.analytic, cfg)

    halfwidth = choose_plot_halfwidth(cfg)
    x, y, Zn = crop_xy(prob.x, prob.y, Zn, halfwidth)
    _, _, Za = crop_xy(prob.x, prob.y, Za, halfwidth)

    n_plot = max(41, int(cfg.bubble_plot_points))
    xi = np.linspace(x[0], x[-1], n_plot)
    yi = np.linspace(y[0], y[-1], n_plot)

    Zni = interp2d_to_grid(x, y, Zn, xi, yi)
    Zai = interp2d_to_grid(x, y, Za, xi, yi)

    XI, YI = np.meshgrid(xi, yi, indexing="ij")
    ZI = np.full_like(XI, prob.z[k])

    Bi = bubble_profile_surface_quantity(XI, YI, ZI, cfg, mode=cfg.bubble_surface_mode)

    if obs.signed:
        vlim = symmetric_vlim([Zni, Zai], percentile=cfg.heatmap_percentile)
        data_norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        data_cmap = plt.cm.RdBu_r
    else:
        vmin, vmax = positive_limits([Zni, Zai], percentile=cfg.heatmap_percentile)
        data_norm = Normalize(vmin=vmin, vmax=vmax)
        data_cmap = plt.cm.viridis

    bubble_height_scale = compute_bubble_height_scale(cfg, Bi, xi, yi)
    H = bubble_height_scale * Bi
    hlim = symmetric_vlim([H], percentile=100.0)

    if cfg.bubble_surface_mode == "top_hat":
        zmin = float(np.nanmin(H)) - 0.20 * hlim
        zmax = float(np.nanmax(H)) + 0.20 * hlim
        z_offset = zmin
        zlabel = r"scaled bubble profile $f(r)$"
    else:
        zmin = -1.20 * hlim
        zmax = 1.20 * hlim
        z_offset = zmin
        zlabel = r"scaled bubble wall $v\,\partial_x f$"

    radius = wall_radius_in_central_slice(cfg, prob.z[k]) if cfg.wall_overlay else None
    wire_stride = max(1, int(cfg.bubble_wire_stride))

    fig = plt.figure(figsize=(14.5, 6.4), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for ax, Zraw, name in [
        (ax1, Zni, "NQS observable on bubble geometry"),
        (ax2, Zai, "analytic observable on bubble geometry"),
    ]:
        ax.plot_surface(XI, YI, np.zeros_like(H), color="0.78", alpha=0.28, linewidth=0.0, shade=False)

        ax.plot_surface(
            XI,
            YI,
            H,
            facecolors=data_cmap(data_norm(Zraw)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            edgecolor="none",
            antialiased=True,
            shade=True,
            alpha=0.98,
        )

        ax.plot_wireframe(
            XI,
            YI,
            H,
            rstride=wire_stride,
            cstride=wire_stride,
            color="0.12",
            linewidth=0.35,
            alpha=0.58,
        )

        ax.contour(
            XI,
            YI,
            Zraw,
            zdir="z",
            offset=z_offset,
            levels=cfg.bubble_observable_contours,
            cmap=data_cmap,
            norm=data_norm,
            linewidths=1.0,
            alpha=0.95,
        )

        if obs.signed and np.nanmin(Zraw) < 0.0 < np.nanmax(Zraw):
            ax.contour(XI, YI, Zraw, zdir="z", offset=z_offset, levels=[0.0], colors="black", linewidths=1.25)

        overlay_wall_circle_3d(ax, xi, yi, H, radius)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(zlabel)
        ax.set_xlim(xi[0], xi[-1])
        ax.set_ylim(yi[0], yi[-1])
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=cfg.surface_elev, azim=cfg.surface_azim)
        try:
            ax.set_box_aspect((xi[-1] - xi[0], yi[-1] - yi[0], 0.70 * max(xi[-1] - xi[0], yi[-1] - yi[0])))
        except Exception:
            pass

    mappable = plt.cm.ScalarMappable(norm=data_norm, cmap=data_cmap)
    mappable.set_array(np.concatenate([Zni.reshape(-1), Zai.reshape(-1)]))
    cbar = fig.colorbar(mappable, ax=[ax1, ax2], shrink=0.76, pad=0.04)
    cbar.set_label(obs.label)

    fig.suptitle(
        f"Alcubierre bubble geometry with NQS / analytic comparison\n"
        f"3+1D NQS vs analytic: {obs.title}; bubble mode = {cfg.bubble_surface_mode}; "
        f"observable color = {obs.label}; plot points = {n_plot}"
    )

    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_linecut_plot(
    path: Path,
    cfg: Config,
    prob: Problem,
    obs: ObservableDefinition,
):
    arr_nqs = np.asarray(obs.nqs).reshape(cfg.Nx, cfg.Ny, cfg.Nz)
    arr_an = np.asarray(obs.analytic).reshape(cfg.Nx, cfg.Ny, cfg.Nz)
    j = cfg.Ny // 2
    k = cfg.Nz // 2
    x = prob.x
    nqs_cut = arr_nqs[:, j, k]
    an_cut = arr_an[:, j, k]
    diff_cut = nqs_cut - an_cut

    halfwidth = choose_plot_halfwidth(cfg)
    mask = np.abs(x) <= halfwidth
    x = x[mask]
    nqs_cut = nqs_cut[mask]
    an_cut = an_cut[mask]
    diff_cut = diff_cut[mask]

    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)

    ax.plot(
        x,
        nqs_cut,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.6,
        marker="o",
        markersize=8,
        markerfacecolor="#1f77b4",
        markeredgecolor="white",
        markeredgewidth=0.9,
        label="NQS",
        zorder=3,
    )

    ax.plot(
        x,
        an_cut,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2.6,
        marker="s",
        markersize=8,
        markerfacecolor="#ff7f0e",
        markeredgecolor="white",
        markeredgewidth=0.9,
        label="analytic",
        zorder=3,
    )

    ax.plot(
        x,
        diff_cut,
        color="#2ca02c",
        linestyle="-.",
        linewidth=2.3,
        marker="^",
        markersize=8,
        markerfacecolor="#2ca02c",
        markeredgecolor="white",
        markeredgewidth=0.9,
        label="NQS - analytic",
        zorder=3,
    )

    ax.axhline(0.0, color="0.30", linewidth=1.0, alpha=0.9, zorder=1)
    ax.axvline(cfg.R, color="0.40", linestyle="--", linewidth=1.2, alpha=0.85, zorder=1)
    ax.axvline(-cfg.R, color="0.40", linestyle="--", linewidth=1.2, alpha=0.85, zorder=1)

    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("x on central line (y=0, z=0)", fontsize=13)
    ax.set_ylabel(obs.label, fontsize=13)
    ax.set_title(f"Central line cut: {obs.title}", fontsize=15, pad=10)
    ax.tick_params(labelsize=11)

    leg = ax.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("0.75")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    outdir: Path,
    cfg: Config,
    prob: Problem,
    derived: dict[str, ObservableDefinition],
    hist: np.ndarray | None,
    E_an: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if hist is not None and len(hist) > 1:
        save_energy_plot(outdir / "energy_training_vs_analytic.png", hist, E_an)

    for key, obs in derived.items():
        save_compare_heatmap(outdir / f"compare_{key}_zoom_heatmap.png", cfg, prob, obs)
        save_error_heatmap(outdir / f"error_{key}_zoom_heatmap.png", cfg, prob, obs)
        save_compare_surface(outdir / f"surface_{key}_zoom.png", cfg, prob, obs)

    if cfg.bubble_compare_key not in derived:
        print(
            f"\nWARNING: bubble_compare_key={cfg.bubble_compare_key!r} is not valid. "
            "Falling back to 'rho_sub_c'."
        )
        cfg.bubble_compare_key = "rho_sub_c"

    save_bubble_profile_compare_surface(
        outdir / f"bubble_profile_compare_{cfg.bubble_compare_key}.png",
        cfg,
        prob,
        derived[cfg.bubble_compare_key],
    )

    for key in ["rho_sub_c", "Ttx_sub_c", "q_var_sub"]:
        save_linecut_plot(outdir / f"linecut_{key}.png", cfg, prob, derived[key])
