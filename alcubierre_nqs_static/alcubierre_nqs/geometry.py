from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config


@dataclass
class Problem:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    r: np.ndarray
    dx: float
    dy: float
    dz: float
    dVol: float
    Dx: np.ndarray
    Dy: np.ndarray
    Dz: np.ndarray
    f: np.ndarray
    V: np.ndarray
    V_grid: np.ndarray
    bubble_wall_grid: np.ndarray
    bubble_top_hat_grid: np.ndarray
    K: np.ndarray
    A: np.ndarray
    Omega2: np.ndarray

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def index_3d(ix: int, iy: int, iz: int, Ny: int, Nz: int) -> int:
    return (ix * Ny + iy) * Nz + iz


def derivative_matrices_periodic_3d(
    Nx: int, Ny: int, Nz: int, dx: float, dy: float, dz: float
):
    n = Nx * Ny * Nz
    Dx = np.zeros((n, n), dtype=np.float64)
    Dy = np.zeros((n, n), dtype=np.float64)
    Dz = np.zeros((n, n), dtype=np.float64)

    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                row = index_3d(ix, iy, iz, Ny, Nz)
                ip, im = (ix + 1) % Nx, (ix - 1) % Nx
                jp, jm = (iy + 1) % Ny, (iy - 1) % Ny
                kp, km = (iz + 1) % Nz, (iz - 1) % Nz

                Dx[row, index_3d(ip, iy, iz, Ny, Nz)] = 1.0 / (2.0 * dx)
                Dx[row, index_3d(im, iy, iz, Ny, Nz)] = -1.0 / (2.0 * dx)

                Dy[row, index_3d(ix, jp, iz, Ny, Nz)] = 1.0 / (2.0 * dy)
                Dy[row, index_3d(ix, jm, iz, Ny, Nz)] = -1.0 / (2.0 * dy)

                Dz[row, index_3d(ix, iy, kp, Ny, Nz)] = 1.0 / (2.0 * dz)
                Dz[row, index_3d(ix, iy, km, Ny, Nz)] = -1.0 / (2.0 * dz)

    return Dx, Dy, Dz


def bubble_profile_radial(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    denom = 2.0 * np.tanh(sigma * R)
    return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / denom


def sech2(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -50.0, 50.0)
    c = np.cosh(x_clip)
    return 1.0 / (c * c)


def bubble_profile_dfdr(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    # Radial derivative of the Alcubierre/Hiscock tanh shape function. f(r) see readme
    denom = 2.0 * np.tanh(sigma * R)
    return sigma * (
        sech2(sigma * (r + R)) - sech2(sigma * (r - R))
    ) / denom


def bubble_profile_surface_quantity(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cfg: Config,
    mode: str = "wall",
) -> np.ndarray:
    """
    mode="top_hat": plots f(r), the bubble shape/top-hat profile.
    mode="wall": plots v * partial_x f = v * (x/r) f'(r), giving the
    positive/negative expansion-contraction wall profile.
    """
    r = np.sqrt(X**2 + Y**2 + Z**2)
    f = bubble_profile_radial(r, cfg.R, cfg.sigma)

    if mode == "top_hat":
        return f

    if mode != "wall":
        raise ValueError(f"Unknown bubble_surface_mode={mode!r}; use 'wall' or 'top_hat'.")

    dfdr = bubble_profile_dfdr(r, cfg.R, cfg.sigma)
    x_over_r = np.divide(X, r, out=np.zeros_like(X), where=r > 1e-14)
    return cfg.v * x_over_r * dfdr


def make_problem(cfg: Config) -> Problem:
    x = np.linspace(-cfg.Lx / 2.0, cfg.Lx / 2.0, cfg.Nx, endpoint=False)
    y = np.linspace(-cfg.Ly / 2.0, cfg.Ly / 2.0, cfg.Ny, endpoint=False)
    z = np.linspace(-cfg.Lz / 2.0, cfg.Lz / 2.0, cfg.Nz, endpoint=False)
    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    dVol = dx * dy * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    f_grid = bubble_profile_radial(r, cfg.R, cfg.sigma)
    V_grid = cfg.v * f_grid

    Dx, Dy, Dz = derivative_matrices_periodic_3d(cfg.Nx, cfg.Ny, cfg.Nz, dx, dy, dz)
    n = cfg.Nx * cfg.Ny * cfg.Nz
    K = Dx.T @ Dx + Dy.T @ Dy + Dz.T @ Dz + (cfg.mass**2) * np.eye(n)
    V = V_grid.reshape(-1)
    A = V[:, None] * Dx
    Omega2 = K - A.T @ A

    bubble_wall_grid = bubble_profile_surface_quantity(X, Y, Z, cfg, mode="wall")
    bubble_top_hat_grid = bubble_profile_surface_quantity(X, Y, Z, cfg, mode="top_hat")

    return Problem(
        x=x,
        y=y,
        z=z,
        X=X,
        Y=Y,
        Z=Z,
        r=r,
        dx=dx,
        dy=dy,
        dz=dz,
        dVol=dVol,
        Dx=Dx,
        Dy=Dy,
        Dz=Dz,
        f=f_grid.reshape(-1),
        V=V,
        V_grid=V_grid,
        bubble_wall_grid=bubble_wall_grid,
        bubble_top_hat_grid=bubble_top_hat_grid,
        K=K,
        A=A,
        Omega2=Omega2,
    )
