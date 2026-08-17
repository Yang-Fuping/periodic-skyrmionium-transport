"""Classical magnetic textures and lattice-solid-angle topology.

Coordinates are lattice-site coordinates.  A finite sample uses the half-integer
center ``((L-1)/2, (W-1)/2)`` by default.  Periodic arrays are made by tiling a
normalized unit cell; magnetization vectors are never added together.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

import numpy as np


class TextureKind(str, Enum):
    UNIFORM = "uniform"
    SKYRMION_Q_PLUS = "skyrmion_q_plus"
    SKYRMION_Q_MINUS = "skyrmion_q_minus"
    SKYRMIONIUM_Q_ZERO = "skyrmionium_q_zero"


def _normalize(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=-1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("A zero magnetization vector cannot be normalized")
    return m / norm


def make_texture(
    kind: str | TextureKind,
    L: int,
    W: int,
    R: float,
    *,
    center: tuple[float, float] | None = None,
    helicity: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
    ellipticity: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Return an ``(L, W, 3)`` unit-vector field.

    ``skyrmion_q_plus/minus`` are labelled by the measured lattice topological
    charge under the orientation used in :func:`lattice_topological_charge`.
    The skyrmionium uses a smooth 0 -> 2π polar-angle profile, so its inner and
    outer rings carry opposite topological-charge density.
    """
    kind = TextureKind(kind)
    if L < 1 or W < 1 or R <= 0:
        raise ValueError("L, W and R must be positive")
    ex, ey = ellipticity
    if ex <= 0 or ey <= 0:
        raise ValueError("ellipticity axes must be positive")

    if center is None:
        center = ((L - 1) / 2.0, (W - 1) / 2.0)
    cx = center[0] + offset[0]
    cy = center[1] + offset[1]
    x, y = np.meshgrid(np.arange(L), np.arange(W), indexing="ij")
    dx = (x - cx) / ex
    dy = (y - cy) / ey
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)

    if kind is TextureKind.UNIFORM:
        m = np.zeros((L, W, 3), dtype=float)
        m[..., 2] = 1.0
        return m

    inside = r <= R
    if kind is TextureKind.SKYRMIONIUM_Q_ZERO:
        theta = np.where(inside, np.pi * (1.0 - np.cos(np.pi * r / R)), 0.0)
        winding = 1
    else:
        # Smooth π -> 0 profile with zero radial derivative at r=0 and r=R.
        theta = np.where(inside, 0.5 * np.pi * (1.0 + np.cos(np.pi * r / R)), 0.0)
        # With core -z and background +z, winding -1 gives Q=+1.
        winding = -1 if kind is TextureKind.SKYRMION_Q_PLUS else 1

    azimuth = winding * phi + helicity
    m = np.stack(
        (np.sin(theta) * np.cos(azimuth),
         np.sin(theta) * np.sin(azimuth),
         np.cos(theta)),
        axis=-1,
    )
    return _normalize(m)


def make_array_texture(
    kind: str | TextureKind,
    A: int,
    Nx: int,
    Ny: int,
    R: float,
    *,
    padding: tuple[int, int] = (0, 0),
    **texture_kwargs,
) -> np.ndarray:
    """Make a strict array, optionally surrounded by uniform ``+z`` padding.

    ``padding=(px, py)`` adds sites on both sides of the array along x and y.
    The default reproduces the historical strict tiling exactly.
    """
    if A < 2 or Nx < 1 or Ny < 1:
        raise ValueError("A >= 2 and Nx, Ny >= 1 are required")
    if 2 * R >= A:
        raise ValueError("Require 2R < A so the texture reaches uniform background at cell edges")
    px, py = padding
    if any(int(v) != v or v < 0 for v in (px, py)):
        raise ValueError("padding entries must be non-negative integers")
    px, py = int(px), int(py)
    cell = make_texture(kind, A, A, R, **texture_kwargs)
    tiled = np.tile(cell, (Nx, Ny, 1))
    if px == 0 and py == 0:
        return _normalize(tiled)
    out = np.zeros((Nx * A + 2 * px, Ny * A + 2 * py, 3), dtype=float)
    out[..., 2] = 1.0
    out[px:px + Nx * A, py:py + Ny * A] = tiled
    return _normalize(out)


def make_skyrmionium_wall(
    component: str,
    L: int,
    W: int,
    R: float,
    *,
    center: tuple[float, float] | None = None,
    helicity: float = 0.0,
) -> np.ndarray:
    """Return a continuous inner- or outer-wall counterfactual texture.

    The inner wall follows theta=0 to pi and ends in a ``-z`` background.  The
    outer wall starts from a ``-z`` core, follows theta=pi to 2pi and ends in a
    ``+z`` background.  They diagnose mechanisms; they are not additive pieces
    of the coherent skyrmionium scattering problem.
    """
    if component not in {"inner", "outer"}:
        raise ValueError("component must be 'inner' or 'outer'")
    if L < 1 or W < 1 or R <= 0:
        raise ValueError("L, W and R must be positive")
    if center is None:
        center = ((L - 1) / 2.0, (W - 1) / 2.0)
    x, y = np.meshgrid(np.arange(L), np.arange(W), indexing="ij")
    dx = x - center[0]
    dy = y - center[1]
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx)
    full_theta = np.pi * (1.0 - np.cos(np.pi * np.minimum(r, R) / R))
    if component == "inner":
        theta = np.where(r <= R / 2.0, full_theta, np.pi)
    else:
        theta = np.where(
            r < R / 2.0,
            np.pi,
            np.where(r <= R, full_theta, 2.0 * np.pi),
        )
    azimuth = phi + helicity
    m = np.stack(
        (np.sin(theta) * np.cos(azimuth),
         np.sin(theta) * np.sin(azimuth),
         np.cos(theta)),
        axis=-1,
    )
    return _normalize(m)


def make_skyrmionium_wall_array(
    component: str,
    A: int,
    Nx: int,
    Ny: int,
    R: float,
    *,
    padding: tuple[int, int] = (0, 0),
    helicity: float = 0.0,
) -> np.ndarray:
    """Tile a diagnostic skyrmionium wall with its matched background."""
    if A < 2 or Nx < 1 or Ny < 1 or 2 * R >= A:
        raise ValueError("Require A >= 2, Nx/Ny >= 1 and 2R < A")
    px, py = padding
    if any(int(v) != v or v < 0 for v in (px, py)):
        raise ValueError("padding entries must be non-negative integers")
    px, py = int(px), int(py)
    cell = make_skyrmionium_wall(component, A, A, R, helicity=helicity)
    tiled = np.tile(cell, (Nx, Ny, 1))
    background_z = -1.0 if component == "inner" else 1.0
    out = np.zeros((Nx * A + 2 * px, Ny * A + 2 * py, 3), dtype=float)
    out[..., 2] = background_z
    out[px:px + Nx * A, py:py + Ny * A] = tiled
    return _normalize(out)


def _solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    numerator = np.sum(a * np.cross(b, c), axis=-1)
    denominator = (
        1.0
        + np.sum(a * b, axis=-1)
        + np.sum(b * c, axis=-1)
        + np.sum(c * a, axis=-1)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def lattice_topological_charge(
    m: np.ndarray, *, periodic: bool = False
) -> tuple[float, np.ndarray]:
    """Return total Q and the charge per plaquette using two solid angles.

    For a magnetic unit cell use ``periodic=True``.  The density then has the
    same first two dimensions as the input.  For an open texture it has shape
    ``(L-1, W-1)``.
    """
    m = np.asarray(m, dtype=float)
    if m.ndim != 3 or m.shape[-1] != 3:
        raise ValueError("m must have shape (L, W, 3)")
    if periodic:
        m00 = m
        m10 = np.roll(m, -1, axis=0)
        m01 = np.roll(m, -1, axis=1)
        m11 = np.roll(m10, -1, axis=1)
    else:
        m00, m10 = m[:-1, :-1], m[1:, :-1]
        m11, m01 = m[1:, 1:], m[:-1, 1:]
    omega = _solid_angle(m00, m10, m11) + _solid_angle(m00, m11, m01)
    density = omega / (4.0 * np.pi)
    return float(np.sum(density)), density


def max_norm_error(m: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.norm(m, axis=-1) - 1.0)))


def topological_charge_profile_x(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return open-boundary plaquette charge density and its y-integrated profile."""
    _, density = lattice_topological_charge(m, periodic=False)
    return density, np.sum(density, axis=1)


def windowed_topological_charge(
    m: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Integrate lattice topological charge under an x-directed probe window."""
    L = np.asarray(m).shape[0]
    if start < 0 or stop <= start or stop > L:
        raise ValueError("Require 0 <= start < stop <= L")
    density, _ = topological_charge_profile_x(m)
    return float(np.sum(density[start:min(stop, L - 1), :]))
