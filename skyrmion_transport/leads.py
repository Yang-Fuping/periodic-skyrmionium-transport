"""Uniform ferromagnetic lead surface Green functions."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .model import S0, SZ


def uniform_lead_channel_summary(
    energy: float,
    width: int,
    J: float,
    t: float,
) -> dict[str, float | int]:
    """Return analytic propagating-channel counts for a z-polarized strip.

    ``J`` is the signed exchange field in ``-J sigma_z``.  Positive and
    negative values therefore describe opposite lead magnetizations without
    changing the global spin-z labels.  The open-y transverse eigenenergies are
    exact for the lead used by :func:`uniform_lead_self_energy`.
    """
    if width < 1 or t <= 0:
        raise ValueError("width and t must be positive")
    transverse = -2.0 * t * np.cos(
        np.arange(1, width + 1) * np.pi / (width + 1)
    )
    n_up = int(np.count_nonzero(np.abs(energy - (transverse - J)) < 2.0 * t))
    n_down = int(np.count_nonzero(np.abs(energy - (transverse + J)) < 2.0 * t))
    total = n_up + n_down
    polarization = (n_up - n_down) / total if total else 0.0
    return {
        "n_up": n_up,
        "n_down": n_down,
        "n_total": total,
        "polarization": float(polarization),
    }


def uniform_lead_slice(width: int, J: float, t: float, potential: float = 0.0) -> np.ndarray:
    h = np.zeros((2 * width, 2 * width), dtype=complex)
    for n in range(width):
        sl = slice(2 * n, 2 * n + 2)
        h[sl, sl] = potential * S0 - J * SZ
        if n:
            prev = slice(2 * (n - 1), 2 * n)
            h[sl, prev] += -t * S0
            h[prev, sl] += -t * S0
    return h


def surface_gf_sancho_rubio(
    energy: float,
    h_slice: np.ndarray,
    coupling: np.ndarray,
    *,
    eta: float = 1e-7,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """Surface Green function by the Lopez-Sancho decimation algorithm."""
    zI = (energy + 1j * eta) * np.eye(h_slice.shape[0], dtype=complex)
    e_bulk = h_slice.copy()
    e_surface = h_slice.copy()
    alpha = coupling.copy()
    beta = coupling.conj().T.copy()
    for _ in range(max_iter):
        g = np.linalg.inv(zI - e_bulk)
        ag = alpha @ g
        bg = beta @ g
        e_surface += ag @ beta
        e_bulk += ag @ beta + bg @ alpha
        alpha = ag @ alpha
        beta = bg @ beta
        if max(np.max(np.abs(alpha)), np.max(np.abs(beta))) < tol:
            break
    else:
        raise RuntimeError("Sancho-Rubio surface Green function did not converge")
    return np.linalg.inv(zI - e_surface)


@lru_cache(maxsize=4096)
def uniform_lead_self_energy(
    energy: float,
    width: int,
    J: float,
    t: float,
    *,
    eta: float = 1e-7,
    potential: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    h = uniform_lead_slice(width, J, t, potential)
    v = -t * np.eye(2 * width, dtype=complex)
    g = surface_gf_sancho_rubio(energy, h, v, eta=eta)
    sigma = v.conj().T @ g @ v
    gamma = 1j * (sigma - sigma.conj().T)
    gamma = 0.5 * (gamma + gamma.conj().T)
    # Cached lead matrices are shared read-only within a worker process.
    sigma.setflags(write=False)
    gamma.setflags(write=False)
    return sigma, gamma
