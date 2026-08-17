"""Two-terminal recursive Green-function transport."""

from __future__ import annotations

import numpy as np

from .leads import uniform_lead_self_energy
from .model import dense_device_hamiltonian, slice_hamiltonians


def clean_lead_modes(energies: np.ndarray, W: int, J: float, t: float) -> np.ndarray:
    """Analytic propagating-channel count for an open-y square-lattice strip."""
    energies = np.atleast_1d(energies)
    n = np.arange(1, W + 1)
    transverse = -2.0 * t * np.cos(n * np.pi / (W + 1))
    centers = np.concatenate((transverse - J, transverse + J))
    return np.array([np.count_nonzero(np.abs(E - centers) < 2.0 * t) for E in energies])


def two_terminal_transmission(
    texture: np.ndarray,
    energies: np.ndarray,
    J: float,
    t: float,
    *,
    eta: float = 1e-7,
    onsite_disorder: np.ndarray | None = None,
) -> np.ndarray:
    """Compute T(E) with x-slice RGF and identical uniform-z leads."""
    L, W, _ = texture.shape
    if L < 2:
        raise ValueError("RGF needs at least two x slices")
    hs = slice_hamiltonians(texture, J, t, onsite_disorder)
    d = 2 * W
    eye = np.eye(d, dtype=complex)
    V = -t * eye
    result = []
    for energy in np.atleast_1d(energies):
        sigma_l, gamma_l = uniform_lead_self_energy(energy, W, J, t, eta=eta)
        sigma_r, gamma_r = uniform_lead_self_energy(energy, W, J, t, eta=eta)
        z = energy + 1j * eta
        left_connected: list[np.ndarray] = []
        g = np.linalg.inv(z * eye - hs[0] - sigma_l)
        left_connected.append(g)
        for x in range(1, L - 1):
            g = np.linalg.inv(z * eye - hs[x] - V.conj().T @ g @ V)
            left_connected.append(g)
        g_last = np.linalg.inv(
            z * eye - hs[-1] - sigma_r - V.conj().T @ left_connected[-1] @ V
        )
        g_0n = left_connected[0]
        for g_mid in left_connected[1:]:
            g_0n = g_0n @ V @ g_mid
        g_0n = g_0n @ V @ g_last
        value = np.trace(gamma_l @ g_0n @ gamma_r @ g_0n.conj().T).real
        result.append(max(0.0, float(value)))
    return np.asarray(result)


def paired_prefix_transmission(
    texture: np.ndarray,
    prefix_lengths: tuple[int, ...],
    energies: np.ndarray,
    J: float,
    t: float,
    *,
    eta: float = 1e-7,
    onsite_disorder: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Compute several nested device lengths in one RGF traversal.

    Each requested device must be an x-prefix of ``texture``.  The result is
    algebraically identical to separate :func:`two_terminal_transmission`
    calls, while the left-connected Green functions of shorter prefixes are
    reused for longer ones.
    """
    total_length, W, _ = texture.shape
    lengths = tuple(sorted(set(int(length) for length in prefix_lengths)))
    if not lengths or lengths[0] < 2 or lengths[-1] > total_length:
        raise ValueError("prefix lengths must lie between 2 and the texture length")
    hs = slice_hamiltonians(texture, J, t, onsite_disorder)
    d = 2 * W
    eye = np.eye(d, dtype=complex)
    V = -t * eye
    result = {length: [] for length in lengths}
    target_last_slices = {length - 1: length for length in lengths}

    for energy in np.atleast_1d(energies):
        sigma, gamma = uniform_lead_self_energy(energy, W, J, t, eta=eta)
        z = energy + 1j * eta
        g = np.linalg.inv(z * eye - hs[0] - sigma)
        g_0x = g
        for x in range(1, lengths[-1]):
            if x in target_last_slices:
                g_last = np.linalg.inv(z * eye - hs[x] - sigma - V.conj().T @ g @ V)
                g_0n = g_0x @ V @ g_last
                value = np.trace(gamma @ g_0n @ gamma @ g_0n.conj().T).real
                result[target_last_slices[x]].append(max(0.0, float(value)))
            if x < lengths[-1] - 1:
                g = np.linalg.inv(z * eye - hs[x] - V.conj().T @ g @ V)
                g_0x = g_0x @ V @ g
    return {length: np.asarray(values) for length, values in result.items()}


def direct_inverse_transmission(
    texture: np.ndarray,
    energy: float,
    J: float,
    t: float,
    *,
    eta: float = 1e-7,
    onsite_disorder: np.ndarray | None = None,
) -> float:
    """Reference full-matrix calculation for small-system validation."""
    L, W, _ = texture.shape
    H = dense_device_hamiltonian(texture, J, t, onsite_disorder)
    sigma_l, gamma_l = uniform_lead_self_energy(energy, W, J, t, eta=eta)
    sigma_r, gamma_r = uniform_lead_self_energy(energy, W, J, t, eta=eta)
    n = H.shape[0]
    embedded_l = np.zeros_like(H)
    embedded_r = np.zeros_like(H)
    d = 2 * W
    embedded_l[:d, :d] = sigma_l
    embedded_r[-d:, -d:] = sigma_r
    G = np.linalg.inv((energy + 1j * eta) * np.eye(n) - H - embedded_l - embedded_r)
    g_lr = G[:d, -d:]
    return float(np.trace(gamma_l @ g_lr @ gamma_r @ g_lr.conj().T).real)


def full_green_observables(
    texture: np.ndarray,
    energy: float,
    J: float,
    t: float,
    *,
    eta: float = 1e-7,
    injection: str = "left",
    onsite_disorder: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """LDOS, injectivity and bond currents for a small two-terminal device.

    Bond-current arrays use the positive x/y bond orientation.  Overall current
    units are omitted because plots normally normalize arrows by their maximum.
    """
    L, W, _ = texture.shape
    H = dense_device_hamiltonian(texture, J, t, onsite_disorder)
    sigma_l, gamma_l = uniform_lead_self_energy(energy, W, J, t, eta=eta)
    sigma_r, gamma_r = uniform_lead_self_energy(energy, W, J, t, eta=eta)
    n, d = H.shape[0], 2 * W
    sl = np.zeros_like(H)
    sr = np.zeros_like(H)
    gl = np.zeros_like(H)
    gr = np.zeros_like(H)
    sl[:d, :d], gl[:d, :d] = sigma_l, gamma_l
    sr[-d:, -d:], gr[-d:, -d:] = sigma_r, gamma_r
    G = np.linalg.inv((energy + 1j * eta) * np.eye(n) - H - sl - sr)
    spectral = 1j * (G - G.conj().T)
    source_gamma = gl if injection == "left" else gr
    lesser_spectral = G @ source_gamma @ G.conj().T
    ldos = np.empty((L, W))
    injectivity = np.empty((L, W))
    for x in range(L):
        for y in range(W):
            i = 2 * (x * W + y)
            block = slice(i, i + 2)
            ldos[x, y] = np.trace(spectral[block, block]).real / (2.0 * np.pi)
            injectivity[x, y] = np.trace(lesser_spectral[block, block]).real / (2.0 * np.pi)
    jx = np.zeros((L - 1, W))
    jy = np.zeros((L, W - 1))
    for x in range(L):
        for y in range(W):
            i = slice(2 * (x * W + y), 2 * (x * W + y) + 2)
            if x + 1 < L:
                j = slice(2 * ((x + 1) * W + y), 2 * ((x + 1) * W + y) + 2)
                jx[x, y] = 2.0 * np.imag(np.trace(H[i, j] @ lesser_spectral[j, i]))
            if y + 1 < W:
                j = slice(2 * (x * W + y + 1), 2 * (x * W + y + 1) + 2)
                jy[x, y] = 2.0 * np.imag(np.trace(H[i, j] @ lesser_spectral[j, i]))
    g_lr = G[:d, -d:]
    transmission = float(np.trace(gamma_l @ g_lr @ gamma_r @ g_lr.conj().T).real)
    return {"transmission": transmission, "ldos": ldos, "injectivity": injectivity,
            "bond_current_x": jx, "bond_current_y": jy}


def fit_exponential_length(
    lengths: np.ndarray,
    transmission: np.ndarray,
    floor: float = np.finfo(float).tiny,
) -> dict[str, float]:
    """Fit ln(T)=intercept+slope*L and report R² and decay length."""
    x = np.asarray(lengths, dtype=float)
    if floor <= 0:
        raise ValueError("floor must be positive")
    y = np.log(np.maximum(np.asarray(transmission, dtype=float), floor))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2,
            "decay_length": float(-1.0 / slope) if slope < 0 else np.inf}
