"""Square-lattice spinful tight-binding Hamiltonians."""

from __future__ import annotations

import numpy as np

S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def exchange_onsite(m: np.ndarray, J: float, potential: float = 0.0) -> np.ndarray:
    return potential * S0 - J * (m[0] * SX + m[1] * SY + m[2] * SZ)


def slice_hamiltonians(
    texture: np.ndarray,
    J: float,
    t: float,
    onsite_disorder: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build x-directed slices, each containing all y and spin orbitals."""
    L, W, _ = texture.shape
    if onsite_disorder is None:
        onsite_disorder = np.zeros((L, W), dtype=float)
    if onsite_disorder.shape != (L, W):
        raise ValueError("onsite_disorder must have shape (L, W)")
    slices: list[np.ndarray] = []
    for x in range(L):
        h = np.zeros((2 * W, 2 * W), dtype=complex)
        for y in range(W):
            sl = slice(2 * y, 2 * y + 2)
            h[sl, sl] = exchange_onsite(texture[x, y], J, onsite_disorder[x, y])
            if y:
                prev = slice(2 * (y - 1), 2 * y)
                h[sl, prev] += -t * S0
                h[prev, sl] += -t * S0
        slices.append(h)
    return slices


def dense_device_hamiltonian(
    texture: np.ndarray,
    J: float,
    t: float,
    onsite_disorder: np.ndarray | None = None,
) -> np.ndarray:
    """Build the full open-boundary device Hamiltonian (testing/small devices)."""
    L, W, _ = texture.shape
    hs = slice_hamiltonians(texture, J, t, onsite_disorder)
    n = 2 * L * W
    H = np.zeros((n, n), dtype=complex)
    V = -t * np.eye(2 * W, dtype=complex)
    for x, h in enumerate(hs):
        a = slice(2 * W * x, 2 * W * (x + 1))
        H[a, a] = h
        if x:
            b = slice(2 * W * (x - 1), 2 * W * x)
            H[a, b] = V
            H[b, a] = V.conj().T
    return H


def sparse_device_hamiltonian(
    texture: np.ndarray,
    J: float,
    t: float,
    onsite_disorder: np.ndarray | None = None,
):
    """CSR device Hamiltonian for large multi-terminal selected solves."""
    try:
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError("SciPy is required for sparse multi-terminal calculations") from exc
    L, W, _ = texture.shape
    if onsite_disorder is None:
        onsite_disorder = np.zeros((L, W), dtype=float)
    rows, cols, data = [], [], []

    def add_block(i0: int, j0: int, block: np.ndarray):
        for a in range(2):
            for b in range(2):
                if block[a, b] != 0:
                    rows.append(i0 + a)
                    cols.append(j0 + b)
                    data.append(block[a, b])

    for x in range(L):
        for y in range(W):
            i = 2 * (x * W + y)
            add_block(i, i, exchange_onsite(texture[x, y], J, onsite_disorder[x, y]))
            if x + 1 < L:
                j = 2 * ((x + 1) * W + y)
                add_block(i, j, -t * S0)
                add_block(j, i, -t * S0)
            if y + 1 < W:
                j = 2 * (x * W + y + 1)
                add_block(i, j, -t * S0)
                add_block(j, i, -t * S0)
    n = 2 * L * W
    return coo_matrix((data, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
