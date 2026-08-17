"""Bloch supercells, band gaps, Berry flux and FHS Chern numbers."""

from __future__ import annotations

import numpy as np

from .model import S0, exchange_onsite


def bloch_hamiltonian(texture_cell: np.ndarray, kx: float, ky: float, J: float, t: float) -> np.ndarray:
    """Hamiltonian for a periodic square supercell; k is in inverse lattice units."""
    Ax, Ay, _ = texture_cell.shape
    n = 2 * Ax * Ay
    H = np.zeros((n, n), dtype=complex)

    def site(x: int, y: int) -> slice:
        i = x * Ay + y
        return slice(2 * i, 2 * i + 2)

    for x in range(Ax):
        for y in range(Ay):
            a = site(x, y)
            H[a, a] += exchange_onsite(texture_cell[x, y], J)
            for dx, dy, phase in (
                (1, 0, np.exp(1j * kx * Ax) if x == Ax - 1 else 1.0),
                (0, 1, np.exp(1j * ky * Ay) if y == Ay - 1 else 1.0),
            ):
                xn, yn = (x + dx) % Ax, (y + dy) % Ay
                b = site(xn, yn)
                hop = -t * phase * S0
                H[a, b] += hop
                H[b, a] += hop.conj().T
    return H


def hermiticity_error(H: np.ndarray) -> float:
    return float(np.max(np.abs(H - H.conj().T)))


def high_symmetry_path(A: int, points_per_segment: int = 20):
    """Γ-X-M-Γ path in the magnetic Brillouin zone."""
    g = np.array([0.0, 0.0])
    x = np.array([np.pi / A, 0.0])
    m = np.array([np.pi / A, np.pi / A])
    vertices = [g, x, m, g]
    kpoints = []
    distances = []
    ticks = [0.0]
    distance = 0.0
    for start, stop in zip(vertices[:-1], vertices[1:]):
        for j in range(points_per_segment):
            frac = j / points_per_segment
            k = start + frac * (stop - start)
            if kpoints:
                distance += float(np.linalg.norm(k - kpoints[-1]))
            kpoints.append(k)
            distances.append(distance)
        distance += float(np.linalg.norm(stop - kpoints[-1]))
        ticks.append(distance)
    kpoints.append(g)
    distances.append(distance)
    return np.asarray(kpoints), np.asarray(distances), np.asarray(ticks), ["Γ", "X", "M", "Γ"]


def band_structure(texture_cell: np.ndarray, kpoints: np.ndarray, J: float, t: float) -> np.ndarray:
    return np.asarray([
        np.linalg.eigvalsh(bloch_hamiltonian(texture_cell, k[0], k[1], J, t))
        for k in kpoints
    ])


def uniform_folded_energies(A: int, kx: float, ky: float, J: float, t: float) -> np.ndarray:
    vals = []
    for nx in range(A):
        for ny in range(A):
            qx = kx + 2.0 * np.pi * nx / A
            qy = ky + 2.0 * np.pi * ny / A
            base = -2.0 * t * (np.cos(qx) + np.cos(qy))
            vals.extend((base - J, base + J))
    return np.sort(np.asarray(vals))


def direct_indirect_gap(eigenvalues_grid: np.ndarray, n_occ: int) -> dict[str, float]:
    """Gaps between bands n_occ-1 and n_occ over a 2D k grid."""
    valence = eigenvalues_grid[..., n_occ - 1]
    conduction = eigenvalues_grid[..., n_occ]
    return {
        "direct_gap": float(np.min(conduction - valence)),
        "indirect_gap": float(np.min(conduction) - np.max(valence)),
        "valence_max": float(np.max(valence)),
        "conduction_min": float(np.min(conduction)),
    }


def find_global_gaps(eigenvalues_grid: np.ndarray, minimum: float = 1e-4) -> list[dict[str, float | int]]:
    """Find every positive indirect gap on a sampled full Brillouin zone."""
    nbands = eigenvalues_grid.shape[-1]
    gaps = []
    for n_occ in range(1, nbands):
        info = direct_indirect_gap(eigenvalues_grid, n_occ)
        if info["indirect_gap"] > minimum:
            gaps.append({"n_occ": n_occ, **info})
    return gaps


def k_grid_eigenvalues(texture_cell: np.ndarray, nk: int, J: float, t: float) -> np.ndarray:
    A = texture_cell.shape[0]
    ks = np.linspace(-np.pi / A, np.pi / A, nk, endpoint=False)
    return np.asarray([[np.linalg.eigvalsh(bloch_hamiltonian(texture_cell, kx, ky, J, t))
                        for ky in ks] for kx in ks])


def gaussian_dos(eigenvalues: np.ndarray, energies: np.ndarray, broadening: float) -> np.ndarray:
    if broadening <= 0:
        raise ValueError("broadening must be positive")
    flat = np.ravel(eigenvalues)
    delta = np.asarray(energies)[:, None] - flat[None, :]
    dos = np.exp(-0.5 * (delta / broadening) ** 2).sum(axis=1)
    return dos / (np.sqrt(2.0 * np.pi) * broadening * flat.size)


def fhs_chern_subspace(
    texture_cell: np.ndarray,
    n_occ: int,
    nk: int,
    J: float,
    t: float,
) -> tuple[float, np.ndarray]:
    """Gauge-invariant Fukui-Hatsugai-Suzuki Chern number.

    The determinant of the overlap matrix makes this valid for a possibly
    internally degenerate occupied subspace, provided it is separated from the
    unoccupied space by a direct gap everywhere.
    """
    A = texture_cell.shape[0]
    ks = np.linspace(-np.pi / A, np.pi / A, nk, endpoint=False)
    nband = 2 * texture_cell.shape[0] * texture_cell.shape[1]
    if not 0 < n_occ < nband:
        raise ValueError("n_occ must lie strictly between 0 and the number of bands")
    # The total finite-orbital Hilbert bundle is trivial.  For fillings above
    # half, evaluate the smaller unoccupied complement and reverse the sign.
    use_complement = n_occ > nband // 2
    selection = slice(n_occ, None) if use_complement else slice(None, n_occ)
    sign = -1.0 if use_complement else 1.0

    def eigen_row(kx: float) -> list[np.ndarray]:
        row = []
        for ky in ks:
            _, u = np.linalg.eigh(bloch_hamiltonian(texture_cell, kx, ky, J, t))
            row.append(u[:, selection])
        return row

    def link(a: np.ndarray, b: np.ndarray) -> complex:
        phase = np.linalg.slogdet(a.conj().T @ b)[0]
        if phase == 0:
            raise RuntimeError("Singular neighboring-subspace overlap; refine the k grid")
        return complex(phase)

    # Stream two kx rows instead of storing nk rows of large occupied vectors.
    # This changes the A=18,nk=71 half-filled memory scale from ~17 GB to <0.5 GB.
    flux = np.empty((nk, nk), dtype=float)
    current = eigen_row(ks[0])
    for ix in range(nk):
        following = eigen_row(ks[ix + 1]) if ix + 1 < nk else eigen_row(ks[0])
        uy_current = np.asarray([link(current[iy], current[(iy + 1) % nk]) for iy in range(nk)])
        uy_following = np.asarray([link(following[iy], following[(iy + 1) % nk]) for iy in range(nk)])
        ux = np.asarray([link(current[iy], following[iy]) for iy in range(nk)])
        for iy in range(nk):
            loop = ux[iy] * uy_following[iy] / (ux[(iy + 1) % nk] * uy_current[iy])
            flux[ix, iy] = sign * np.angle(loop)
        current = following
    return float(np.sum(flux) / (2.0 * np.pi)), flux


def fhs_chern_band(
    texture_cell: np.ndarray,
    band_index: int,
    nk: int,
    J: float,
    t: float,
) -> tuple[float, np.ndarray]:
    """FHS Chern number for one isolated, nondegenerate zero-based band.

    SciPy's subset eigensolver obtains only the requested eigenvector, making
    31/51/71 grids practical for large magnetic cells.
    """
    try:
        from scipy.linalg import eigh
    except ImportError as exc:
        raise RuntimeError("SciPy is required for selected-band Chern calculations") from exc
    nband = 2 * texture_cell.shape[0] * texture_cell.shape[1]
    if not 0 <= band_index < nband:
        raise ValueError("band_index outside the spectrum")
    A = texture_cell.shape[0]
    ks = np.linspace(-np.pi / A, np.pi / A, nk, endpoint=False)

    def eigen_row(kx: float) -> list[np.ndarray]:
        return [eigh(
            bloch_hamiltonian(texture_cell, kx, ky, J, t),
            subset_by_index=(band_index, band_index),
            driver="evr",
            check_finite=False,
        )[1][:, 0] for ky in ks]

    def link(a: np.ndarray, b: np.ndarray) -> complex:
        overlap = np.vdot(a, b)
        if abs(overlap) < 1e-14:
            raise RuntimeError("Neighboring band eigenvectors have zero overlap; band may not be isolated")
        return overlap / abs(overlap)

    flux = np.empty((nk, nk), dtype=float)
    current = eigen_row(ks[0])
    for ix in range(nk):
        following = eigen_row(ks[ix + 1]) if ix + 1 < nk else eigen_row(ks[0])
        uy_current = np.asarray([link(current[iy], current[(iy + 1) % nk]) for iy in range(nk)])
        uy_following = np.asarray([link(following[iy], following[(iy + 1) % nk]) for iy in range(nk)])
        ux = np.asarray([link(current[iy], following[iy]) for iy in range(nk)])
        for iy in range(nk):
            flux[ix, iy] = np.angle(
                ux[iy] * uy_following[iy] / (ux[(iy + 1) % nk] * uy_current[iy])
            )
        current = following
    return float(np.sum(flux) / (2.0 * np.pi)), flux
