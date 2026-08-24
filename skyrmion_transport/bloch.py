"""Bloch supercells, band gaps, Berry flux and FHS Chern numbers."""

from __future__ import annotations

import numpy as np

from .model import (
    S0,
    exchange_onsite,
    slice_hamiltonians,
    sparse_device_hamiltonian,
)


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


def strip_bloch_multipliers(
    texture_period: np.ndarray,
    energy: float,
    J: float,
    t: float,
) -> np.ndarray:
    """Return x-directed Bloch multipliers for an open-y periodic strip.

    ``texture_period`` contains one magnetic period along x and the complete
    open transverse strip along y.  Nearest-neighbour x hopping is invertible,
    so the atomic-slice Schrödinger equation can be written as a first-order
    transfer problem.  The eigenvalues of the product over one magnetic period
    are ``lambda = exp(i k_x A)``.  This formulation matches the central region
    used by the finite-array RGF calculation, including its open-y boundary.
    """
    texture_period = np.asarray(texture_period, dtype=float)
    if texture_period.ndim != 3 or texture_period.shape[-1] != 3:
        raise ValueError("texture_period must have shape (A, W, 3)")
    if texture_period.shape[0] < 1 or texture_period.shape[1] < 1:
        raise ValueError("texture_period must contain at least one site")
    if t == 0:
        raise ValueError("t must be nonzero")

    slices = slice_hamiltonians(texture_period, J, t)
    d = slices[0].shape[0]
    eye = np.eye(d, dtype=complex)
    zero = np.zeros((d, d), dtype=complex)
    period_transfer = np.eye(2 * d, dtype=complex)
    for h_slice in slices:
        # With V=-t I, the recurrence is
        # psi_(x+1) = (H_x-E I)/t psi_x - psi_(x-1).
        atomic_transfer = np.block([
            [(h_slice - energy * eye) / t, -eye],
            [eye, zero],
        ])
        period_transfer = atomic_transfer @ period_transfer
    return np.linalg.eigvals(period_transfer)


def slowest_strip_evanescent_mode(
    texture_period: np.ndarray,
    energy: float,
    J: float,
    t: float,
    *,
    unit_circle_tolerance: float = 1e-7,
) -> dict[str, float | complex | int | bool]:
    """Characterize the slowest right-decaying mode of a periodic strip.

    For an insulating strip the transmission envelope obeys
    ``T ~ exp(-2 kappa L)``.  Consequently the length convention used by
    :func:`skyrmion_transport.transport.fit_exponential_length` predicts
    ``xi = 1/(2 kappa)``.  A multiplier numerically on the unit circle denotes
    a propagating mode and returns an infinite decay length.
    """
    if unit_circle_tolerance <= 0:
        raise ValueError("unit_circle_tolerance must be positive")
    multipliers = strip_bloch_multipliers(texture_period, energy, J, t)
    period = texture_period.shape[0]
    moduli = np.abs(multipliers)
    finite = np.isfinite(moduli) & (moduli > 0)
    near_unit = finite & (np.abs(np.log(moduli)) <= unit_circle_tolerance)
    reciprocal_logs = np.sort(np.log(moduli[finite]))
    reciprocal_pair_error = (
        float(np.max(np.abs(reciprocal_logs + reciprocal_logs[::-1])))
        if reciprocal_logs.size else np.inf
    )
    if np.any(near_unit):
        candidate = multipliers[np.flatnonzero(near_unit)[0]]
        return {
            "multiplier": complex(candidate),
            "modulus": float(abs(candidate)),
            "kappa_per_a": 0.0,
            "xi_transmission_a": np.inf,
            "propagating_mode_count": int(np.count_nonzero(near_unit)),
            "has_propagating_mode": True,
            "reciprocal_log_pair_error": reciprocal_pair_error,
        }

    decaying = finite & (moduli < 1.0)
    if not np.any(decaying):
        raise RuntimeError("No finite right-decaying Bloch multiplier was found")
    indices = np.flatnonzero(decaying)
    index = indices[np.argmax(moduli[indices])]
    multiplier = multipliers[index]
    kappa = -float(np.log(abs(multiplier))) / period
    return {
        "multiplier": complex(multiplier),
        "modulus": float(abs(multiplier)),
        "kappa_per_a": kappa,
        "xi_transmission_a": 1.0 / (2.0 * kappa),
        "propagating_mode_count": 0,
        "has_propagating_mode": False,
        "reciprocal_log_pair_error": reciprocal_pair_error,
    }


def strip_bloch_multipliers_boundary_green(
    texture_period: np.ndarray,
    energy: float,
    J: float,
    t: float,
    *,
    residual_tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve strip complex bands through a stable boundary Green function.

    Directly multiplying atomic transfer matrices over a long magnetic period
    destroys reciprocal Bloch pairs in double precision.  Here the isolated
    cell Green function is reduced to its left/right boundary blocks.  The
    Bloch condition then becomes a quadratic eigenvalue problem of dimension
    only twice the boundary size.  Dense QZ handles the singular leading and
    trailing coefficients without forming exponentially large products.

    The returned residuals refer to the reduced quadratic equation.  Values
    above ``residual_tolerance`` and zero/infinite roots introduced by the
    singular linearization are removed.
    """
    try:
        from scipy.linalg import eig
        from scipy.sparse import csc_matrix, eye
        from scipy.sparse.linalg import splu
    except ImportError as exc:
        raise RuntimeError("SciPy is required for stable complex-band calculations") from exc

    texture_period = np.asarray(texture_period, dtype=float)
    if texture_period.ndim != 3 or texture_period.shape[-1] != 3:
        raise ValueError("texture_period must have shape (A, W, 3)")
    period, width = texture_period.shape[:2]
    if period < 2:
        raise ValueError("The boundary-Green solver requires at least two x slices")
    if t == 0:
        raise ValueError("t must be nonzero")
    if residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be positive")

    h0 = sparse_device_hamiltonian(texture_period, J, t).astype(complex)
    n = h0.shape[0]
    boundary = 2 * width
    system = energy * eye(n, dtype=complex, format="csc") - csc_matrix(h0)
    selectors = np.concatenate((np.arange(boundary), np.arange(n - boundary, n)))
    rhs = np.zeros((n, 2 * boundary), dtype=complex)
    rhs[selectors, np.arange(2 * boundary)] = 1.0
    boundary_green = splu(system).solve(rhs)[selectors]
    g_ll = boundary_green[:boundary, :boundary]
    g_lr = boundary_green[:boundary, boundary:]
    g_rl = boundary_green[boundary:, :boundary]
    g_rr = boundary_green[boundary:, boundary:]

    identity = np.eye(boundary, dtype=complex)
    zero = np.zeros_like(identity)
    q2 = np.block([[t * g_lr, zero], [t * g_rr, zero]])
    q1 = np.eye(2 * boundary, dtype=complex)
    q0 = np.block([[zero, t * g_ll], [zero, t * g_rl]])
    companion_a = np.block([
        [-q1, -q0],
        [np.eye(2 * boundary, dtype=complex), np.zeros((2 * boundary, 2 * boundary), dtype=complex)],
    ])
    companion_b = np.block([
        [q2, np.zeros((2 * boundary, 2 * boundary), dtype=complex)],
        [np.zeros((2 * boundary, 2 * boundary), dtype=complex), np.eye(2 * boundary, dtype=complex)],
    ])
    values, vectors = eig(
        companion_a,
        companion_b,
        right=True,
        check_finite=False,
    )
    residuals = np.full(values.shape, np.inf, dtype=float)
    finite = np.isfinite(values) & (np.abs(values) > np.finfo(float).eps)
    for index in np.flatnonzero(finite):
        value = values[index]
        boundary_state = vectors[2 * boundary:, index]
        term2 = q2 @ boundary_state * value**2
        term1 = q1 @ boundary_state * value
        term0 = q0 @ boundary_state
        denominator = (
            np.linalg.norm(term2) + np.linalg.norm(term1) + np.linalg.norm(term0)
        )
        residuals[index] = np.linalg.norm(term2 + term1 + term0) / max(
            denominator, np.finfo(float).tiny
        )
    keep = finite & (residuals <= residual_tolerance)
    return values[keep], residuals[keep]


def slowest_strip_mode_boundary_green(
    texture_period: np.ndarray,
    energy: float,
    J: float,
    t: float,
    *,
    residual_tolerance: float = 1e-8,
    unit_circle_tolerance: float = 1e-7,
) -> dict[str, float | complex | int | bool]:
    """Return the slowest physical strip mode from the stable QZ solver."""
    values, residuals = strip_bloch_multipliers_boundary_green(
        texture_period,
        energy,
        J,
        t,
        residual_tolerance=residual_tolerance,
    )
    if values.size == 0:
        raise RuntimeError("The boundary-Green complex-band solver found no valid roots")
    moduli = np.abs(values)
    logarithms = np.log(moduli)
    near_unit = np.abs(logarithms) <= unit_circle_tolerance
    common = {
        "valid_root_count": int(values.size),
        "maximum_qep_residual": float(np.max(residuals)),
        "propagating_mode_count": int(np.count_nonzero(near_unit)),
    }
    if np.any(near_unit):
        candidates = np.flatnonzero(near_unit)
        index = candidates[np.argmin(np.abs(logarithms[candidates]))]
        return {
            **common,
            "multiplier": complex(values[index]),
            "modulus": float(moduli[index]),
            "kappa_per_a": 0.0,
            "xi_transmission_a": np.inf,
            "has_propagating_mode": True,
            "selected_qep_residual": float(residuals[index]),
            "reciprocal_partner": complex(values[index]),
            "reciprocal_pair_error": 0.0,
        }
    decaying = np.flatnonzero(moduli < 1.0)
    if decaying.size == 0:
        raise RuntimeError("No right-decaying valid root was found")
    index = decaying[np.argmax(moduli[decaying])]
    value = values[index]
    target = 1.0 / np.conj(value)
    partner_index = int(np.argmin(np.abs(values - target)))
    partner = values[partner_index]
    pair_error = abs(value * np.conj(partner) - 1.0)
    kappa = -float(np.log(abs(value))) / texture_period.shape[0]
    return {
        **common,
        "multiplier": complex(value),
        "modulus": float(abs(value)),
        "kappa_per_a": kappa,
        "xi_transmission_a": 1.0 / (2.0 * kappa),
        "has_propagating_mode": False,
        "selected_qep_residual": float(residuals[index]),
        "reciprocal_partner": complex(partner),
        "reciprocal_pair_error": float(pair_error),
    }


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
