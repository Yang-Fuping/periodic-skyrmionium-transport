"""Multi-terminal NEGF and Landauer-Büttiker observables.

This transparent implementation uses a full dense device Green function and is
therefore intended for benchmark devices.  Large parameter scans should use the
Kwant backend or a sparse selected-inversion implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .leads import uniform_lead_self_energy
from .model import dense_device_hamiltonian, sparse_device_hamiltonian


@dataclass(frozen=True)
class Contact:
    name: str
    edge: str
    start: int
    stop: int
    lead_J: float | None = None
    coupling_scale: float = 1.0


def standard_four_contacts(
    L: int,
    W: int,
    probe_width: int | None = None,
    *,
    probe_J: float | None = None,
    probe_start: int | None = None,
    longitudinal_J: float | None = None,
    probe_coupling: float = 1.0,
    longitudinal_coupling: float = 1.0,
) -> list[Contact]:
    """L/R full-width leads and centered B/T voltage probes.

    ``longitudinal_J`` may be negative to match a ``-z`` magnetic background.
    ``None`` preserves the historical behavior of using the device exchange.
    """
    if probe_width is None:
        probe_width = L
    probe_width = min(probe_width, max(1, L - 2))
    start = (L - probe_width) // 2 if probe_start is None else probe_start
    stop = start + probe_width
    if start < 1 or stop > L - 1:
        raise ValueError("Side probes must avoid the left/right corner contact sites")
    if probe_coupling <= 0 or longitudinal_coupling <= 0:
        raise ValueError("Contact coupling scales must be positive")
    return [
        Contact("L", "left", 0, W, longitudinal_J, longitudinal_coupling),
        Contact("R", "right", 0, W, longitudinal_J, longitudinal_coupling),
        Contact("B", "bottom", start, stop, probe_J, probe_coupling),
        Contact("T", "top", start, stop, probe_J, probe_coupling),
    ]


def _contact_sites(contact: Contact, L: int, W: int) -> list[tuple[int, int]]:
    if contact.edge == "left":
        return [(0, y) for y in range(contact.start, contact.stop)]
    if contact.edge == "right":
        return [(L - 1, y) for y in range(contact.start, contact.stop)]
    if contact.edge == "bottom":
        return [(x, 0) for x in range(contact.start, contact.stop)]
    if contact.edge == "top":
        return [(x, W - 1) for x in range(contact.start, contact.stop)]
    raise ValueError(f"Unknown edge {contact.edge!r}")


def _orbital_indices(sites: list[tuple[int, int]], W: int) -> np.ndarray:
    return np.asarray([2 * (x * W + y) + spin for x, y in sites for spin in (0, 1)])


def transmission_matrix(
    texture: np.ndarray,
    energy: float,
    J: float,
    t: float,
    contacts: list[Contact],
    *,
    eta: float = 1e-7,
    onsite_disorder: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return T[p,q] (transmission q -> p) and diagnostic matrices."""
    L, W, _ = texture.shape
    H = dense_device_hamiltonian(texture, J, t, onsite_disorder)
    n = H.shape[0]
    sigma_total = np.zeros((n, n), dtype=complex)
    indices: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    lead_Js = []
    for c in contacts:
        sites = _contact_sites(c, L, W)
        idx = _orbital_indices(sites, W)
        contact_J = J if c.lead_J is None else c.lead_J
        sigma, gamma = uniform_lead_self_energy(energy, len(sites), contact_J, t, eta=eta)
        coupling_weight = float(c.coupling_scale) ** 2
        sigma = coupling_weight * sigma
        gamma = coupling_weight * gamma
        sigma_total[np.ix_(idx, idx)] += sigma
        indices.append(idx)
        gammas.append(gamma)
        sigmas.append(sigma)
        lead_Js.append(contact_J)
    G = np.linalg.inv((energy + 1j * eta) * np.eye(n) - H - sigma_total)
    nlead = len(contacts)
    T = np.zeros((nlead, nlead), dtype=float)
    for p in range(nlead):
        for q in range(nlead):
            if p == q:
                continue
            gpq = G[np.ix_(indices[p], indices[q])]
            T[p, q] = max(0.0, float(np.trace(
                gammas[p] @ gpq @ gammas[q] @ gpq.conj().T
            ).real))
    return T, {"G": G, "indices": indices, "gammas": gammas, "sigmas": sigmas,
               "energy": energy, "J": J, "lead_Js": np.asarray(lead_Js), "t": t,
               "coupling_scales": np.asarray([c.coupling_scale for c in contacts]),
               "widths": np.asarray([len(i) // 2 for i in indices], dtype=int)}


def transmission_matrix_sparse(
    texture: np.ndarray,
    energy: float,
    J: float,
    t: float,
    contacts: list[Contact],
    *,
    eta: float = 1e-7,
    onsite_disorder: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sparse-LU multi-terminal Caroli calculation using contact columns only."""
    try:
        from scipy.sparse import eye
        from scipy.sparse.linalg import splu
    except ImportError as exc:
        raise RuntimeError("SciPy is required for sparse multi-terminal calculations") from exc
    L, W, _ = texture.shape
    H = sparse_device_hamiltonian(texture, J, t, onsite_disorder)
    n = H.shape[0]
    global_indices: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    lead_Js = []
    for c in contacts:
        sites = _contact_sites(c, L, W)
        idx = _orbital_indices(sites, W)
        contact_J = J if c.lead_J is None else c.lead_J
        sigma, gamma = uniform_lead_self_energy(energy, len(sites), contact_J, t, eta=eta)
        coupling_weight = float(c.coupling_scale) ** 2
        sigma = coupling_weight * sigma
        gamma = coupling_weight * gamma
        global_indices.append(idx)
        sigmas.append(sigma)
        gammas.append(gamma)
        lead_Js.append(contact_J)
    A = ((energy + 1j * eta) * eye(n, dtype=complex, format="lil") - H).tolil()
    for idx, sigma in zip(global_indices, sigmas):
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                value = sigma[a, b]
                if value != 0:
                    A[ia, ib] -= value
    union = np.concatenate(global_indices)
    if len(np.unique(union)) != len(union):
        raise ValueError("Contacts must not share device orbitals")
    rhs = np.zeros((n, len(union)), dtype=complex)
    rhs[union, np.arange(len(union))] = 1.0
    selected = splu(A.tocsc()).solve(rhs)
    Gc = selected[union, :]
    offsets = np.cumsum([0] + [len(i) for i in global_indices])
    local_indices = [np.arange(offsets[p], offsets[p + 1]) for p in range(len(contacts))]
    T = np.zeros((len(contacts), len(contacts)), dtype=float)
    for p in range(len(contacts)):
        for q in range(len(contacts)):
            if p == q:
                continue
            gpq = Gc[np.ix_(local_indices[p], local_indices[q])]
            T[p, q] = max(0.0, float(np.trace(
                gammas[p] @ gpq @ gammas[q] @ gpq.conj().T
            ).real))
    diag = {"G_contact": Gc, "G_selected": selected,
            "selected_global_columns": union,
            "indices": local_indices, "global_indices": global_indices,
            "gammas": gammas, "sigmas": sigmas, "energy": energy, "J": J,
            "lead_Js": np.asarray(lead_Js), "t": t,
            "coupling_scales": np.asarray([c.coupling_scale for c in contacts]),
            "widths": np.asarray([len(i) // 2 for i in global_indices], dtype=int)}
    return T, diag


def spin_resolved_transmission(diagnostics: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``Ts[p,q,s_out,s_in]`` in the z-spin basis of uniform leads.

    The charge sum over both spin indices reproduces the off-diagonal Caroli
    transmission.  This quantity is directly comparable to Kwant lead-block
    transmissions when the lead conservation law is sigma_z.
    """
    G = diagnostics.get("G_contact", diagnostics.get("G"))
    indices = diagnostics["indices"]
    gammas = diagnostics["gammas"]
    nlead = len(indices)
    Ts = np.zeros((nlead, nlead, 2, 2), dtype=float)
    projected = []
    for gamma in gammas:
        width = gamma.shape[0] // 2
        by_spin = []
        for spin in (0, 1):
            mask = np.zeros(2 * width)
            mask[spin::2] = 1.0
            P = np.diag(mask)
            by_spin.append(P @ gamma @ P)
        projected.append(by_spin)
    for p in range(nlead):
        for q in range(nlead):
            if p == q:
                continue
            gpq = G[np.ix_(indices[p], indices[q])]
            for so in (0, 1):
                for si in (0, 1):
                    Ts[p, q, so, si] = max(0.0, float(np.trace(
                        projected[p][so] @ gpq @ projected[q][si] @ gpq.conj().T
                    ).real))
    return Ts


def spin_current_proxy(Ts: np.ndarray, voltages: np.ndarray) -> np.ndarray:
    """Inter-lead z-spin current in units of ``e/4π`` up to a factor convention.

    Positive means net +z angular momentum flowing *out of* the reservoir.  It
    includes all inter-lead spin-resolved processes but not spin-flip reflection;
    the latter requires a true scattering-matrix implementation before this can
    support a final publication spin-current claim.
    """
    signs = np.array([1.0, -1.0])
    nlead = Ts.shape[0]
    out = np.zeros(nlead)
    for p in range(nlead):
        for q in range(nlead):
            if p == q:
                continue
            # Emission p -> q, weighted by spin in the emitting p channel.
            emitted = np.sum(signs[:, None] * Ts[q, p].T)
            # Arrival q -> p, weighted by outgoing spin in p.
            arrived = np.sum(signs[:, None] * Ts[p, q])
            out[p] += emitted * voltages[p] - arrived * voltages[q]
    return out


def scattering_matrix_from_green(
    diagnostics: dict[str, np.ndarray],
) -> tuple[list[list[np.ndarray]], list[np.ndarray], float]:
    """Construct flux-normalized lead-channel scattering blocks.

    ``S[p][q]`` maps incoming channels in q to outgoing channels in p.  The
    factorization Gamma=W W† and Fisher-Lee relation ``S=1-i W†GW`` include
    reflection, which is essential for a conserved terminal spin current.
    Propagating-channel counts are fixed by the analytic uniform-lead bands,
    avoiding spurious O(eta) evanescent eigenvalues of Gamma.
    """
    G = diagnostics.get("G_contact", diagnostics.get("G"))
    indices = diagnostics["indices"]
    gammas = diagnostics["gammas"]
    energy = float(diagnostics["energy"])
    lead_Js = np.asarray(diagnostics.get("lead_Js", np.full(len(indices), diagnostics["J"])))
    t = float(diagnostics["t"])
    widths = diagnostics["widths"]
    factors: list[np.ndarray] = []
    spin_labels: list[np.ndarray] = []
    for gamma, width, lead_J in zip(gammas, widths, lead_Js):
        columns = []
        labels = []
        transverse = -2.0 * t * np.cos(np.arange(1, width + 1) * np.pi / (width + 1))
        for spin, onsite_shift in ((0, -lead_J), (1, lead_J)):
            n_open = int(np.count_nonzero(np.abs(energy - (transverse + onsite_shift)) < 2.0 * t))
            block = gamma[spin::2, spin::2]
            values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
            if n_open:
                chosen = np.argsort(values)[-n_open:]
                for j in chosen:
                    if values[j] <= 0:
                        raise RuntimeError("Expected a positive broadening eigenvalue for an open channel")
                    col = np.zeros(2 * width, dtype=complex)
                    col[spin::2] = vectors[:, j] * np.sqrt(values[j])
                    columns.append(col)
                    labels.append(1.0 if spin == 0 else -1.0)
        factors.append(np.column_stack(columns) if columns else np.zeros((2 * width, 0), complex))
        spin_labels.append(np.asarray(labels))

    nlead = len(indices)
    blocks: list[list[np.ndarray]] = [[None for _ in range(nlead)] for _ in range(nlead)]
    for p in range(nlead):
        for q in range(nlead):
            gpq = G[np.ix_(indices[p], indices[q])]
            block = -1j * factors[p].conj().T @ gpq @ factors[q]
            if p == q:
                block += np.eye(block.shape[0], dtype=complex)
            blocks[p][q] = block
    total_channels = sum(len(s) for s in spin_labels)
    S = np.zeros((total_channels, total_channels), dtype=complex)
    offsets = np.cumsum([0] + [len(s) for s in spin_labels])
    for p in range(nlead):
        for q in range(nlead):
            S[offsets[p]:offsets[p + 1], offsets[q]:offsets[q + 1]] = blocks[p][q]
    unitary_error = float(np.max(np.abs(S.conj().T @ S - np.eye(total_channels))))
    return blocks, spin_labels, unitary_error


def terminal_currents_from_scattering(
    blocks: list[list[np.ndarray]],
    spin_labels: list[np.ndarray],
    voltages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact charge and z-spin terminal currents from the full S matrix.

    Charge is in ``e²/h`` times voltage.  Spin is the dimensionless coefficient
    multiplying ``e/4π`` times voltage.  Both include reflection and vanish at
    equilibrium by scattering-matrix unitarity.
    """
    nlead = len(blocks)
    charge = np.zeros(nlead)
    spin = np.zeros(nlead)
    for p in range(nlead):
        charge[p] += len(spin_labels[p]) * voltages[p]
        spin[p] += np.sum(spin_labels[p]) * voltages[p]
        for q in range(nlead):
            probabilities = np.abs(blocks[p][q]) ** 2
            charge[p] -= np.sum(probabilities) * voltages[q]
            spin[p] -= np.sum(spin_labels[p][:, None] * probabilities) * voltages[q]
    return charge, spin


def terminal_currents_channel_reference(
    blocks: list[list[np.ndarray]],
    spin_labels: list[np.ndarray],
    voltages: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal channel-by-channel reference for the terminal-current formula.

    This deliberately favors transparency over speed.  It is used in tests to
    verify the vectorized full-scattering-matrix implementation, including
    spin-flip reflection, against the Yin/Ndiaye lead-current definition.
    """
    nlead = len(blocks)
    charge = np.zeros(nlead)
    spin = np.zeros(nlead)
    for p in range(nlead):
        for a, spin_out in enumerate(spin_labels[p]):
            charge[p] += voltages[p]
            spin[p] += spin_out * voltages[p]
            for q in range(nlead):
                for b in range(len(spin_labels[q])):
                    probability = float(abs(blocks[p][q][a, b]) ** 2)
                    charge[p] -= probability * voltages[q]
                    spin[p] -= spin_out * probability * voltages[q]
    return charge, spin


def four_terminal_hall_observables(
    voltages: np.ndarray,
    charge_currents: np.ndarray,
    spin_currents: np.ndarray,
    *,
    source_channels: int | None = None,
    min_current: float = 1e-8,
    min_current_fraction: float = 1e-6,
    names=("L", "R", "B", "T"),
) -> dict[str, float | bool]:
    """Return publication charge and spin Hall angles with a validity flag.

    Spin currents are the dimensionless coefficients multiplying ``e/4pi``;
    charge currents multiply ``e^2/h``.  Consequently the factor ``2e/hbar``
    in the Yin/Ndiaye spin-Hall definition cancels in their ratio here.
    """
    index = {name: i for i, name in enumerate(names)}
    vlr = float(voltages[index["L"]] - voltages[index["R"]])
    charge_denominator = float(
        charge_currents[index["L"]] - charge_currents[index["R"]]
    )
    spin_numerator = float(
        spin_currents[index["T"]] - spin_currents[index["B"]]
    )
    charge_angle = float(
        (voltages[index["T"]] - voltages[index["B"]]) / vlr
    ) if vlr != 0 else np.nan
    spin_angle = float(spin_numerator / charge_denominator) \
        if abs(charge_denominator) >= min_current else np.nan
    source_current = float(abs(charge_currents[index["L"]]))
    channel_scale = float(max(source_channels or 0, 1))
    valid = (
        np.isfinite(charge_angle)
        and np.isfinite(spin_angle)
        and source_current >= min_current
        and source_current / channel_scale >= min_current_fraction
    )
    return {
        "charge_hall_angle": charge_angle,
        "spin_hall_angle": spin_angle,
        "spin_hall_numerator": spin_numerator,
        "longitudinal_charge_denominator": charge_denominator,
        "source_current": source_current,
        "source_current_fraction": source_current / channel_scale,
        "valid_hall_point": bool(valid),
    }


def landauer_buttiker_matrix(T: np.ndarray) -> np.ndarray:
    """Matrix I=LV for T[p,q] defined as transmission q -> p."""
    T = np.asarray(T, dtype=float)
    L = -T.copy()
    np.fill_diagonal(L, np.sum(T, axis=0))
    return L


def solve_voltage_probes(
    T: np.ndarray,
    fixed_voltages: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve all unspecified leads as zero-current voltage probes."""
    n = T.shape[0]
    fixed = np.array(sorted(fixed_voltages), dtype=int)
    floating = np.array([i for i in range(n) if i not in fixed_voltages], dtype=int)
    V = np.zeros(n, dtype=float)
    V[fixed] = [fixed_voltages[i] for i in fixed]
    Lmat = landauer_buttiker_matrix(T)
    if floating.size:
        V[floating] = np.linalg.solve(
            Lmat[np.ix_(floating, floating)],
            -Lmat[np.ix_(floating, fixed)] @ V[fixed],
        )
    I = Lmat @ V
    return V, I


def four_terminal_observables(T: np.ndarray, names=("L", "R", "B", "T")) -> dict[str, float | np.ndarray]:
    index = {name: i for i, name in enumerate(names)}
    Lmat = landauer_buttiker_matrix(T)
    V, I = solve_voltage_probes(T, {index["L"]: 0.5, index["R"]: -0.5})
    source_current = I[index["L"]]
    if abs(source_current) < 1e-14:
        rxx = rxy = np.nan
    else:
        rxx = (V[index["L"]] - V[index["R"]]) / source_current
        rxy = (V[index["T"]] - V[index["B"]]) / source_current
    return {
        "voltages": V,
        "currents": I,
        "Rxx_h_over_e2": float(rxx),
        "Rxy_h_over_e2": float(rxy),
        "hall_angle": float(rxy / rxx) if np.isfinite(rxx) and rxx != 0 else np.nan,
        "current_conservation_error": float(abs(np.sum(I))),
        "probe_current_error": float(max(abs(I[index["T"]]), abs(I[index["B"]]))),
        "gauge_invariance_error": float(np.max(np.abs(Lmat @ np.ones(len(names))))),
    }
