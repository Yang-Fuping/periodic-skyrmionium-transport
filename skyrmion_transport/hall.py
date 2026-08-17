"""Publication observables for four-terminal charge and spin Hall transport."""

from __future__ import annotations

import numpy as np

from .leads import uniform_lead_channel_summary
from .multiterminal import (
    Contact,
    four_terminal_hall_observables,
    four_terminal_observables,
    scattering_matrix_from_green,
    terminal_currents_from_scattering,
    transmission_matrix_sparse,
)
from .textures import lattice_topological_charge, windowed_topological_charge


def contact_spectral_maps(
    diagnostics: dict[str, np.ndarray],
    shape: tuple[int, int],
    *,
    source_lead: int = 0,
) -> dict[str, np.ndarray]:
    """Return LDOS, source injectivity and source-driven bond currents.

    Sparse multi-terminal Green calculations already solve every device row for
    all contact columns.  This reconstructs local spectral quantities from
    those selected columns without a full dense inverse.  Currents use the
    positive x/y bond convention and omit the common physical prefactor.
    """
    if "G_selected" not in diagnostics:
        raise ValueError("diagnostics must include sparse selected Green columns")
    L, W = shape
    n = 2 * L * W
    selected = diagnostics["G_selected"]
    if selected.shape[0] != n:
        raise ValueError("shape is inconsistent with the selected Green matrix")
    gammas = diagnostics["gammas"]
    widths = [gamma.shape[0] for gamma in gammas]
    offsets = np.cumsum([0] + widths)

    lead_columns = [selected[:, offsets[p]:offsets[p + 1]]
                    for p in range(len(gammas))]
    ldos = np.zeros((L, W), dtype=float)
    injectivity = np.zeros((L, W), dtype=float)
    source_x = lead_columns[source_lead]
    source_gamma = gammas[source_lead]
    for x in range(L):
        for y in range(W):
            site = x * W + y
            block = slice(2 * site, 2 * site + 2)
            total_trace = 0.0
            for columns, gamma in zip(lead_columns, gammas):
                local = columns[block]
                total_trace += np.trace(local @ gamma @ local.conj().T).real
            source_local = source_x[block]
            ldos[x, y] = total_trace / (2.0 * np.pi)
            injectivity[x, y] = (
                np.trace(source_local @ source_gamma @ source_local.conj().T).real
                / (2.0 * np.pi)
            )

    t = float(diagnostics["t"])
    hopping = -t * np.eye(2, dtype=complex)
    jx = np.zeros((L - 1, W), dtype=float)
    jy = np.zeros((L, W - 1), dtype=float)
    for x in range(L):
        for y in range(W):
            i = slice(2 * (x * W + y), 2 * (x * W + y) + 2)
            xi = source_x[i]
            if x + 1 < L:
                j = slice(2 * ((x + 1) * W + y), 2 * ((x + 1) * W + y) + 2)
                lesser_ji = source_x[j] @ source_gamma @ xi.conj().T
                jx[x, y] = 2.0 * np.imag(np.trace(hopping @ lesser_ji))
            if y + 1 < W:
                j = slice(2 * (x * W + y + 1), 2 * (x * W + y + 1) + 2)
                lesser_ji = source_x[j] @ source_gamma @ xi.conj().T
                jy[x, y] = 2.0 * np.imag(np.trace(hopping @ lesser_ji))
    return {
        "ldos": ldos,
        "source_injectivity": injectivity,
        "bond_current_x": jx,
        "bond_current_y": jy,
    }


def evaluate_hall_point(
    texture: np.ndarray,
    energy: float,
    J: float,
    t: float,
    contacts: list[Contact],
    *,
    eta: float = 1e-8,
    onsite_disorder: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate one energy with all diagnostics needed by the paper workflow."""
    transmission, diagnostics = transmission_matrix_sparse(
        texture, energy, J, t, contacts, eta=eta,
        onsite_disorder=onsite_disorder,
    )
    lb = four_terminal_observables(transmission)
    blocks, spin_labels, unitary_error = scattering_matrix_from_green(diagnostics)
    charge_s, spin_s = terminal_currents_from_scattering(
        blocks, spin_labels, lb["voltages"]
    )
    source_width = int(diagnostics["widths"][0])
    source_exchange = float(diagnostics["lead_Js"][0])
    channels = uniform_lead_channel_summary(
        energy, source_width, source_exchange, t
    )
    hall = four_terminal_hall_observables(
        lb["voltages"], charge_s, spin_s,
        source_channels=int(channels["n_total"]),
    )
    q_total, _ = lattice_topological_charge(texture)
    side_contact = contacts[2]
    q_window = windowed_topological_charge(
        texture, side_contact.start, side_contact.stop
    )
    t_lr = float(transmission[1, 0])
    bound_tolerance = 1e-8
    return {
        "energy": float(energy),
        "T": transmission.tolist(),
        "T_L_to_R": t_lr,
        "N_up": int(channels["n_up"]),
        "N_down": int(channels["n_down"]),
        "N_total": int(channels["n_total"]),
        "lead_polarization": float(channels["polarization"]),
        "topological_charge": float(q_total),
        "windowed_topological_charge": q_window,
        "Rxx_h_over_e2": lb["Rxx_h_over_e2"],
        "Rxy_h_over_e2": lb["Rxy_h_over_e2"],
        "hall_angle": lb["hall_angle"],
        **hall,
        "voltages": lb["voltages"].tolist(),
        "currents": lb["currents"].tolist(),
        "spin_currents_e_over_4pi": spin_s.tolist(),
        "scattering_charge_currents": charge_s.tolist(),
        "scattering_charge_mismatch": float(
            np.max(np.abs(charge_s - lb["currents"]))
        ),
        "current_conservation_error": lb["current_conservation_error"],
        "probe_current_error": lb["probe_current_error"],
        "gauge_invariance_error": lb["gauge_invariance_error"],
        "scattering_unitarity_error": unitary_error,
        "transmission_bound_ok": bool(
            -bound_tolerance <= t_lr <= channels["n_total"] + bound_tolerance
        ),
    }
