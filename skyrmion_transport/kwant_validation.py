"""Independent Kwant builders used only for cross-validation.

The production calculations remain in the transparent NumPy/SciPy solvers.
This module deliberately rebuilds the device Hamiltonian and all semi-infinite
leads with Kwant.  Only the frozen magnetization array and contact geometry are
shared, which keeps the comparison independent without risking texture or
coordinate mismatches.

Kwant is an optional dependency and is imported lazily.  On Windows, run this
module from an activated Conda environment so that BLAS/MUMPS DLLs are visible.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .model import S0, SZ, exchange_onsite
from .multiterminal import Contact


def _kwant():
    try:
        import kwant
    except ImportError as exc:
        raise RuntimeError(
            "Kwant is optional. Run this validation from the dedicated "
            "Conda environment, for example `conda run -n kwant-validate`."
        ) from exc
    return kwant


def _validated_inputs(
    texture: np.ndarray,
    onsite_disorder: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    texture = np.asarray(texture, dtype=float)
    if texture.ndim != 3 or texture.shape[2] != 3:
        raise ValueError("texture must have shape (L, W, 3)")
    L, W, _ = texture.shape
    if onsite_disorder is None:
        onsite_disorder = np.zeros((L, W), dtype=float)
    onsite_disorder = np.asarray(onsite_disorder, dtype=float)
    if onsite_disorder.shape != (L, W):
        raise ValueError("onsite_disorder must have shape (L, W)")
    return texture, onsite_disorder


def _finite_device_builder(
    texture: np.ndarray,
    J: float,
    t: float,
    onsite_disorder: np.ndarray | None,
):
    kwant = _kwant()
    texture, onsite_disorder = _validated_inputs(texture, onsite_disorder)
    L, W, _ = texture.shape
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()
    for x in range(L):
        for y in range(W):
            syst[lat(x, y)] = exchange_onsite(
                texture[x, y], J, onsite_disorder[x, y]
            )
            if x:
                syst[lat(x - 1, y), lat(x, y)] = -t * S0
            if y:
                syst[lat(x, y - 1), lat(x, y)] = -t * S0
    return kwant, lat, syst


def _strip_lead(
    lat,
    sites: Iterable[tuple[int, int]],
    translation: tuple[int, int],
    lead_J: float,
    t: float,
):
    """Build one square-lattice strip lead from its interface unit cell."""
    kwant = _kwant()
    sites = tuple((int(x), int(y)) for x, y in sites)
    if not sites:
        raise ValueError("A lead needs at least one interface site")
    site_set = set(sites)
    dx, dy = translation
    lead = kwant.Builder(kwant.TranslationalSymmetry(translation))
    onsite = -lead_J * SZ
    hopping = -t * S0
    for x, y in sites:
        lead[lat(x, y)] = onsite

    # Kwant requires both endpoints of an intracell hopping to be present in
    # the fundamental domain.  Register the full strip cross section first,
    # then add longitudinal and transverse hoppings in a second pass.
    for x, y in sites:
        lead[lat(x, y), lat(x + dx, y + dy)] = hopping
        for nx, ny in ((x + 1, y), (x, y + 1)):
            if (nx, ny) in site_set:
                lead[lat(x, y), lat(nx, ny)] = hopping
    return lead


def build_two_terminal_system(
    texture: np.ndarray,
    J: float,
    t: float,
    *,
    lead_J: float | None = None,
    onsite_disorder: np.ndarray | None = None,
):
    """Finalize a two-terminal device with uniform ``+z`` strip leads."""
    kwant, lat, syst = _finite_device_builder(texture, J, t, onsite_disorder)
    L, W, _ = np.asarray(texture).shape
    lead_J = J if lead_J is None else float(lead_J)
    left = _strip_lead(lat, ((0, y) for y in range(W)), (-1, 0), lead_J, t)
    right = _strip_lead(
        lat, ((L - 1, y) for y in range(W)), (1, 0), lead_J, t
    )
    syst.attach_lead(left)
    syst.attach_lead(right)
    return syst.finalized()


def two_terminal_transmission(
    finalized_system,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total transmission, lead channel counts and S-unitarity errors."""
    kwant = _kwant()
    transmissions = []
    channels = []
    unitarity = []
    for energy in np.atleast_1d(energies).astype(float):
        smatrix = kwant.smatrix(finalized_system, energy, check_hermiticity=True)
        transmissions.append(float(smatrix.transmission(1, 0)))
        channels.append(int(smatrix.num_propagating(0)))
        S = np.asarray(smatrix.data)
        identity = np.eye(S.shape[1], dtype=complex)
        unitarity.append(float(np.max(np.abs(S.conj().T @ S - identity))))
    return (
        np.asarray(transmissions),
        np.asarray(channels, dtype=int),
        np.asarray(unitarity),
    )


def _contact_sites(contact: Contact, L: int, W: int) -> tuple[tuple[int, int], ...]:
    if contact.edge == "left":
        return tuple((0, y) for y in range(contact.start, contact.stop))
    if contact.edge == "right":
        return tuple((L - 1, y) for y in range(contact.start, contact.stop))
    if contact.edge == "bottom":
        return tuple((x, 0) for x in range(contact.start, contact.stop))
    if contact.edge == "top":
        return tuple((x, W - 1) for x in range(contact.start, contact.stop))
    raise ValueError(f"Unknown edge {contact.edge!r}")


def _translation_for_edge(edge: str) -> tuple[int, int]:
    return {
        "left": (-1, 0),
        "right": (1, 0),
        "bottom": (0, -1),
        "top": (0, 1),
    }[edge]


def build_multiterminal_system(
    texture: np.ndarray,
    J: float,
    t: float,
    contacts: list[Contact],
    *,
    onsite_disorder: np.ndarray | None = None,
):
    """Finalize a device with leads ordered exactly as ``contacts``."""
    _, lat, syst = _finite_device_builder(texture, J, t, onsite_disorder)
    L, W, _ = np.asarray(texture).shape
    occupied: set[tuple[int, int]] = set()
    for contact in contacts:
        if not np.isclose(contact.coupling_scale, 1.0):
            raise NotImplementedError(
                "The independent Kwant validation builder currently supports "
                "only the baseline device-lead coupling scale 1."
            )
        sites = _contact_sites(contact, L, W)
        overlap = occupied.intersection(sites)
        if overlap:
            raise ValueError(f"Contacts overlap at {sorted(overlap)!r}")
        occupied.update(sites)
        lead_J = J if contact.lead_J is None else float(contact.lead_J)
        lead = _strip_lead(
            lat, sites, _translation_for_edge(contact.edge), lead_J, t
        )
        syst.attach_lead(lead)
    return syst.finalized()


def transmission_matrix(finalized_system, energy: float) -> tuple[np.ndarray, dict]:
    """Return Kwant ``T[p,q]`` for transmission from lead ``q`` to ``p``."""
    kwant = _kwant()
    smatrix = kwant.smatrix(finalized_system, float(energy), check_hermiticity=True)
    nlead = len(finalized_system.leads)
    T = np.zeros((nlead, nlead), dtype=float)
    for p in range(nlead):
        for q in range(nlead):
            if p != q:
                T[p, q] = float(smatrix.transmission(p, q))
    S = np.asarray(smatrix.data)
    identity = np.eye(S.shape[1], dtype=complex)
    diagnostics = {
        "channels": np.asarray(
            [smatrix.num_propagating(p) for p in range(nlead)], dtype=int
        ),
        "unitarity_error": float(np.max(np.abs(S.conj().T @ S - identity))),
    }
    return T, diagnostics
