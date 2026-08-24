"""Independent Kwant/NEGF regression tests.

The module is skipped when Kwant is not installed, so the lightweight main
test suite remains usable.  Run it in the environment defined by
``environment-kwant.yml`` for the full cross-validation suite.
"""

import numpy as np
import pytest

pytest.importorskip("kwant")

from skyrmion_transport.kwant_validation import (
    build_multiterminal_system,
    build_two_terminal_system,
    transmission_matrix as kwant_transmission_matrix,
    two_terminal_transmission as kwant_two_terminal_transmission,
)
from skyrmion_transport.multiterminal import (
    four_terminal_observables,
    standard_four_contacts,
    transmission_matrix_sparse,
)
from skyrmion_transport.textures import make_array_texture, make_texture
from skyrmion_transport.transport import (
    clean_lead_modes,
    two_terminal_transmission as negf_two_terminal_transmission,
)


def _relative_error(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return np.linalg.norm(candidate - reference) / max(
        np.linalg.norm(reference), 1e-14
    )


def _kwant_and_negf_four_terminal(texture, energy, J, contacts):
    system = build_multiterminal_system(texture, J, 1.0, contacts)
    kwant_T, diagnostics = kwant_transmission_matrix(system, energy)
    negf_T, _ = transmission_matrix_sparse(
        texture, energy, J, 1.0, contacts, eta=1e-7
    )
    assert diagnostics["unitarity_error"] < 5e-10
    assert _relative_error(negf_T, kwant_T) < 1e-5
    return kwant_T, negf_T


def test_kwant_uniform_channels_and_transmission():
    texture = make_texture("uniform", 8, 6, 2)
    energies = np.array([-1.0, 0.0, 1.0])
    system = build_two_terminal_system(texture, 1.5, 1.0)
    kwant_T, channels, unitarity = kwant_two_terminal_transmission(
        system, energies
    )
    analytic = clean_lead_modes(energies, 6, 1.5, 1.0)
    negf_T = negf_two_terminal_transmission(
        texture, energies, 1.5, 1.0, eta=1e-7
    )
    np.testing.assert_array_equal(channels, analytic)
    np.testing.assert_allclose(kwant_T, analytic, atol=1e-11, rtol=0)
    assert _relative_error(negf_T, kwant_T) < 1e-5
    assert np.max(unitarity) < 5e-10


@pytest.mark.parametrize(
    "kind",
    ["skyrmion_q_plus", "skyrmionium_q_zero"],
)
def test_kwant_single_texture_matches_negf(kind):
    texture = make_texture(kind, 60, 30, 8)
    system = build_two_terminal_system(texture, 1.5, 1.0)
    kwant_T, channels, unitarity = kwant_two_terminal_transmission(
        system, np.array([0.0])
    )
    negf_T = negf_two_terminal_transmission(
        texture, np.array([0.0]), 1.5, 1.0, eta=1e-7
    )
    assert channels[0] == 34
    assert _relative_error(negf_T, kwant_T) < 1e-5
    assert unitarity[0] < 5e-10


def test_kwant_four_terminal_q_reversal():
    L, W, R = 24, 18, 6
    contacts = standard_four_contacts(L, W, 4, probe_J=0.0)
    hall = {}
    for kind in ("uniform", "skyrmion_q_plus", "skyrmion_q_minus"):
        texture = make_texture(kind, L, W, R)
        kwant_T, _ = _kwant_and_negf_four_terminal(
            texture, 0.0, 1.5, contacts
        )
        hall[kind] = four_terminal_observables(kwant_T)["hall_angle"]
    assert abs(hall["uniform"]) < 1e-12
    assert abs(hall["skyrmion_q_plus"] + hall["skyrmion_q_minus"]) < 1e-12


@pytest.mark.slow
def test_kwant_representative_array_hall_matches_negf():
    texture = make_array_texture(
        "skyrmionium_q_zero", 18, 1, 2, 8, padding=(0, 12)
    )
    contacts = standard_four_contacts(
        texture.shape[0],
        texture.shape[1],
        4,
        probe_J=0.0,
        probe_start=7,
    )
    kwant_T, negf_T = _kwant_and_negf_four_terminal(
        texture, 1.0997714941836594, 5.0, contacts
    )
    kwant_hall = four_terminal_observables(kwant_T)
    negf_hall = four_terminal_observables(negf_T)
    assert abs(kwant_hall["hall_angle"] - negf_hall["hall_angle"]) < 1e-6
    assert kwant_hall["current_conservation_error"] < 1e-12
    assert kwant_hall["probe_current_error"] < 1e-12
