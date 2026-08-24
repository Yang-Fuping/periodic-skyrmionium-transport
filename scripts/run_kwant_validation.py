"""Run independent Kwant/NumPy-SciPy transport cross-checks."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

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


def _errors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    difference = candidate - reference
    scale = max(float(np.linalg.norm(reference)), 1e-14)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / scale),
    }


def _two_terminal_case(name: str, texture: np.ndarray, energies, J: float) -> dict:
    energies = np.asarray(energies, dtype=float)
    system = build_two_terminal_system(texture, J, 1.0)
    kwant_T, kwant_N, unitarity = kwant_two_terminal_transmission(system, energies)
    negf_T = negf_two_terminal_transmission(texture, energies, J, 1.0, eta=1e-7)
    analytic_N = clean_lead_modes(energies, texture.shape[1], J, 1.0)
    error = _errors(negf_T, kwant_T)
    return {
        "name": name,
        "shape": list(texture.shape[:2]),
        "J_over_t": J,
        "energies": energies.tolist(),
        "kwant_T": kwant_T.tolist(),
        "negf_T_eta_1e-7": negf_T.tolist(),
        "kwant_channels": kwant_N.tolist(),
        "analytic_channels": analytic_N.tolist(),
        "channel_match": bool(np.array_equal(kwant_N, analytic_N)),
        "max_unitarity_error": float(np.max(unitarity)),
        **error,
    }


def _four_terminal_case(
    name: str,
    texture: np.ndarray,
    energy: float,
    J: float,
    *,
    probe_width: int,
    probe_start: int | None = None,
) -> dict:
    L, W, _ = texture.shape
    contacts = standard_four_contacts(
        L,
        W,
        probe_width,
        probe_J=0.0,
        probe_start=probe_start,
    )
    system = build_multiterminal_system(texture, J, 1.0, contacts)
    kwant_T, diagnostics = kwant_transmission_matrix(system, energy)
    negf_T, _ = transmission_matrix_sparse(
        texture, energy, J, 1.0, contacts, eta=1e-7
    )
    kwant_obs = four_terminal_observables(kwant_T)
    negf_obs = four_terminal_observables(negf_T)
    error = _errors(negf_T, kwant_T)
    return {
        "name": name,
        "shape": [L, W],
        "J_over_t": J,
        "energy": energy,
        "contacts": [contact.__dict__ for contact in contacts],
        "kwant_T": kwant_T.tolist(),
        "negf_T_eta_1e-7": negf_T.tolist(),
        "kwant_channels": diagnostics["channels"].tolist(),
        "kwant_unitarity_error": diagnostics["unitarity_error"],
        "kwant_hall": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in kwant_obs.items()
        },
        "negf_hall": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in negf_obs.items()
        },
        **error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-array-hall",
        action="store_true",
        help="also run the larger A=18, Ny=2, padded four-terminal case",
    )
    args = parser.parse_args()

    import kwant
    import scipy

    report = {
        "environment": {
            "python": platform.python_version(),
            "kwant": kwant.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "two_terminal": [],
        "four_terminal": [],
    }

    report["two_terminal"].append(
        _two_terminal_case(
            "uniform_small",
            make_texture("uniform", 8, 6, 2),
            [-1.0, 0.0, 1.0],
            1.5,
        )
    )
    report["two_terminal"].append(
        _two_terminal_case(
            "single_skyrmion_q_plus",
            make_texture("skyrmion_q_plus", 60, 30, 8),
            [0.0],
            1.5,
        )
    )
    report["two_terminal"].append(
        _two_terminal_case(
            "single_skyrmionium_q_zero",
            make_texture("skyrmionium_q_zero", 60, 30, 8),
            [0.0],
            1.5,
        )
    )
    report["two_terminal"].append(
        _two_terminal_case(
            "array_q_zero_A18_Nx1_Ny2",
            make_array_texture("skyrmionium_q_zero", 18, 1, 2, 8),
            [1.065, 1.0997714941836594, 1.15],
            5.0,
        )
    )

    for kind in ("uniform", "skyrmion_q_plus", "skyrmion_q_minus", "skyrmionium_q_zero"):
        report["four_terminal"].append(
            _four_terminal_case(
                f"single_{kind}",
                make_texture(kind, 24, 18, 6),
                0.0,
                1.5,
                probe_width=4,
            )
        )

    if args.include_array_hall:
        report["four_terminal"].append(
            _four_terminal_case(
                "array_q_zero_A18_Nx1_Ny2_py12",
                make_array_texture(
                    "skyrmionium_q_zero", 18, 1, 2, 8, padding=(0, 12)
                ),
                1.0997714941836594,
                5.0,
                probe_width=4,
                probe_start=7,
            )
        )

    all_cases = report["two_terminal"] + report["four_terminal"]
    report["summary"] = {
        "max_abs_error": max(case["max_abs"] for case in all_cases),
        "max_relative_l2_error": max(case["relative_l2"] for case in all_cases),
        "max_kwant_unitarity_error": max(
            case.get("max_unitarity_error", case.get("kwant_unitarity_error", 0.0))
            for case in all_cases
        ),
        "all_channel_counts_match": all(
            case.get("channel_match", True) for case in all_cases
        ),
    }
    output = ROOT / "results" / "kwant_validation"
    output.mkdir(parents=True, exist_ok=True)
    path = output / (
        "full_with_array_hall.json" if args.include_array_hall else "key_cases.json"
    )
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(path)


if __name__ == "__main__":
    main()
