"""Validate finite-array attenuation against stable strip complex bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.bloch import (
    slowest_strip_mode_boundary_green,
    strip_bloch_multipliers_boundary_green,
)
from skyrmion_transport.textures import make_array_texture


def encode_complex(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def serializable_mode(mode: dict) -> dict:
    result = dict(mode)
    for key in ("multiplier", "reciprocal_partner"):
        result[key] = encode_complex(result[key])
    if not np.isfinite(result["xi_transmission_a"]):
        result["xi_transmission_a"] = None
    return result


def finite_scaling_for_width(width_cells: int, energy: float, A: int) -> tuple[dict, Path]:
    folder = ROOT / "results" / "length_scaling"
    if width_cells == 2:
        path = folder / "peer_review_A18_R8_J5_Ny2_Nx1_2_3_4_6_8.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = min(payload["fits"], key=lambda item: abs(float(item) - energy))
        nx = np.asarray(payload["parameters"]["Nx"], dtype=int)
        transmission = np.asarray(payload["transmissions"], dtype=float)[:, 0]
        global_fit = payload["fits"][key]
        asymptotic_fit = global_fit
    else:
        path = ROOT / "results" / "peer_review_complex_band" / f"finite_length_Ny{width_cells}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        nx = np.asarray(payload["Nx"], dtype=int)
        transmission = np.asarray(payload["transmission"], dtype=float)
        global_fit = payload["all_point_fit"]
        asymptotic_fit = payload["asymptotic_fit_Nx_ge_3"]
    terminal_xi = -A * float(nx[-1] - nx[-2]) / np.log(
        transmission[-1] / transmission[-2]
    )
    return {
        "Nx": nx.tolist(),
        "transmission": transmission.tolist(),
        "global_fit": global_fit,
        "asymptotic_fit": asymptotic_fit,
        "terminal_pair_Nx": [int(nx[-2]), int(nx[-1])],
        "terminal_pair_xi_a": float(terminal_xi),
    }, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8.0)
    parser.add_argument("--J", type=float, default=5.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--Ny", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=[1.065, 1.09977149418366, 1.15],
    )
    args = parser.parse_args()
    center = args.energies[len(args.energies) // 2]
    output = ROOT / "results" / "peer_review_complex_band"
    output.mkdir(parents=True, exist_ok=True)

    energy_rows = []
    reference_texture = make_array_texture(
        "skyrmionium_q_zero", args.A, 1, args.Ny[0], args.R
    )
    for energy in args.energies:
        values, residuals = strip_bloch_multipliers_boundary_green(
            reference_texture, energy, args.J, args.t
        )
        mode = slowest_strip_mode_boundary_green(
            reference_texture, energy, args.J, args.t
        )
        energy_rows.append({
            "energy_t": energy,
            "mode": serializable_mode(mode),
            "multipliers": [encode_complex(value) for value in values],
            "qep_residuals": residuals.tolist(),
        })

    width_rows = []
    for width_cells in args.Ny:
        texture = make_array_texture(
            "skyrmionium_q_zero", args.A, 1, width_cells, args.R
        )
        mode = slowest_strip_mode_boundary_green(texture, center, args.J, args.t)
        finite_scaling, finite_path = finite_scaling_for_width(
            width_cells, center, args.A
        )
        xi_complex = float(mode["xi_transmission_a"])
        xi_finite = float(finite_scaling["global_fit"]["decay_length"])
        xi_asymptotic = float(finite_scaling["asymptotic_fit"]["decay_length"])
        xi_terminal = float(finite_scaling["terminal_pair_xi_a"])
        width_rows.append({
            "Ny": width_cells,
            "strip_width_a": args.A * width_cells,
            "complex_band": serializable_mode(mode),
            "finite_array_global_fit_xi_a": xi_finite,
            "finite_array_global_fit_r2": float(
                finite_scaling["global_fit"]["r2"]
            ),
            "finite_array_asymptotic_fit_xi_a": xi_asymptotic,
            "finite_array_asymptotic_fit_r2": float(
                finite_scaling["asymptotic_fit"]["r2"]
            ),
            "finite_array_terminal_pair_Nx": finite_scaling["terminal_pair_Nx"],
            "finite_array_terminal_pair_xi_a": xi_terminal,
            "relative_global_fit_xi_difference": abs(xi_finite / xi_complex - 1.0),
            "relative_asymptotic_fit_xi_difference": abs(
                xi_asymptotic / xi_complex - 1.0
            ),
            "relative_terminal_pair_xi_difference": abs(
                xi_terminal / xi_complex - 1.0
            ),
            "finite_array_source": str(finite_path.relative_to(ROOT)),
        })
        partial = {
            "parameters": vars(args),
            "energy_classification_Ny": args.Ny[0],
            "energy_rows": energy_rows,
            "width_rows": width_rows,
        }
        (output / "report.partial.json").write_text(
            json.dumps(partial, indent=2), encoding="utf-8"
        )

    report = {
        "parameters": vars(args),
        "method": (
            "boundary Green-function reduction followed by dense generalized "
            "QZ solution of the quadratic Bloch equation"
        ),
        "energy_classification_Ny": args.Ny[0],
        "energy_rows": energy_rows,
        "width_rows": width_rows,
        "acceptance": {
            "maximum_selected_qep_residual": max(
                row["complex_band"]["selected_qep_residual"] for row in width_rows
            ),
            "maximum_reciprocal_pair_error": max(
                row["complex_band"]["reciprocal_pair_error"] for row in width_rows
            ),
            "maximum_relative_terminal_pair_xi_difference": max(
                row["relative_terminal_pair_xi_difference"] for row in width_rows
            ),
            "Ny2_relative_global_fit_xi_difference": width_rows[0][
                "relative_global_fit_xi_difference"
            ],
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output / "report.json"),
        "width_rows": width_rows,
        "energy_modes": [row["mode"] for row in energy_rows],
    }, indent=2))


if __name__ == "__main__":
    main()
