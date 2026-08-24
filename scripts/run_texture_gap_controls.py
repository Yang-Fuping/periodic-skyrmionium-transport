"""Compare the baseline same-filling gap across magnetic-texture controls."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from _bootstrap import ROOT
from skyrmion_transport.bloch import bloch_hamiltonian
from skyrmion_transport.textures import (
    lattice_topological_charge,
    make_texture,
)


KINDS = (
    "uniform",
    "skyrmion_q_plus",
    "skyrmion_q_minus",
    "skyrmionium_q_zero",
    "skyrmionium_q_zero_quintic",
    "nonwinding_double_wall_q_zero",
)


def adjacent(cell: np.ndarray, kx: float, ky: float, J: float,
             n_occ: int) -> tuple[float, float]:
    values = eigh(
        bloch_hamiltonian(cell, kx, ky, J, 1.0),
        subset_by_index=(n_occ - 1, n_occ),
        eigvals_only=True,
        driver="evr",
        check_finite=False,
        overwrite_a=True,
    )
    return float(values[0]), float(values[1])


def refine(cell: np.ndarray, A: int, J: float, n_occ: int,
           k0: np.ndarray, select: str, grid_step: float) -> dict:
    def objective(delta: np.ndarray) -> float:
        k = k0 + delta
        valence, conduction = adjacent(cell, float(k[0]), float(k[1]), J,
                                       n_occ)
        if select == "valence":
            return -valence
        if select == "conduction":
            return conduction
        return conduction - valence

    result = minimize(
        objective,
        np.zeros(2),
        method="L-BFGS-B",
        bounds=[(-grid_step, grid_step), (-grid_step, grid_step)],
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 80},
    )
    sign = -1.0 if select == "valence" else 1.0
    return {
        "value": float(sign * result.fun),
        "k": (k0 + result.x).tolist(),
        "success": bool(result.success),
        "function_evaluations": int(result.nfev),
    }


def fourier_summary(cell: np.ndarray) -> dict:
    A = cell.shape[0]
    transformed = np.fft.fft2(cell, axes=(0, 1)) / (A * A)
    power = np.sum(np.abs(transformed) ** 2, axis=-1)
    power[0, 0] = 0.0
    flat_order = np.argsort(power.ravel())[::-1]
    leading = []
    for flat in flat_order[:8]:
        gx, gy = np.unravel_index(flat, power.shape)
        # Express FFT indices in the signed reciprocal-supercell convention.
        sx = gx if gx <= A // 2 else gx - A
        sy = gy if gy <= A // 2 else gy - A
        leading.append({
            "harmonic": [int(sx), int(sy)],
            "total_power": float(power[gx, gy]),
            "mz_power": float(abs(transformed[gx, gy, 2]) ** 2),
            "mxy_power": float(abs(transformed[gx, gy, 0]) ** 2 +
                               abs(transformed[gx, gy, 1]) ** 2),
        })
    return {
        "mean_m": np.mean(cell, axis=(0, 1)).tolist(),
        "nonzero_harmonic_power": float(power.sum()),
        "leading_harmonics": leading,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8)
    parser.add_argument("--J", type=float, default=5)
    parser.add_argument("--n-occ", type=int, default=325)
    parser.add_argument("--nk", type=int, default=21)
    args = parser.parse_args()

    output = (ROOT / "results" / "texture_gap_controls" /
              f"A{args.A}_R{args.R:g}_J{args.J:g}_n{args.n_occ}_nk{args.nk}")
    output.mkdir(parents=True, exist_ok=True)
    ks = np.linspace(-np.pi / args.A, np.pi / args.A, args.nk,
                     endpoint=False)
    grid_step = 2.0 * np.pi / (args.A * args.nk)
    report = {"parameters": vars(args), "textures": {}}

    for kind in KINDS:
        cell = make_texture(kind, args.A, args.A, args.R)
        rows_dir = output / kind / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)
        for ix, kx in enumerate(ks):
            path = rows_dir / f"row_{ix:03d}.npz"
            if path.exists():
                continue
            row = np.asarray([
                adjacent(cell, float(kx), float(ky), args.J, args.n_occ)
                for ky in ks
            ])
            np.savez_compressed(path, bands=row)
            print(f"{kind}: completed row {ix + 1}/{args.nk}", flush=True)

        bands = np.stack([
            np.load(rows_dir / f"row_{ix:03d}.npz")["bands"]
            for ix in range(args.nk)
        ])
        valence, conduction = bands[..., 0], bands[..., 1]
        direct = conduction - valence
        iv = np.unravel_index(np.argmax(valence), valence.shape)
        ic = np.unravel_index(np.argmin(conduction), conduction.shape)
        idirect = np.unravel_index(np.argmin(direct), direct.shape)
        rv = refine(cell, args.A, args.J, args.n_occ,
                    np.asarray([ks[iv[0]], ks[iv[1]]]), "valence", grid_step)
        rc = refine(cell, args.A, args.J, args.n_occ,
                    np.asarray([ks[ic[0]], ks[ic[1]]]), "conduction", grid_step)
        rd = refine(cell, args.A, args.J, args.n_occ,
                    np.asarray([ks[idirect[0]], ks[idirect[1]]]), "direct", grid_step)
        q, _ = lattice_topological_charge(cell)
        entry = {
            "lattice_topological_charge": float(q),
            "grid": {
                "valence_max": float(valence.max()),
                "conduction_min": float(conduction.min()),
                "indirect_gap": float(conduction.min() - valence.max()),
                "minimum_direct_gap": float(direct.min()),
            },
            "locally_refined": {
                "valence_max": rv,
                "conduction_min": rc,
                "minimum_direct_gap": rd,
                "indirect_gap": float(rc["value"] - rv["value"]),
            },
            "fourier": fourier_summary(cell),
        }
        report["textures"][kind] = entry
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps({kind: entry}, indent=2), flush=True)

    report["comparisons"] = {
        "q_plus_minus_refined_gap_difference": abs(
            report["textures"]["skyrmion_q_plus"]["locally_refined"]["indirect_gap"] -
            report["textures"]["skyrmion_q_minus"]["locally_refined"]["indirect_gap"]
        ),
        "skyrmionium_to_q_plus_gap_ratio": (
            report["textures"]["skyrmionium_q_zero"]["locally_refined"]["indirect_gap"] /
            report["textures"]["skyrmion_q_plus"]["locally_refined"]["indirect_gap"]
        ),
        "same_mz_profile_control": (
            "skyrmionium_q_zero and nonwinding_double_wall_q_zero have identical m_z "
            "site by site; their gap difference isolates the role of in-plane winding "
            "within this analytic control pair."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
