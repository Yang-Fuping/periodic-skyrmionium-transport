"""Dense full-zone and local-extremum validation of the baseline minigap.

Only the two bands adjacent to the selected filling are diagonalized.  Each
full-zone row is checkpointed so an interrupted 61x61 calculation can resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from _bootstrap import ROOT
from skyrmion_transport.bloch import bloch_hamiltonian
from skyrmion_transport.textures import make_texture


def adjacent_bands(cell: np.ndarray, kx: float, ky: float, J: float,
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


def local_optima(cell: np.ndarray, A: int, J: float, n_occ: int) -> dict:
    # delta is a dimensionless displacement from the equivalent M point.
    # The interval reaches both sides of the periodically identified boundary.
    bound = 0.20

    def bands(delta: np.ndarray) -> tuple[float, float]:
        kx = (-np.pi + float(delta[0])) / A
        ky = (-np.pi + float(delta[1])) / A
        return adjacent_bands(cell, kx, ky, J, n_occ)

    def valence_objective(delta: np.ndarray) -> float:
        return -bands(delta)[0]

    def conduction_objective(delta: np.ndarray) -> float:
        return bands(delta)[1]

    def direct_objective(delta: np.ndarray) -> float:
        valence, conduction = bands(delta)
        return conduction - valence

    bounds = [(-bound, bound), (-bound, bound)]
    options = {"ftol": 1e-13, "gtol": 1e-10, "maxiter": 80}
    rv = minimize(valence_objective, np.zeros(2), method="L-BFGS-B",
                  bounds=bounds, options=options)
    rc = minimize(conduction_objective, np.zeros(2), method="L-BFGS-B",
                  bounds=bounds, options=options)
    rd = minimize(direct_objective, np.zeros(2), method="L-BFGS-B",
                  bounds=bounds, options=options)

    def record(result, sign: float = 1.0) -> dict:
        delta = np.asarray(result.x, dtype=float)
        k = (-np.pi + delta) / A
        return {
            "value": float(sign * result.fun),
            "delta_from_M_dimensionless": delta.tolist(),
            "k": k.tolist(),
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
        }

    return {
        "valence_max": record(rv, -1.0),
        "conduction_min": record(rc),
        "minimum_direct_gap": record(rd),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="skyrmionium_q_zero")
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8)
    parser.add_argument("--J", type=float, default=5)
    parser.add_argument("--n-occ", type=int, default=325)
    parser.add_argument("--nk", type=int, default=61)
    args = parser.parse_args()

    tag = (f"{args.kind}_A{args.A}_R{args.R:g}_J{args.J:g}_"
           f"n{args.n_occ}_nk{args.nk}")
    output = ROOT / "results" / "peer_review_gap_validation" / tag
    rows = output / "rows"
    rows.mkdir(parents=True, exist_ok=True)

    cell = make_texture(args.kind, args.A, args.A, args.R)
    ks = np.linspace(-np.pi / args.A, np.pi / args.A, args.nk,
                     endpoint=False)

    for ix, kx in enumerate(ks):
        row_path = rows / f"row_{ix:03d}.npz"
        if row_path.exists():
            continue
        row = np.empty((args.nk, 2), dtype=float)
        for iy, ky in enumerate(ks):
            row[iy] = adjacent_bands(cell, float(kx), float(ky), args.J,
                                     args.n_occ)
        np.savez_compressed(row_path, bands=row, kx=float(kx))
        print(f"completed row {ix + 1}/{args.nk}", flush=True)

    bands = np.stack([
        np.load(rows / f"row_{ix:03d}.npz")["bands"]
        for ix in range(args.nk)
    ])
    valence = bands[..., 0]
    conduction = bands[..., 1]
    direct = conduction - valence
    iv = np.unravel_index(np.argmax(valence), valence.shape)
    ic = np.unravel_index(np.argmin(conduction), conduction.shape)
    idirect = np.unravel_index(np.argmin(direct), direct.shape)

    local = local_optima(cell, args.A, args.J, args.n_occ)
    local_indirect = (local["conduction_min"]["value"] -
                      local["valence_max"]["value"])
    report = {
        "parameters": vars(args),
        "band_numbering": {
            "one_based_valence": args.n_occ,
            "one_based_conduction": args.n_occ + 1,
            "zero_based_valence": args.n_occ - 1,
            "zero_based_conduction": args.n_occ,
        },
        "full_grid": {
            "valence_max": float(valence[iv]),
            "valence_max_index": [int(iv[0]), int(iv[1])],
            "valence_max_k": [float(ks[iv[0]]), float(ks[iv[1]])],
            "conduction_min": float(conduction[ic]),
            "conduction_min_index": [int(ic[0]), int(ic[1])],
            "conduction_min_k": [float(ks[ic[0]]), float(ks[ic[1]])],
            "indirect_gap": float(np.min(conduction) - np.max(valence)),
            "minimum_direct_gap": float(direct[idirect]),
            "minimum_direct_gap_index": [int(idirect[0]), int(idirect[1])],
            "minimum_direct_gap_k": [float(ks[idirect[0]]),
                                      float(ks[idirect[1]])],
        },
        "local_M_optimization": {
            **local,
            "indirect_gap": float(local_indirect),
        },
        "acceptance": {
            "grid_gap_positive": bool(np.min(conduction) - np.max(valence) > 0),
            "local_gap_positive": bool(local_indirect > 0),
            "all_optimizers_succeeded": bool(all(
                entry["success"] for entry in local.values()
            )),
        },
    }
    np.savez_compressed(
        output / "band_edges.npz",
        k_values=ks,
        valence=valence,
        conduction=conduction,
        direct_gap=direct,
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
