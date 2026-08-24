"""Full-zone adjacent-band scan at fixed geometric filling ratio R/A."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from _bootstrap import ROOT

from skyrmion_transport.bloch import bloch_hamiltonian
from skyrmion_transport.textures import make_texture


def adjacent(cell, kx, ky, J, n_occ):
    values = eigh(
        bloch_hamiltonian(cell, kx, ky, J, 1.0),
        subset_by_index=(n_occ - 1, n_occ),
        eigvals_only=True,
        driver="evr",
        check_finite=False,
        overwrite_a=True,
    )
    return float(values[0]), float(values[1])


def refine(cell, A, J, n_occ, start, role, step):
    def objective(delta):
        valence, conduction = adjacent(
            cell, start[0] + delta[0], start[1] + delta[1], J, n_occ
        )
        if role == "valence":
            return -valence
        if role == "conduction":
            return conduction
        return conduction - valence

    result = minimize(
        objective,
        np.zeros(2),
        method="L-BFGS-B",
        bounds=[(-step, step), (-step, step)],
        options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 80},
    )
    sign = -1.0 if role == "valence" else 1.0
    return {
        "value": float(sign * result.fun),
        "k": (start + result.x).tolist(),
        "success": bool(result.success),
        "function_evaluations": int(result.nfev),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", type=int, nargs="+", default=[18, 20, 22, 24])
    parser.add_argument("--ratio", type=float, default=4.0 / 9.0)
    parser.add_argument("--J", type=float, default=5.0)
    parser.add_argument("--nk", type=int, default=11)
    args = parser.parse_args()
    output = ROOT / "results" / "fixed_filling_gap_scan" / f"ratio_{args.ratio:.8f}_J{args.J:g}_nk{args.nk}"
    output.mkdir(parents=True, exist_ok=True)
    report = {"parameters": vars(args), "cells": []}

    for A in args.A:
        R = args.ratio * A
        n_occ = A * A + 1
        cell = make_texture("skyrmionium_q_zero", A, A, R)
        ks = np.linspace(-np.pi / A, np.pi / A, args.nk, endpoint=False)
        rows = output / f"A{A}" / "rows"
        rows.mkdir(parents=True, exist_ok=True)
        for ix, kx in enumerate(ks):
            path = rows / f"row_{ix:03d}.npz"
            if not path.exists():
                bands = np.asarray([
                    adjacent(cell, float(kx), float(ky), args.J, n_occ)
                    for ky in ks
                ])
                np.savez_compressed(path, bands=bands)
            print(f"A={A}: row {ix + 1}/{args.nk}", flush=True)
        bands = np.stack([
            np.load(rows / f"row_{ix:03d}.npz")["bands"]
            for ix in range(args.nk)
        ])
        valence, conduction = bands[..., 0], bands[..., 1]
        direct = conduction - valence
        iv = np.unravel_index(np.argmax(valence), valence.shape)
        ic = np.unravel_index(np.argmin(conduction), conduction.shape)
        idirect = np.unravel_index(np.argmin(direct), direct.shape)
        step = 2.0 * np.pi / (A * args.nk)
        rv = refine(cell, A, args.J, n_occ,
                    np.asarray([ks[iv[0]], ks[iv[1]]]), "valence", step)
        rc = refine(cell, A, args.J, n_occ,
                    np.asarray([ks[ic[0]], ks[ic[1]]]), "conduction", step)
        rd = refine(cell, A, args.J, n_occ,
                    np.asarray([ks[idirect[0]], ks[idirect[1]]]), "direct", step)
        entry = {
            "A": A,
            "R": R,
            "R_over_A": R / A,
            "n_occ": n_occ,
            "grid": {
                "valence_max": float(valence.max()),
                "conduction_min": float(conduction.min()),
                "indirect_gap": float(conduction.min() - valence.max()),
                "minimum_direct_gap": float(direct.min()),
            },
            "refined": {
                "valence_max": rv,
                "conduction_min": rc,
                "minimum_direct_gap": rd,
                "indirect_gap": float(rc["value"] - rv["value"]),
                "midgap_energy": float((rc["value"] + rv["value"]) / 2.0),
            },
        }
        report["cells"].append(entry)
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(entry, indent=2), flush=True)


if __name__ == "__main__":
    main()
