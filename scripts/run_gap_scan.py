import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import find_global_gaps, k_grid_eigenvalues
from skyrmion_transport.textures import make_texture


def main():
    p = argparse.ArgumentParser(description="Full-zone coarse scan of indirect supercell gaps")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, nargs="+", default=[18, 20, 24, 30])
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--nk", type=int, default=7)
    p.add_argument("--minimum", type=float, default=1e-4)
    p.add_argument("--interior-fraction", type=float, default=0.02)
    args = p.parse_args()
    root = ROOT / "results" / "gap_scan" / args.kind / f"R{args.R:g}_J{args.J:g}_nk{args.nk}"
    root.mkdir(parents=True, exist_ok=True)
    report = {"parameters": vars(args), "cells": []}
    for A in args.A:
        if 2 * args.R >= A:
            report["cells"].append({"A": A, "skipped": "requires 2R < A"})
            continue
        cell = make_texture(args.kind, A, A, args.R)
        eig = k_grid_eigenvalues(cell, args.nk, args.J, 1.0)
        all_gaps = find_global_gaps(eig, args.minimum)
        nbands = eig.shape[-1]
        lo = int(np.ceil(args.interior_fraction * nbands))
        hi = int(np.floor((1.0 - args.interior_fraction) * nbands))
        interior = []
        for gap in all_gaps:
            if lo <= gap["n_occ"] <= hi:
                gap = dict(gap)
                gap["midgap_energy"] = 0.5 * (gap["valence_max"] + gap["conduction_min"])
                interior.append(gap)
        interior.sort(key=lambda item: item["indirect_gap"], reverse=True)
        np.savez_compressed(root / f"A{A}.npz", eigenvalues=eig)
        entry = {
            "A": A,
            "nbands": nbands,
            "largest_interior_candidates": interior[:20],
            "number_of_interior_candidates": len(interior),
        }
        report["cells"].append(entry)
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(entry, indent=2), flush=True)


if __name__ == "__main__":
    main()
