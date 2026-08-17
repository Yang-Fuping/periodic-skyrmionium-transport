import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.bloch import (
    band_structure, direct_indirect_gap, fhs_chern_subspace, find_global_gaps,
    high_symmetry_path, k_grid_eigenvalues,
)
from skyrmion_transport.textures import make_texture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--nk", type=int, default=11)
    p.add_argument("--n-occ", type=int, default=None)
    p.add_argument("--chern", action="store_true")
    args = p.parse_args()
    if 2 * args.R >= args.A:
        raise SystemExit("Require 2R < A; use A=18 or larger for R=8")
    out = ROOT / "results" / "bloch" / f"{args.kind}_A{args.A}_nk{args.nk}"
    out.mkdir(parents=True, exist_ok=True)
    cell = make_texture(args.kind, args.A, args.A, args.R)
    kpath, distance, ticks, labels = high_symmetry_path(args.A)
    bands = band_structure(cell, kpath, args.J, args.t)
    grid = k_grid_eigenvalues(cell, args.nk, args.J, args.t)
    n_occ = args.n_occ if args.n_occ is not None else grid.shape[-1] // 2
    gaps = direct_indirect_gap(grid, n_occ)
    candidate_gaps = find_global_gaps(grid)
    payload = dict(cell=cell, kpath=kpath, distance=distance, ticks=ticks,
                   labels=np.asarray(labels), bands=bands, grid_eigenvalues=grid)
    report = {"kind": args.kind, "A": args.A, "R": args.R, "J": args.J,
              "t": args.t, "nk": args.nk, "n_occ": n_occ,
              "candidate_global_gaps": candidate_gaps, **gaps}
    if args.chern:
        if gaps["direct_gap"] <= 1e-8:
            raise RuntimeError("Selected occupied subspace is not isolated by a direct gap; Chern number is undefined")
        chern, flux = fhs_chern_subspace(cell, n_occ, args.nk, args.J, args.t)
        payload["berry_flux"] = flux
        report["chern"] = chern
    np.savez_compressed(out / "bloch_data.npz", **payload)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
