import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import (
    direct_indirect_gap, fhs_chern_subspace, k_grid_eigenvalues,
)
from skyrmion_transport.textures import make_texture


def main():
    p = argparse.ArgumentParser(description="Chern convergence for an isolated occupied subspace")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--n-occ", type=int, required=True)
    p.add_argument("--nk", type=int, nargs="+", default=[31, 51, 71])
    args = p.parse_args()
    cell = make_texture(args.kind, args.A, args.A, args.R)
    tag = (f"{args.kind}_A{args.A}_R{args.R:g}_J{args.J:g}"
           f"_n{args.n_occ}")
    out = ROOT / "results" / "chern" / tag
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        previous = report.get("parameters", {})
        identity = ("kind", "A", "R", "J", "t", "n_occ")
        if any(previous.get(key) != vars(args).get(key) for key in identity):
            raise RuntimeError("Existing Chern report has incompatible parameters")
    else:
        report = {"parameters": vars(args), "convergence": []}
    report["parameters"] = {
        **vars(args),
        "nk": sorted(set(report.get("parameters", {}).get("nk", [])) |
                     set(args.nk)),
    }
    for nk in args.nk:
        eig = k_grid_eigenvalues(cell, nk, args.J, args.t)
        gap = direct_indirect_gap(eig, args.n_occ)
        if gap["direct_gap"] <= 1e-8:
            raise RuntimeError(f"Occupied subspace closes its direct gap at nk={nk}")
        chern, flux = fhs_chern_subspace(
            cell, args.n_occ, nk, args.J, args.t
        )
        np.savez_compressed(
            out / f"nk{nk}.npz",
            berry_flux=flux,
            k_values=np.linspace(-np.pi / args.A, np.pi / args.A, nk,
                                 endpoint=False),
            chern=chern,
        )
        entry = {
            "nk": nk,
            "chern": chern,
            "nearest_integer": int(np.rint(chern)),
            "integer_residual": float(abs(chern - np.rint(chern))),
            **gap,
        }
        report["convergence"] = [
            item for item in report.get("convergence", []) if item["nk"] != nk
        ] + [entry]
        report["convergence"].sort(key=lambda item: item["nk"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
