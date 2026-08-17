import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import fhs_chern_band
from skyrmion_transport.textures import make_texture


def main():
    p = argparse.ArgumentParser(description="Chern convergence of one isolated Bloch band")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--J", type=float, default=5)
    p.add_argument("--band", type=int, required=True, help="zero-based band index")
    p.add_argument("--nk", type=int, nargs="+", default=[31, 51, 71])
    args = p.parse_args()
    cell = make_texture(args.kind, args.A, args.A, args.R)
    out = ROOT / "results" / "chern_band" / f"{args.kind}_A{args.A}_J{args.J:g}_b{args.band}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"parameters": vars(args), "convergence": []}
    for nk in args.nk:
        chern, flux = fhs_chern_band(cell, args.band, nk, args.J, 1.0)
        np.savez_compressed(out / f"nk{nk}.npz", berry_flux=flux, chern=chern)
        report["convergence"].append({"nk": nk, "chern": chern})
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["convergence"][-1]), flush=True)


if __name__ == "__main__":
    main()
