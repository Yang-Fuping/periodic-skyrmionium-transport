import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.statistics import finite_temperature_conductance
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import two_terminal_transmission


def main():
    p = argparse.ArgumentParser(description="Clean finite-temperature conductance near a minigap")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, default=4)
    p.add_argument("--Ny", type=int, default=2)
    p.add_argument("--J", type=float, default=5)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--emin", type=float, default=0.85)
    p.add_argument("--emax", type=float, default=1.35)
    p.add_argument("--nenergy", type=int, default=251)
    p.add_argument("--ef", type=float, nargs="+", default=[1.0997714941836594])
    p.add_argument("--kbt", type=float, nargs="+", default=[0.002, 0.005, 0.01, 0.02])
    args = p.parse_args()
    energy = np.linspace(args.emin, args.emax, args.nenergy)
    texture = make_array_texture(args.kind, args.A, args.Nx, args.Ny, args.R)
    T = two_terminal_transmission(texture, energy, args.J, 1.0, eta=args.eta)
    report = {"parameters": vars(args), "conductance_e2_over_h": {}}
    arrays = {"energy": energy, "transmission": T, "ef": np.asarray(args.ef)}
    for kbt in args.kbt:
        G = finite_temperature_conductance(energy, T, np.asarray(args.ef), kbt)
        arrays[f"G_kBT_{kbt:g}"] = G
        report["conductance_e2_over_h"][f"kBT_{kbt:g}"] = G.tolist()
    out = ROOT / "results" / "temperature" / f"{args.kind}_A{args.A}_J{args.J:g}_Nx{args.Nx}_Ny{args.Ny}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **arrays)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
