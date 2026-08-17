import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import two_terminal_transmission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, default=2)
    p.add_argument("--Ny", type=int, default=1)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, nargs="+", default=[1e-6, 1e-7, 1e-8])
    p.add_argument("--emin", type=float, default=-3.0)
    p.add_argument("--emax", type=float, default=3.0)
    p.add_argument("--nenergy", type=int, default=121)
    args = p.parse_args()
    texture = make_array_texture(args.kind, args.A, args.Nx, args.Ny, args.R)
    coarse = np.linspace(args.emin, args.emax, args.nenergy)
    fine = np.linspace(args.emin, args.emax, 2 * args.nenergy - 1)
    arrays = {"energy_coarse": coarse, "energy_fine": fine}
    report = {"parameters": vars(args), "eta_differences": {}}
    reference = None
    for eta in args.eta:
        Tc = two_terminal_transmission(texture, coarse, args.J, 1.0, eta=eta)
        Tf = two_terminal_transmission(texture, fine, args.J, 1.0, eta=eta)
        key = f"eta_{eta:.0e}"
        arrays[f"{key}_coarse"] = Tc
        arrays[f"{key}_fine"] = Tf
        interpolated = np.interp(fine, coarse, Tc)
        report["eta_differences"][f"{key}_energy_interpolation_max"] = float(
            np.max(np.abs(Tf - interpolated))
        )
        report["eta_differences"][f"{key}_integral_difference"] = float(abs(
            np.trapezoid(Tf, fine) - np.trapezoid(Tc, coarse)
        ))
        if reference is None:
            reference = Tc
        else:
            report["eta_differences"][f"{key}_vs_first"] = float(np.max(np.abs(Tc - reference)))
    out = ROOT / "results" / "convergence" / args.kind
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **arrays)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
