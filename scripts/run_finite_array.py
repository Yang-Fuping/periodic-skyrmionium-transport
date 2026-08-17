import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import clean_lead_modes, two_terminal_transmission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--Ny", type=int, default=1)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--emin", type=float, default=-3.0)
    p.add_argument("--emax", type=float, default=3.0)
    p.add_argument("--nenergy", type=int, default=121)
    args = p.parse_args()
    out = ROOT / "results" / "finite_array" / args.kind
    out.mkdir(parents=True, exist_ok=True)
    energy = np.linspace(args.emin, args.emax, args.nenergy)
    summary = {}
    for nx in args.Nx:
        texture = make_array_texture(args.kind, args.A, nx, args.Ny, args.R)
        T = two_terminal_transmission(texture, energy, args.J, 1.0, eta=args.eta)
        N = clean_lead_modes(energy, texture.shape[1], args.J, 1.0)
        if np.any(T > N + 1e-5):
            raise RuntimeError("Transmission exceeded the clean-lead mode count")
        np.savez_compressed(out / f"Nx{nx}_Ny{args.Ny}.npz", energy=energy, T=T, N=N)
        summary[str(nx)] = {"T_min": float(T.min()), "T_max": float(T.max())}
    (out / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
