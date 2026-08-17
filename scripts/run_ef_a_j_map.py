import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import two_terminal_transmission


def main():
    p = argparse.ArgumentParser(description="Focused (EF,A,J/t) longitudinal transport map")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, nargs="+", default=[18, 20, 24])
    p.add_argument("--J", type=float, nargs="+", default=[4.0, 4.5, 5.0])
    p.add_argument("--Nx", type=int, default=2)
    p.add_argument("--Ny", type=int, default=2)
    p.add_argument("--R", type=float, default=8.0)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--emin", type=float, default=0.0)
    p.add_argument("--emax", type=float, default=1.2)
    p.add_argument("--nenergy", type=int, default=121)
    args = p.parse_args()

    energy = np.linspace(args.emin, args.emax, args.nenergy)
    values = np.empty((len(args.J), len(args.A), len(energy)))
    for ij, coupling in enumerate(args.J):
        for ia, period in enumerate(args.A):
            texture = make_array_texture(args.kind, period, args.Nx, args.Ny, args.R)
            values[ij, ia] = two_terminal_transmission(
                texture, energy, coupling, 1.0, eta=args.eta
            )
            print(json.dumps({
                "J": coupling, "A": period,
                "T_min": float(values[ij, ia].min()),
                "E_at_T_min": float(energy[np.argmin(values[ij, ia])]),
            }), flush=True)

    label = f"{args.kind}_Nx{args.Nx}_Ny{args.Ny}_R{args.R:g}"
    out = ROOT / "results" / "ef_a_j_map" / label
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "map.npz", energy=energy, A=np.asarray(args.A),
        J=np.asarray(args.J), Txx=values,
    )
    (out / "parameters.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"Saved {out / 'map.npz'}")


if __name__ == "__main__":
    main()
