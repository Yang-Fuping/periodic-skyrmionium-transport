import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.multiterminal import (
    four_terminal_observables, standard_four_contacts, transmission_matrix_sparse,
)
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import two_terminal_transmission


def main():
    p = argparse.ArgumentParser(description="Compute the (EF,A,Nx) transport map")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, nargs="+", default=[18, 20, 24, 30])
    p.add_argument("--Nx", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--Ny", type=int, default=1)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--emin", type=float, default=-3.0)
    p.add_argument("--emax", type=float, default=3.0)
    p.add_argument("--nenergy", type=int, default=121)
    p.add_argument("--hall", action="store_true",
                   help="Also use dense four-terminal NEGF; intended only for small maps")
    p.add_argument("--probe-width", type=int, default=4)
    args = p.parse_args()
    energy = np.linspace(args.emin, args.emax, args.nenergy)
    Txx = np.empty((len(args.A), len(args.Nx), len(energy)))
    Rxy = np.full_like(Txx, np.nan)
    for ia, A in enumerate(args.A):
        for ix, nx in enumerate(args.Nx):
            texture = make_array_texture(args.kind, A, nx, args.Ny, args.R)
            Txx[ia, ix] = two_terminal_transmission(texture, energy, args.J, 1.0, eta=args.eta)
            if args.hall:
                contacts = standard_four_contacts(texture.shape[0], texture.shape[1], args.probe_width)
                for ie, E in enumerate(energy):
                    T, _ = transmission_matrix_sparse(texture, E, args.J, 1.0, contacts, eta=args.eta)
                    Rxy[ia, ix, ie] = four_terminal_observables(T)["Rxy_h_over_e2"]
    out = ROOT / "results" / "phase_map" / args.kind
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "map.npz", energy=energy, A=args.A, Nx=args.Nx, Txx=Txx, Rxy=Rxy)
    (out / "parameters.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Saved {out / 'map.npz'}")


if __name__ == "__main__":
    main()
