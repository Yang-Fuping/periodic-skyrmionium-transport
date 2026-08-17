import argparse

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import full_green_observables


def main():
    p = argparse.ArgumentParser(description="Small-device LDOS and local-current calculation")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, default=1)
    p.add_argument("--Ny", type=int, default=1)
    p.add_argument("--energy", type=float, required=True)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-7)
    args = p.parse_args()
    texture = make_array_texture(args.kind, args.A, args.Nx, args.Ny, args.R)
    if 2 * texture.shape[0] * texture.shape[1] > 2400:
        raise RuntimeError("Full local Green function is limited to 2400 orbitals")
    obs = full_green_observables(texture, args.energy, args.J, 1.0, eta=args.eta)
    out = ROOT / "results" / "local_observables" / args.kind
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"E{args.energy:+.6f}.npz", texture=texture, **obs)
    print(f"T={obs['transmission']:.10g}; saved {out}")


if __name__ == "__main__":
    main()
