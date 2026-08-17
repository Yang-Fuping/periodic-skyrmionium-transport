import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import (
    clean_lead_modes,
    fit_exponential_length,
    two_terminal_transmission,
)


def main():
    p = argparse.ArgumentParser(description="Length scaling at candidate Bloch-gap energies")
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--Ny", type=int, default=1)
    p.add_argument("--energy", type=float, nargs="+", required=True)
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--output-label", default=None,
                   help="Independent output stem; the historical stem is used if omitted")
    args = p.parse_args()
    transmissions = np.zeros((len(args.Nx), len(args.energy)))
    for ix, nx in enumerate(args.Nx):
        texture = make_array_texture(args.kind, args.A, nx, args.Ny, args.R)
        transmissions[ix] = two_terminal_transmission(
            texture, np.asarray(args.energy), args.J, 1.0, eta=args.eta
        )
    fits = {str(E): fit_exponential_length(np.asarray(args.Nx) * args.A, transmissions[:, j])
            for j, E in enumerate(args.energy)}
    channel_count = clean_lead_modes(np.asarray(args.energy), args.A * args.Ny,
                                     args.J, 1.0)
    channel_bounds_passed = bool(np.all(transmissions >= -1e-12)
                                 and np.all(transmissions <= channel_count[None, :] + 1e-9))
    report = {"parameters": vars(args), "fits": fits,
              "transmissions": transmissions.tolist(),
              "clean_lead_channel_count": channel_count.tolist(),
              "channel_bounds_passed": channel_bounds_passed}
    out = ROOT / "results" / "length_scaling"
    out.mkdir(parents=True, exist_ok=True)
    stem = args.output_label or f"{args.kind}_A{args.A}_Ny{args.Ny}"
    path = out / f"{stem}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
