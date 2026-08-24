"""Six-point length scaling for wider strips at the minigap centre."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import fit_exponential_length, paired_prefix_transmission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ny", type=int, required=True)
    parser.add_argument("--Nx", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8.0)
    parser.add_argument("--J", type=float, default=5.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--energy", type=float, default=1.09977149418366)
    parser.add_argument("--eta", type=float, default=1e-7)
    args = parser.parse_args()
    nx = np.asarray(sorted(set(args.Nx)), dtype=int)
    texture = make_array_texture(
        "skyrmionium_q_zero", args.A, int(nx[-1]), args.Ny, args.R
    )
    transmissions = paired_prefix_transmission(
        texture,
        tuple((args.A * nx).tolist()),
        np.asarray([args.energy]),
        args.J,
        args.t,
        eta=args.eta,
    )
    values = np.asarray([transmissions[int(args.A * item)][0] for item in nx])
    all_fit = fit_exponential_length(args.A * nx, values)
    tail_mask = nx >= 3
    tail_fit = fit_exponential_length(args.A * nx[tail_mask], values[tail_mask])
    report = {
        "parameters": vars(args),
        "Nx": nx.tolist(),
        "transmission": values.tolist(),
        "all_point_fit": all_fit,
        "asymptotic_fit_Nx_ge_3": tail_fit,
    }
    output = ROOT / "results" / "peer_review_complex_band"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"finite_length_Ny{args.Ny}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), **report}, indent=2))


if __name__ == "__main__":
    main()
