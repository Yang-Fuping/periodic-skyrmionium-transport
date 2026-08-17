import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import (
    make_array_texture, make_skyrmionium_wall_array,
)


def main():
    p = argparse.ArgumentParser(description="Strict Q=0,+1,-1 finite-array Hall comparison")
    p.add_argument("--kind", nargs="+", default=[
        "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"
    ])
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--R", type=float, default=8)
    p.add_argument("--Nx", type=int, default=1)
    p.add_argument("--Ny", type=int, default=2)
    p.add_argument("--J", type=float, default=5)
    p.add_argument("--probe-J", type=float, default=0.0)
    p.add_argument("--probe-width", type=int, default=4)
    p.add_argument("--probe-start", type=int, default=None)
    p.add_argument("--longitudinal-J", type=float, default=None,
                   help="Signed L/R lead exchange; default matches device J")
    p.add_argument("--padding", type=int, nargs=2, default=[0, 0],
                   metavar=("PX", "PY"))
    p.add_argument("--offset", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("DX", "DY"))
    p.add_argument("--ellipticity", type=float, nargs=2, default=[1.0, 1.0],
                   metavar=("EX", "EY"))
    p.add_argument("--energy", type=float, nargs="+", default=None)
    p.add_argument("--emin", type=float, default=None)
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--nenergy", type=int, default=71)
    p.add_argument("--eta", type=float, default=1e-8)
    p.add_argument("--output-label", default=None,
                   help="Optional explicit folder name below results/array_hall")
    args = p.parse_args()
    if args.energy is None:
        if args.emin is None or args.emax is None:
            p.error("use --energy or provide both --emin and --emax")
        energies = np.linspace(args.emin, args.emax, args.nenergy)
    else:
        energies = np.asarray(args.energy, dtype=float)
    start_label = "center" if args.probe_start is None else str(args.probe_start)
    geometry_label = (f"dx{args.offset[0]:g}_dy{args.offset[1]:g}_"
                      f"ex{args.ellipticity[0]:g}_ey{args.ellipticity[1]:g}")
    folder_label = (args.output_label or
                    f"A{args.A}_R{args.R:g}_J{args.J:g}_Nx{args.Nx}_Ny{args.Ny}_pJ{args.probe_J:g}_pw{args.probe_width}_ps{start_label}_{geometry_label}")
    out = ROOT / "results" / "array_hall" / folder_label
    out.mkdir(parents=True, exist_ok=True)
    report = {"parameters": vars(args), "data": {}}
    for kind in args.kind:
        if kind in {"skyrmionium_inner_wall", "skyrmionium_outer_wall"}:
            component = "inner" if "inner" in kind else "outer"
            texture = make_skyrmionium_wall_array(
                component, args.A, args.Nx, args.Ny, args.R,
                padding=tuple(args.padding),
            )
            matched_longitudinal_J = -abs(args.J) if component == "inner" else abs(args.J)
        else:
            texture = make_array_texture(
                kind, args.A, args.Nx, args.Ny, args.R,
                padding=tuple(args.padding), offset=tuple(args.offset),
                ellipticity=tuple(args.ellipticity),
            )
            matched_longitudinal_J = args.J
        longitudinal_J = (matched_longitudinal_J if args.longitudinal_J is None
                          else args.longitudinal_J)
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], args.probe_width,
            probe_J=args.probe_J, probe_start=args.probe_start,
            longitudinal_J=longitudinal_J,
        )
        rows = []
        report["data"][kind] = rows
        for energy in energies:
            rows.append(evaluate_hall_point(
                texture, energy, args.J, 1.0, contacts, eta=args.eta
            ))
            print(json.dumps({"kind": kind, **rows[-1]}), flush=True)
            (out / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
