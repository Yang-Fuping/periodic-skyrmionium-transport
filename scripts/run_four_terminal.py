import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.multiterminal import (
    four_terminal_observables, scattering_matrix_from_green,
    spin_current_proxy, spin_resolved_transmission, standard_four_contacts,
    terminal_currents_from_scattering, transmission_matrix,
    transmission_matrix_sparse,
)
from skyrmion_transport.textures import make_texture


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--L", type=int, default=24)
    p.add_argument("--W", type=int, default=18)
    p.add_argument("--R", type=float, default=6)
    p.add_argument("--probe-width", type=int, default=4)
    p.add_argument("--probe-J", type=float, default=None,
                   help="Side-probe exchange; use 0 for normal-metal Hall probes")
    p.add_argument("--energy", type=float, nargs="+", default=[0.0])
    p.add_argument("--J", type=float, default=1.5)
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--solver", choices=["dense", "sparse"], default="sparse")
    args = p.parse_args()
    texture = make_texture(args.kind, args.L, args.W, args.R)
    contacts = standard_four_contacts(args.L, args.W, args.probe_width, probe_J=args.probe_J)
    rows = []
    for energy in args.energy:
        solver = transmission_matrix_sparse if args.solver == "sparse" else transmission_matrix
        T, diag = solver(texture, energy, args.J, 1.0, contacts, eta=args.eta)
        obs = four_terminal_observables(T)
        # This proxy omits spin-flip reflection.  It is exploratory and must not
        # be presented as the paper's final conserved spin-current observable.
        Ts = spin_resolved_transmission(diag)
        spin_current_exploratory = spin_current_proxy(Ts, obs["voltages"])
        Sblocks, spin_labels, unitary_error = scattering_matrix_from_green(diag)
        scattering_charge, scattering_spin = terminal_currents_from_scattering(
            Sblocks, spin_labels, obs["voltages"]
        )
        rows.append({"energy": energy, "T": T.tolist(),
                     "spin_current_proxy": spin_current_exploratory.tolist(),
                     "scattering_charge_current": scattering_charge.tolist(),
                     "scattering_spin_current_e_over_4pi": scattering_spin.tolist(),
                     "scattering_unitarity_error": unitary_error,
                     **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in obs.items()}})
    out = ROOT / "results" / "four_terminal"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.kind}_pw{args.probe_width}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
