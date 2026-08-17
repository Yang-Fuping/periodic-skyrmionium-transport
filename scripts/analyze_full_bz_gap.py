"""Extract band-edge maps and extrema for one full-zone minigap."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import direct_indirect_gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="skyrmionium_q_zero")
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8)
    parser.add_argument("--J", type=float, default=5)
    parser.add_argument("--n-occ", type=int, default=325)
    parser.add_argument("--nk", type=int, nargs="+", default=[11, 21, 31])
    args = parser.parse_args()

    tag = (f"{args.kind}_A{args.A}_R{args.R:g}_J{args.J:g}"
           f"_n{args.n_occ}")
    output = ROOT / "results" / "full_bz_gap" / tag
    output.mkdir(parents=True, exist_ok=True)
    report = {"parameters": vars(args), "convergence": []}

    for nk in args.nk:
        source = (ROOT / "results" / "gap_scan" / args.kind /
                  f"R{args.R:g}_J{args.J:g}_nk{nk}" / f"A{args.A}.npz")
        if not source.exists():
            raise FileNotFoundError(f"Missing full-zone archive: {source}")
        with np.load(source) as archive:
            eigenvalues = archive["eigenvalues"]
        if eigenvalues.shape[:2] != (nk, nk):
            raise ValueError(f"Unexpected k-grid shape in {source}")

        valence = eigenvalues[..., args.n_occ - 1]
        conduction = eigenvalues[..., args.n_occ]
        direct = conduction - valence
        ks = np.linspace(-np.pi / args.A, np.pi / args.A, nk,
                         endpoint=False)
        iv = np.unravel_index(np.argmax(valence), valence.shape)
        ic = np.unravel_index(np.argmin(conduction), conduction.shape)
        gap = direct_indirect_gap(eigenvalues, args.n_occ)
        entry = {
            "nk": nk,
            **gap,
            "valence_max_index": [int(iv[0]), int(iv[1])],
            "valence_max_k": [float(ks[iv[0]]), float(ks[iv[1]])],
            "conduction_min_index": [int(ic[0]), int(ic[1])],
            "conduction_min_k": [float(ks[ic[0]]), float(ks[ic[1]])],
            "source": str(source.relative_to(ROOT.parent)),
        }
        np.savez_compressed(
            output / f"nk{nk}.npz",
            k_values=ks,
            valence=valence,
            conduction=conduction,
            direct_gap_map=direct,
        )
        report["convergence"].append(entry)
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(entry, indent=2), flush=True)


if __name__ == "__main__":
    main()
