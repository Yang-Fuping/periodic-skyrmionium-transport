import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.statistics import (
    anderson_disorder, ensemble_summary, finite_temperature_conductance,
)
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
    p.add_argument("--eta", type=float, default=1e-7)
    p.add_argument("--Wd", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--emin", type=float, default=-3.0)
    p.add_argument("--emax", type=float, default=3.0)
    p.add_argument("--nenergy", type=int, default=121)
    p.add_argument("--energy", type=float, nargs="+", default=None,
                   help="Explicit key energies; finite-temperature convolution is then skipped")
    p.add_argument("--kbt", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--output-label", default=None,
                   help="Optional subfolder below results/disorder_temperature")
    args = p.parse_args()
    texture = make_array_texture(args.kind, args.A, args.Nx, args.Ny, args.R)
    energy = (np.asarray(args.energy, dtype=float) if args.energy is not None
              else np.linspace(args.emin, args.emax, args.nenergy))
    rng = np.random.default_rng(args.seed)
    report = {"parameters": vars(args), "disorder": {}}
    arrays = {"energy": energy}
    for wd in args.Wd:
        sample = np.asarray([
            two_terminal_transmission(
                texture, energy, args.J, 1.0, eta=args.eta,
                onsite_disorder=anderson_disorder(texture.shape[:2], wd, rng),
            )
            for _ in range(args.samples)
        ])
        stats = ensemble_summary(sample)
        key = f"Wd_{wd:g}"
        arrays[f"{key}_samples"] = sample
        for stat_name, values in stats.items():
            arrays[f"{key}_{stat_name}"] = values
        report["disorder"][key] = {
            "mean_min": float(stats["mean"].min()),
            "mean_max": float(stats["mean"].max()),
            "median_min": float(stats["median"].min()),
            "median_max": float(stats["median"].max()),
            "q05_min": float(stats["q05"].min()),
            "q95_max": float(stats["q95"].max()),
        }
        if args.energy is None:
            for kbt in args.kbt:
                arrays[f"{key}_G_kBT_{kbt:g}"] = finite_temperature_conductance(
                    energy, stats["mean"], energy, kbt
                )
    out = ROOT / "results" / "disorder_temperature"
    if args.output_label:
        out = out / args.output_label
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "data.npz", **arrays)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
