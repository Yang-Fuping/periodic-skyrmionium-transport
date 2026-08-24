"""Cellwise radius and position disorder at the baseline minigap centre."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.textures import make_cellwise_disordered_array_texture
from skyrmion_transport.transport import paired_prefix_transmission


def read_completed(path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed.add((row["disorder_kind"], int(row["sample"])))
    return completed


def append_row(path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--A", type=int, default=18)
    parser.add_argument("--R", type=float, default=8.0)
    parser.add_argument("--J", type=float, default=5.0)
    parser.add_argument("--Ny", type=int, default=2)
    parser.add_argument("--Nx", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--energy", type=float, default=1.09977149418366)
    parser.add_argument("--eta", type=float, default=1e-7)
    parser.add_argument("--radius-sigma", type=float, default=0.25)
    parser.add_argument("--position-sigma", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    nx = np.asarray(sorted(set(args.Nx)), dtype=int)
    output = ROOT / "results" / "texture_disorder"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw.jsonl"
    completed = read_completed(raw)

    for kind_index, disorder_kind in enumerate(("radius", "position")):
        for sample in range(args.samples):
            key = (disorder_kind, sample)
            if key in completed:
                continue
            rng = np.random.default_rng(args.seed + 100000 * kind_index + sample)
            radius_offsets = None
            center_offsets = None
            if disorder_kind == "radius":
                radius_offsets = np.clip(
                    rng.normal(0.0, args.radius_sigma, (int(nx[-1]), args.Ny)),
                    -2.0 * args.radius_sigma,
                    2.0 * args.radius_sigma,
                )
            else:
                center_offsets = np.clip(
                    rng.normal(
                        0.0,
                        args.position_sigma,
                        (int(nx[-1]), args.Ny, 2),
                    ),
                    -2.0 * args.position_sigma,
                    2.0 * args.position_sigma,
                )
            texture = make_cellwise_disordered_array_texture(
                "skyrmionium_q_zero",
                args.A,
                int(nx[-1]),
                args.Ny,
                args.R,
                radius_offsets=radius_offsets,
                center_offsets=center_offsets,
            )
            values = paired_prefix_transmission(
                texture,
                tuple((args.A * nx).tolist()),
                np.asarray([args.energy]),
                args.J,
                1.0,
                eta=args.eta,
            )
            transmission = [float(values[int(args.A * item)][0]) for item in nx]
            append_row(raw, {
                "disorder_kind": disorder_kind,
                "sample": sample,
                "seed": args.seed + 100000 * kind_index + sample,
                "Nx": nx.tolist(),
                "transmission": transmission,
            })
            if (sample + 1) % 10 == 0:
                print(
                    f"{disorder_kind}: {sample + 1}/{args.samples}",
                    flush=True,
                )

    metadata = {
        "parameters": vars(args),
        "distributions": {
            "radius": (
                "independent cell radius offsets drawn from a zero-mean normal "
                "distribution and clipped at two standard deviations"
            ),
            "position": (
                "independent x/y cell-centre offsets drawn from a zero-mean normal "
                "distribution and clipped at two standard deviations"
            ),
        },
        "raw": raw.name,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(raw), **metadata}, indent=2))


if __name__ == "__main__":
    main()
