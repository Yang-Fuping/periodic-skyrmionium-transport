import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--Ny", type=int, default=2)
    p.add_argument("--output-label", default=None,
                   help="Input/output stem produced by run_length_scaling.py")
    args = p.parse_args()
    folder = ROOT / "results" / "length_scaling"
    stem = args.output_label or f"{args.kind}_A{args.A}_Ny{args.Ny}"
    source = folder / f"{stem}.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    nx = np.asarray(data["parameters"]["Nx"])
    energies = data["parameters"]["energy"]
    values = np.asarray(data["transmissions"])
    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    for j, energy in enumerate(energies):
        fit = data["fits"][str(energy)]
        label = rf"$E/t={energy:.4f}$, $R^2={fit['r2']:.4f}$"
        ax.semilogy(nx, values[:, j], "o-", lw=1.8, label=label)
    ax.set(xlabel=r"Array length $N_x$", ylabel=r"Transmission $T_{xx}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    path = folder / f"{stem}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
