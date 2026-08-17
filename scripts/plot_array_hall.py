import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Folder name below results/array_hall")
    p.add_argument("--gap-min", type=float, default=None)
    p.add_argument("--gap-max", type=float, default=None)
    args = p.parse_args()
    folder = ROOT / "results" / "array_hall" / args.folder
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    styles = {
        "skyrmionium_q_zero": ("Q=0 Skyrmionium", "o-"),
        "skyrmion_q_plus": ("Q=+1 Skyrmion", "s-"),
        "skyrmion_q_minus": ("Q=-1 Skyrmion", "^-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    for kind, rows in data["data"].items():
        label, style = styles.get(kind, (kind, "o-"))
        E = [row["energy"] for row in rows]
        rxy = [row["Rxy_h_over_e2"] for row in rows]
        angle = [row["hall_angle"] for row in rows]
        axes[0].plot(E, rxy, style, label=label)
        axes[1].plot(E, angle, style, label=label)
    axes[0].set(xlabel=r"Energy $E/t$", ylabel=r"$R_{xy}$ ($h/e^2$)")
    axes[1].set(xlabel=r"Energy $E/t$", ylabel="charge Hall angle")
    for ax in axes:
        ax.axhline(0, color="0.5", lw=0.7)
        if args.gap_min is not None and args.gap_max is not None:
            ax.axvspan(args.gap_min, args.gap_max, color="0.45", alpha=0.12,
                       label="Q=0 Bloch minigap")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path = folder / "hall_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
