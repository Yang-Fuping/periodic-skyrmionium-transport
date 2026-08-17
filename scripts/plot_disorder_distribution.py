import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Folder name below results/disorder_temperature")
    args = p.parse_args()
    folder = ROOT / "results" / "disorder_temperature" / args.folder
    data = np.load(folder / "data.npz")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))

    strengths = [float(x) for x in report["parameters"]["Wd"]]
    samples = [np.ravel(data[f"Wd_{x:g}_samples"]) for x in strengths]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    parts = ax.violinplot(samples, positions=np.arange(len(strengths)), showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("#4C78A8")
        body.set_alpha(0.55)
    ax.set_xticks(np.arange(len(strengths)), [f"{x:g}" for x in strengths])
    ax.set_yscale("log")
    ax.set_xlabel(r"Anderson disorder $W_d/t$")
    ax.set_ylabel(r"$T_{xx}(E_g)$")
    ax.grid(axis="y", alpha=0.25)
    path = folder / "disorder_distribution.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
