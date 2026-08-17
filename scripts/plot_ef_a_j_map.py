import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Folder name below results/ef_a_j_map")
    args = p.parse_args()
    folder = ROOT / "results" / "ef_a_j_map" / args.folder
    data = np.load(folder / "map.npz")
    energy, periods, couplings = data["energy"], data["A"], data["J"]
    transmission = data["Txx"]

    fig, axes = plt.subplots(
        1, len(couplings), figsize=(4.4 * len(couplings), 4.0),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for ax, coupling, values in zip(axes, couplings, transmission):
        image = ax.imshow(
            np.log10(np.maximum(values, 1e-12)), origin="lower", aspect="auto",
            extent=[energy[0], energy[-1], periods[0], periods[-1]],
            vmin=-8, vmax=1, cmap="magma",
        )
        ax.set_title(fr"$J/t={coupling:g}$")
        ax.set_xlabel(r"$E_F/t$")
        ax.set_yticks(periods)
        lead_edge = coupling - 4.0
        if energy[0] <= lead_edge <= energy[-1]:
            ax.axvline(lead_edge, color="cyan", ls="--", lw=1.0,
                       label=r"clean-lead edge $J-4t$")
            ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel(r"array period $A/a$")
    cbar = fig.colorbar(image, ax=axes, shrink=0.9)
    cbar.set_label(r"$\log_{10} T_{xx}$")
    path = folder / "ef_a_j_transport_map.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
