import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import gaussian_dos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="skyrmionium_q_zero")
    p.add_argument("--A", type=int, default=18)
    p.add_argument("--nk", type=int, default=11)
    p.add_argument("--broadening", type=float, default=0.03)
    args = p.parse_args()
    folder = ROOT / "results" / "bloch" / f"{args.kind}_A{args.A}_nk{args.nk}"
    data = np.load(folder / "bloch_data.npz")
    bands = data["bands"]
    energy = np.linspace(bands.min(), bands.max(), 800)
    dos = gaussian_dos(data["grid_eigenvalues"], energy, args.broadening)
    has_flux = "berry_flux" in data.files
    fig, axes = plt.subplots(1, 3 if has_flux else 2, figsize=(11 if has_flux else 8, 4.5),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot(data["distance"], bands, color="black", lw=0.35)
    ax.set_xticks(data["ticks"], data["labels"])
    for tick in data["ticks"]:
        ax.axvline(tick, color="0.8", lw=0.6)
    ax.set(xlabel="magnetic Brillouin-zone path", ylabel=r"Energy $E/t$")
    axes[1].plot(dos, energy, color="#1746d1")
    axes[1].set(xlabel="DOS (arb. units)", ylabel=r"Energy $E/t$")
    if has_flux:
        im = axes[2].imshow(data["berry_flux"].T, origin="lower", cmap="RdBu_r")
        axes[2].set_title("FHS Berry flux")
        fig.colorbar(im, ax=axes[2], shrink=0.8)
    fig.savefig(folder / "bloch_bands_dos_berry.png", dpi=220)
    plt.close(fig)
    print(f"Saved {folder / 'bloch_bands_dos_berry.png'}")


if __name__ == "__main__":
    main()
