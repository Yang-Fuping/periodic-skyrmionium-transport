from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    source = ROOT / "results" / "stage0"
    if not source.exists():
        raise SystemExit("Run scripts/run_stage0.py first")
    kinds = ["skyrmion_q_plus", "skyrmionium_q_zero", "skyrmion_q_minus"]
    labels = [r"Skyrmion $Q=+1$", r"Skyrmionium $Q=0$", r"Skyrmion $Q=-1$"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), constrained_layout=True)
    for col, (kind, label) in enumerate(zip(kinds, labels)):
        data = np.load(source / f"{kind}.npz")
        m = data["m"]
        rho = data["q_density"]
        q = float(data["Q"])
        im0 = axes[0, col].imshow(m[..., 2].T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
        step = 3
        x, y = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        axes[0, col].quiver(x[::step, ::step], y[::step, ::step],
                            m[::step, ::step, 0], m[::step, ::step, 1],
                            color="k", scale=18, width=0.004)
        axes[0, col].set_title(f"{label}\ncomputed Q={q:+.3f}")
        im1 = axes[1, col].imshow(rho.T, origin="lower", cmap="RdBu_r",
                                  vmin=-max(abs(rho.min()), abs(rho.max())),
                                  vmax=max(abs(rho.min()), abs(rho.max())))
        axes[1, col].set_title("solid-angle charge per plaquette")
        for row in (0, 1):
            axes[row, col].set_xlabel("x/a")
            axes[row, col].set_ylabel("y/a")
    fig.colorbar(im0, ax=axes[0, :], label=r"$m_z$", shrink=0.8)
    fig.colorbar(im1, ax=axes[1, :], label=r"$q_{\mathrm{plaquette}}$", shrink=0.8)
    fig.savefig(source / "figure1_textures_topology.png", dpi=220)
    plt.close(fig)

    d = np.load(source / "single_skyrmionium_transport.npz")
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.step(d["energy"], d["N"], where="mid", color="0.35", ls="--", label="clean modes N(E)")
    ax.plot(d["energy"], d["T"], color="#1746d1", lw=2, label="single skyrmionium")
    ax.set(xlabel=r"Energy $E/t$", ylabel=r"Transmission $T(E)$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(source / "single_skyrmionium_baseline.png", dpi=220)
    plt.close(fig)
    print(f"Saved figures in {source}")


if __name__ == "__main__":
    main()
