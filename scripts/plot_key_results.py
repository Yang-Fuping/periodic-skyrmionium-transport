import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)

    # Full-zone gap convergence for the isolated band above the exchange gap.
    nks, gaps = [], []
    for nk in (7, 11, 21, 31):
        path = ROOT / "results" / "gap_scan" / "skyrmionium_q_zero" / f"R8_J5_nk{nk}" / "report.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        candidates = data["cells"][0]["largest_interior_candidates"]
        selected = next(item for item in candidates if item["n_occ"] == 325)
        nks.append(nk)
        gaps.append(selected["indirect_gap"])
    axes[0, 0].plot(nks, gaps, "o-", color="#1746d1")
    axes[0, 0].set(xlabel=r"Full-zone grid $n_k\times n_k$", ylabel=r"Indirect gap $\Delta/t$")
    axes[0, 0].set_xticks(nks)
    axes[0, 0].grid(alpha=0.25)

    # Finite-length transport.
    length_path = ROOT / "results" / "length_scaling" / "skyrmionium_q_zero_A18_Ny2.json"
    length = json.loads(length_path.read_text(encoding="utf-8"))
    nx = np.asarray(length["parameters"]["Nx"])
    values = np.asarray(length["transmissions"])
    for j, energy in enumerate(length["parameters"]["energy"]):
        axes[0, 1].semilogy(nx, values[:, j], "o-", label=rf"$E/t={energy:.4f}$")
    axes[0, 1].set(xlabel=r"Array length $N_x$", ylabel=r"Transmission $T_{xx}$")
    axes[0, 1].grid(alpha=0.25, which="both")
    axes[0, 1].legend(fontsize=8)

    # Disorder ensemble.
    disorder_path = ROOT / "results" / "disorder_temperature" / "data.npz"
    d = np.load(disorder_path)
    wd = np.asarray([0.0, 0.25, 0.5, 1.0])
    mean = np.asarray([d[f"Wd_{x:g}_mean"][0] for x in wd])
    sem = np.asarray([d[f"Wd_{x:g}_sem"][0] for x in wd])
    axes[1, 0].errorbar(wd, mean, yerr=sem, fmt="o-", capsize=3, color="#b33b2e")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel=r"Anderson disorder $W_d/t$", ylabel=r"$\langle T(E_{gap})\rangle$")
    axes[1, 0].grid(alpha=0.25, which="both")

    # Thermal convolution.
    temp_path = ROOT / "results" / "temperature" / "skyrmionium_q_zero_A18_J5_Nx4_Ny2" / "report.json"
    temp = json.loads(temp_path.read_text(encoding="utf-8"))
    kbts = np.asarray(temp["parameters"]["kbt"])
    conductance = np.asarray([
        temp["conductance_e2_over_h"][f"kBT_{x:g}"][0] for x in kbts
    ])
    axes[1, 1].semilogy(kbts, conductance, "o-", color="#198c45")
    axes[1, 1].set(xlabel=r"Temperature $k_BT/t$", ylabel=r"$G(E_F)/(e^2/h)$")
    axes[1, 1].grid(alpha=0.25, which="both")

    for label, ax in zip("abcd", axes.ravel()):
        ax.text(0.02, 0.96, f"({label})", transform=ax.transAxes, va="top", fontweight="bold")
    out = ROOT / "results" / "key_results_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
