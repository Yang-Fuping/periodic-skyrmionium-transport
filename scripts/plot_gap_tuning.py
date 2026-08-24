import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def candidate(kind, A, R, J, nk, n_occ):
    path = ROOT / "results" / "gap_scan" / kind / f"R{R:g}_J{J:g}_nk{nk}" / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    cell = next(item for item in report["cells"] if item["A"] == A)
    return next(item for item in cell["largest_interior_candidates"] if item["n_occ"] == n_occ)


def main():
    J = np.asarray([4.0, 4.5, 5.0])
    jdata = [candidate("skyrmionium_q_zero", 18, 8, x, 11 if x < 5 else 31, 325) for x in J]
    A = np.asarray([18, 20, 24])
    adata = [candidate("skyrmionium_q_zero", int(x), 8, 5, 31 if x == 18 else 11,
                       int(x * x + 1)) for x in A]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].plot(J, [x["midgap_energy"] for x in jdata], "o-", label="gap center")
    axes[0].fill_between(
        J, [x["valence_max"] for x in jdata], [x["conduction_min"] for x in jdata],
        alpha=0.25, label="minigap",
    )
    axes[0].set(xlabel=r"exchange coupling $J/t$", ylabel=r"Energy $E/t$")
    axes[0].legend()
    axes[1].plot(A, [x["indirect_gap"] for x in adata], "o-", color="#b33b2e")
    axes[1].set(xlabel=r"Array period $A/a$", ylabel=r"Indirect minigap $\Delta/t$")
    axes[1].set_xticks(A)
    for label, ax in zip("ab", axes):
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.95, f"({label})", transform=ax.transAxes, va="top", fontweight="bold")
    out = ROOT / "results" / "gap_tuning.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
