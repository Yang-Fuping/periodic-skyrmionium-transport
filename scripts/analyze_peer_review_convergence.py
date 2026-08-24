"""Assemble peer-review length and transverse-width convergence evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from paper_plot_style import LEGEND_SIZE, configure_paper_style


INPUT = ROOT / "results" / "length_scaling"
OUTPUT = ROOT / "results" / "peer_review_convergence"


def load(name: str) -> dict:
    return json.loads((INPUT / name).read_text(encoding="utf-8"))


def effective_xi(A: int, nx: np.ndarray, transmission: np.ndarray) -> float:
    slope = np.polyfit(A * nx, np.log(transmission), 1)[0]
    return float(-1.0 / slope)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    length = load("peer_review_A18_R8_J5_Ny2_Nx1_2_3_4_6_8.json")
    width_sets = {
        2: length,
        4: load("peer_review_A18_R8_J5_Ny4_Nx2_4.json"),
        6: load("peer_review_A18_R8_J5_Ny6_Nx2_4.json"),
        8: load("peer_review_A18_R8_J5_Ny8_Nx2_4.json"),
    }

    A = int(length["parameters"]["A"])
    nx = np.asarray(length["parameters"]["Nx"], dtype=float)
    transmission = np.asarray(length["transmissions"], dtype=float)[:, 0]
    energy_key = next(iter(length["fits"]))
    fit = length["fits"][energy_key]

    width_rows = []
    for ny, data in width_sets.items():
        data_nx = np.asarray(data["parameters"]["Nx"], dtype=float)
        data_t = np.asarray(data["transmissions"], dtype=float)[:, 0]
        mask = np.isin(data_nx, [2, 4])
        nx24 = data_nx[mask]
        t24 = data_t[mask]
        order = np.argsort(nx24)
        nx24, t24 = nx24[order], t24[order]
        if nx24.tolist() != [2.0, 4.0]:
            raise ValueError(f"Ny={ny} does not contain Nx=2,4")
        width_rows.append({
            "Ny": ny,
            "T_Nx2": float(t24[0]),
            "T_Nx4": float(t24[1]),
            "T_per_Ny_Nx2": float(t24[0] / ny),
            "T_per_Ny_Nx4": float(t24[1] / ny),
            "xi_from_Nx2_4": effective_xi(A, nx24, t24),
            "channel_bounds_passed": bool(data["channel_bounds_passed"]),
        })

    rows_by_ny = {row["Ny"]: row for row in width_rows}
    ny6, ny8 = rows_by_ny[6], rows_by_ny[8]
    tail_changes = {
        "T_per_Ny_Nx2_Ny6_to_8": abs(ny8["T_per_Ny_Nx2"] / ny6["T_per_Ny_Nx2"] - 1.0),
        "T_per_Ny_Nx4_Ny6_to_8": abs(ny8["T_per_Ny_Nx4"] / ny6["T_per_Ny_Nx4"] - 1.0),
        "xi_Ny6_to_8": abs(ny8["xi_from_Nx2_4"] / ny6["xi_from_Nx2_4"] - 1.0),
    }

    assessment = {
        "parameters": {
            "A": A,
            "R": length["parameters"]["R"],
            "J": length["parameters"]["J"],
            "energy": length["parameters"]["energy"][0],
            "eta": length["parameters"]["eta"],
        },
        "extended_length_scaling": {
            "Nx": [int(value) for value in nx],
            "transmission": transmission.tolist(),
            "decay_length": fit["decay_length"],
            "r2": fit["r2"],
            "channel_bounds_passed": length["channel_bounds_passed"],
        },
        "transverse_width_rows": width_rows,
        "relative_tail_changes": tail_changes,
        "interpretation": (
            "The total transmission grows with transverse width, as required for a wider strip. "
            "The transmission per magnetic-cell row and the Nx=2-to-4 attenuation length "
            "stabilize between Ny=6 and Ny=8. The minigap attenuation is therefore not a "
            "Ny=2 artefact; the evidence does not imply convergence of the unnormalised total transmission."
        ),
    }
    (OUTPUT / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )

    x = A * nx
    fitted = np.exp(float(fit["intercept"]) + float(fit["slope"]) * x)
    nys = np.asarray([row["Ny"] for row in width_rows], dtype=float)
    xi = np.asarray([row["xi_from_Nx2_4"] for row in width_rows])
    t2n = np.asarray([row["T_per_Ny_Nx2"] for row in width_rows])
    t4n = np.asarray([row["T_per_Ny_Nx4"] for row in width_rows])

    configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(nx, transmission, "o", label="calculation")
    ax.semilogy(nx, fitted, "-", label=rf"fit: $\xi={fit['decay_length']:.2f}a$")
    ax.set_xlabel(r"array length $N_x$")
    ax.set_ylabel(r"transmission $T(E_{\rm mid})$")
    ax.set_title("(a) extended length scaling")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)

    ax = axes[1]
    ax.semilogy(nys, t2n, "o-", label=r"$T/N_y$, $N_x=2$")
    ax.semilogy(nys, t4n, "s-", label=r"$T/N_y$, $N_x=4$")
    ax.set_xlabel(r"transverse cell count $N_y$")
    ax.set_ylabel(r"row-normalized transmission $T/N_y$")
    ax.set_title("(b) transverse-width check")
    ax2 = ax.twinx()
    ax2.plot(nys, xi, "^--", color="tab:green", label=r"$\xi_{2\to4}$")
    ax2.set_ylabel(r"effective decay length $\xi_{2\to4}/a$", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False,
              fontsize=LEGEND_SIZE, loc="best")

    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"length_width_convergence.{suffix}", dpi=300)
    plt.close(fig)

    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
