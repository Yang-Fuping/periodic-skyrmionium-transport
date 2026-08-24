"""Quantify the reciprocal-space m_z form factor of texture controls.

This analysis deliberately separates a control-based inference from a
single-harmonic microscopic proof.  It reports every displayed coefficient so
that the manuscript can state exactly what the data do and do not establish.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from paper_plot_style import LEGEND_SIZE, configure_paper_style
from skyrmion_transport.textures import make_texture


KINDS = (
    "uniform",
    "skyrmion_q_plus",
    "skyrmionium_q_zero",
    "skyrmionium_q_zero_quintic",
    "nonwinding_double_wall_q_zero",
)
LABELS = {
    "uniform": "FM",
    "skyrmion_q_plus": r"$Q=+1$",
    "skyrmionium_q_zero": "skyrmionium",
    "skyrmionium_q_zero_quintic": "quintic",
    "nonwinding_double_wall_q_zero": "nonwinding",
}
COLORS = {
    "uniform": "0.45",
    "skyrmion_q_plus": "#de2d26",
    "skyrmionium_q_zero": "#756bb1",
    "skyrmionium_q_zero_quintic": "#fd8d3c",
    "nonwinding_double_wall_q_zero": "#31a354",
}
SELECTED = ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0))


def coefficient(field: np.ndarray, h: int, k: int) -> complex:
    A = field.shape[0]
    transformed = np.fft.fft2(field) / (A * A)
    return complex(transformed[h % A, k % A])


def main() -> None:
    A, R = 18, 8.0
    source = (ROOT / "results" / "texture_gap_controls" /
              "A18_R8_J5_n325_nk21" / "report.json")
    output = ROOT / "results" / "peer_review_mz_form_factor_v2"
    output.mkdir(parents=True, exist_ok=True)
    gap_report = json.loads(source.read_text(encoding="utf-8"))
    cells = {kind: make_texture(kind, A, A, R) for kind in KINDS}
    gaps = {
        kind: float(gap_report["textures"][kind]["locally_refined"]["indirect_gap"])
        for kind in KINDS
    }
    coefficients = {
        kind: {
            f"({h},{k})": float(abs(coefficient(cell[..., 2], h, k)))
            for h, k in SELECTED
        }
        for kind, cell in cells.items()
    }
    same_mz_error = float(np.max(np.abs(
        cells["skyrmionium_q_zero"][..., 2]
        - cells["nonwinding_double_wall_q_zero"][..., 2]
    )))
    gap_ratio = gaps["skyrmionium_q_zero"] / gaps["skyrmion_q_plus"]
    first_shell_ratio = (
        coefficients["skyrmionium_q_zero"]["(1,0)"]
        / coefficients["skyrmion_q_plus"]["(1,0)"]
    )
    high_harmonic_ratio = (
        coefficients["skyrmionium_q_zero"]["(2,2)"]
        / coefficients["skyrmion_q_plus"]["(2,2)"]
    )
    assessment = {
        "parameters": {"A": A, "R": R, "J": 5.0, "n_occ": 325},
        "fourier_convention": (
            "m_z(h,k)=A^{-2} sum_{x,y} m_z(x,y) "
            "exp[-2 pi i (h x+k y)/A]"
        ),
        "nonwinding_definition": {
            "theta": "pi[1-cos(pi r/R)] for r<=R and 0 otherwise",
            "azimuth": "constant helicity chi_0 (zero in all reported data)",
            "magnetization": "(sin(theta) cos(chi_0), sin(theta) sin(chi_0), cos(theta))",
            "sin_theta_zero_convention": (
                "m_x=m_y=0 directly; no azimuthal branch is required"
            ),
        },
        "same_mz_max_site_error": same_mz_error,
        "indirect_gaps": gaps,
        "selected_abs_mz_coefficients": coefficients,
        "ratios": {
            "gap_skyrmionium_over_q_plus": gap_ratio,
            "abs_mz_10_skyrmionium_over_q_plus": first_shell_ratio,
            "abs_mz_22_skyrmionium_over_q_plus": high_harmonic_ratio,
        },
        "inference": (
            "No single first-shell m_z coefficient explains all control gaps: "
            "the skyrmionium (1,0) amplitude is smaller than the Q=+1 value "
            "despite its much larger gap.  The double wall redistributes weight "
            "to higher harmonics, including a (2,2) amplitude ratio close to the "
            "gap ratio, but that correlation is not a matrix-element proof.  The "
            "identical-m_z nonwinding pair therefore supports a dominant "
            "multi-harmonic polar-form-factor contribution within the tested "
            "controls, while in-plane winding remains important for band-edge "
            "location and directness."
        ),
    }
    (output / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )

    configure_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    limit = 4
    images = []
    for ax, kind, label in zip(
        axes[:2],
        ("skyrmion_q_plus", "skyrmionium_q_zero"),
        (r"(a) $Q=+1$ skyrmion", "(b) skyrmionium"),
    ):
        mz_fft = np.fft.fftshift(np.fft.fft2(cells[kind][..., 2]) / (A * A))
        center = A // 2
        image = ax.imshow(
            np.abs(mz_fft[center-limit:center+limit+1,
                          center-limit:center+limit+1]).T,
            origin="lower", cmap="magma", extent=(-limit-0.5, limit+0.5,
                                                   -limit-0.5, limit+0.5),
            vmin=0.0, vmax=0.24, interpolation="nearest",
        )
        images.append(image)
        ax.set(xlabel=r"$h$", ylabel=r"$k$", title=label,
               xticks=(-4, -2, 0, 2, 4), yticks=(-4, -2, 0, 2, 4))
    fig.colorbar(images[-1], ax=axes[:2], label=r"$|m_z(h,k)|$",
                 shrink=0.84, pad=0.02)

    ax = axes[2]
    x = np.arange(len(SELECTED))
    for kind in ("skyrmion_q_plus", "skyrmionium_q_zero",
                 "skyrmionium_q_zero_quintic",
                 "nonwinding_double_wall_q_zero"):
        values = [coefficients[kind][f"({h},{k})"] for h, k in SELECTED]
        ax.plot(x, values, "o-", ms=3.2, lw=1.0, color=COLORS[kind],
                label=LABELS[kind])
    ax.set_xticks(x, [rf"$({h},{k})$" for h, k in SELECTED], rotation=35,
                  ha="right")
    ax.set_ylabel(r"$|m_z(h,k)|$")
    ax.set_title("(c) selected form factors")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=LEGEND_SIZE, handlelength=1.2, labelspacing=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"mz_form_factor_v2.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
