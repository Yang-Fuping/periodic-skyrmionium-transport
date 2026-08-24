"""Plot the same-filling magnetic-texture controls for peer review."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from paper_plot_style import ANNOTATION_SIZE, configure_paper_style
from skyrmion_transport.textures import make_texture


def outside_panel_label(ax, label: str) -> None:
    ax.set_title(f"{label} {ax.get_title()}")


def texture_panel(ax, cell: np.ndarray, title: str) -> None:
    image = ax.imshow(
        cell[..., 2].T,
        origin="lower",
        cmap="coolwarm_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    x, y = np.meshgrid(np.arange(cell.shape[0]), np.arange(cell.shape[1]),
                       indexing="ij")
    step = 2
    ax.quiver(
        x[::step, ::step], y[::step, ::step],
        cell[::step, ::step, 0], cell[::step, ::step, 1],
        color="0.1", pivot="mid", scale=16, width=0.005,
        headwidth=3.0, headlength=3.8,
    )
    ax.set_title(title, pad=6)
    ax.set_xlabel(r"$x/a$")
    ax.set_ylabel(r"$y/a$")
    ax.set_aspect("equal")
    return image


def main() -> None:
    source = (ROOT / "results" / "texture_gap_controls" /
              "A18_R8_J5_n325_nk21" / "report.json")
    output = ROOT / "results" / "peer_review_texture_controls"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(source.read_text(encoding="utf-8"))

    sky = make_texture("skyrmionium_q_zero", 18, 18, 8)
    wall = make_texture("nonwinding_double_wall_q_zero", 18, 18, 8)
    mz_error = float(np.max(np.abs(sky[..., 2] - wall[..., 2])))
    gap_sky = report["textures"]["skyrmionium_q_zero"]["locally_refined"]["indirect_gap"]
    gap_quintic = report["textures"]["skyrmionium_q_zero_quintic"]["locally_refined"]["indirect_gap"]
    gap_wall = report["textures"]["nonwinding_double_wall_q_zero"]["locally_refined"]["indirect_gap"]
    relative_gap_change = abs(gap_wall / gap_sky - 1.0)
    quintic_relative_gap_change = abs(gap_quintic / gap_sky - 1.0)
    report["comparisons"]["same_mz_max_site_error"] = mz_error
    report["comparisons"]["same_mz_relative_gap_change"] = relative_gap_change
    report["comparisons"]["quintic_profile_relative_gap_change"] = quintic_relative_gap_change
    report["comparisons"]["mechanism_assessment"] = (
        "The same-mz nonwinding double wall retains the gap width within 0.5%, "
        "but changes it from direct at M to indirect between M and X. Together with "
        "the Fourier analysis, this strongly indicates that the multi-harmonic "
        "double-wall polar/mz form factor supplies the dominant contribution within "
        "the tested controls, whereas in-plane winding reorganizes the band-edge "
        "momenta. It is not a single-Fourier-coefficient matrix-element proof. "
        "Integer Q alone does not determine the same-filling gap. Replacing the "
        "cosine radial angle by a quintic smoothstep preserves a direct gap at M "
        "within 2.1% of the baseline width."
    )
    (output / "assessment.json").write_text(
        json.dumps(report["comparisons"], indent=2), encoding="utf-8"
    )

    order = [
        "uniform",
        "skyrmion_q_plus",
        "skyrmion_q_minus",
        "skyrmionium_q_zero",
        "skyrmionium_q_zero_quintic",
        "nonwinding_double_wall_q_zero",
    ]
    labels = ["FM", "$+1$", "$-1$", "cos.", "quin.", "nw."]
    gaps = [
        report["textures"][kind]["locally_refined"]["indirect_gap"]
        for kind in order
    ]
    colors = ["0.55", "#de2d26", "#3182bd", "#756bb1", "#fd8d3c",
              "#31a354"]

    configure_paper_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.75), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.16]},
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.045, wspace=0.08)
    im = texture_panel(axes[0], sky, "skyrmionium")
    texture_panel(axes[1], wall, "same-$m_z$ control")
    outside_panel_label(axes[0], "(a)")
    outside_panel_label(axes[1], "(b)")
    colorbar = fig.colorbar(im, ax=axes[:2], label=r"$m_z$", shrink=0.82,
                           pad=0.02)

    ax = axes[2]
    ax.bar(np.arange(len(gaps)), gaps, color=colors, width=0.72)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(gaps)), labels)
    ax.tick_params(axis="x", pad=1.5)
    ax.set_ylabel(r"same-filling indirect gap $\Delta/t$")
    ax.set_title(
        "same-filling gaps\n" +
        rf"same-$m_z$ change: {100 * relative_gap_change:.2f}%",
        pad=6,
    )
    outside_panel_label(ax, "(c)")
    ax.grid(axis="y", alpha=0.22)

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"texture_gap_controls.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
