"""Plot and summarize the complex-band peer-review validation."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import RESULTS
from paper_plot_style import LEGEND_SIZE, configure_paper_style


def outside_panel_label(ax, label: str) -> None:
    ax.set_title(f"{label} {ax.get_title()}")


def main() -> None:
    source = RESULTS / "peer_review_complex_band" / "report.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    output = source.parent

    configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.045, wspace=0.08)
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    axes[0].plot(np.cos(theta), np.sin(theta), "k--", lw=1.0, label=r"$|\lambda|=1$")
    colors = ["tab:blue", "tab:red", "tab:green"]
    labels = [r"$E/t=1.065$", r"$E/t=1.0998$", r"$E/t=1.150$"]
    for row, color, label in zip(report["energy_rows"], colors, labels):
        values = np.asarray(row["multipliers"], dtype=float)
        complex_values = values[:, 0] + 1j * values[:, 1]
        visible = np.abs(complex_values) <= 1.05
        axes[0].scatter(
            complex_values.real[visible],
            complex_values.imag[visible],
            s=14,
            alpha=0.75,
            color=color,
            label=label,
        )
        selected = complex(*row["mode"]["multiplier"])
        axes[0].scatter(
            [selected.real], [selected.imag], s=66, facecolors="none",
            edgecolors=color, linewidths=1.6,
        )
    axes[0].set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08),
                 xlabel=r"Re $\lambda$", ylabel=r"Im $\lambda$",
                 title="strip Bloch multipliers")
    axes[0].set_aspect("equal")
    axes[0].legend(
        fontsize=LEGEND_SIZE, loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, framealpha=0.92, handlelength=1.6, columnspacing=0.9,
    )
    outside_panel_label(axes[0], "(a)")

    widths = np.asarray([row["Ny"] for row in report["width_rows"]])
    cbs = np.asarray([
        row["complex_band"]["xi_transmission_a"] for row in report["width_rows"]
    ])
    global_fit = np.asarray([
        row["finite_array_global_fit_xi_a"] for row in report["width_rows"]
    ])
    terminal = np.asarray([
        row["finite_array_terminal_pair_xi_a"] for row in report["width_rows"]
    ])
    axes[1].plot(widths, cbs, "o-", color="black", label=r"complex band $1/(2\kappa)$")
    axes[1].plot(widths, global_fit, "s--", color="tab:blue", label="global finite-length fit")
    axes[1].plot(widths, terminal, "^--", color="tab:red", label="longest-length pair")
    axes[1].set(
        xlabel=r"transverse cell count $N_y$",
        ylabel=r"attenuation length $\xi/a$",
        title="strip-finite correspondence",
        xticks=widths,
    )
    axes[1].grid(alpha=0.23)
    axes[1].legend(fontsize=LEGEND_SIZE, loc="upper left", framealpha=0.92)
    outside_panel_label(axes[1], "(b)")

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"complex_band_validation.{suffix}", dpi=300)
    plt.close(fig)

    assessment = {
        "baseline_Ny2_complex_band_xi_a": cbs[0],
        "baseline_Ny2_six_point_fit_xi_a": global_fit[0],
        "baseline_relative_difference": abs(global_fit[0] / cbs[0] - 1.0),
        "maximum_longest_pair_relative_difference": max(
            row["relative_terminal_pair_xi_difference"]
            for row in report["width_rows"]
        ),
        "outside_gap_energies_have_propagating_modes": all(
            report["energy_rows"][index]["mode"]["has_propagating_mode"]
            for index in (0, 2)
        ),
        "gap_centre_has_no_propagating_mode": not report["energy_rows"][1]["mode"][
            "has_propagating_mode"
        ],
        "selected_mode_qep_residual_below_1e-12": all(
            row["complex_band"]["selected_qep_residual"] < 1e-12
            for row in report["width_rows"]
        ),
        "reciprocal_pair_error_below_1e-12": all(
            row["complex_band"]["reciprocal_pair_error"] < 1e-12
            for row in report["width_rows"]
        ),
    }
    (output / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
