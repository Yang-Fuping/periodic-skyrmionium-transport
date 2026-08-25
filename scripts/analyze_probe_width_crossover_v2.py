"""Analyze the continuous Hall-probe-width crossover without assuming monotonicity."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import RESULTS
from paper_plot_style import LEGEND_SIZE, TITLE_SIZE, configure_paper_style


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
COLORS = {
    "uniform": "0.4",
    "skyrmionium_q_zero": "#225ea8",
    "skyrmion_q_plus": "#de2d26",
    "skyrmion_q_minus": "#31a354",
}
LABELS = {
    "uniform": "FM",
    "skyrmionium_q_zero": r"$Q=0$",
    "skyrmion_q_plus": r"$Q=+1$",
    "skyrmion_q_minus": r"$Q=-1$",
}
PLOT_STYLES = {
    "uniform": {
        "marker": "x", "linestyle": ":", "markersize": 4.0,
        "markeredgewidth": 0.9, "zorder": 6,
    },
    "skyrmionium_q_zero": {
        "marker": "o", "linestyle": "--", "markersize": 5.0,
        "markerfacecolor": "none", "markeredgewidth": 1.0, "zorder": 5,
    },
    "skyrmion_q_plus": {
        "marker": "s", "linestyle": "-", "markersize": 3.2,
        "markeredgewidth": 0.6, "zorder": 4,
    },
    "skyrmion_q_minus": {
        "marker": "^", "linestyle": "-", "markersize": 3.4,
        "markeredgewidth": 0.6, "zorder": 4,
    },
}


def outside_panel_label(ax, label: str) -> None:
    ax.set_title(f"{label} {ax.get_title()}")


def main() -> None:
    folder = RESULTS / "probe_width_crossover_v2"
    rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    summary = {}
    for kind in KINDS:
        summary[kind] = {}
        for width in range(1, 17):
            group = [row["result"] for row in rows
                     if row["kind"] == kind and row["probe_width"] == width]
            if not group:
                raise RuntimeError(f"Missing {kind}, probe width {width}")
            angles = np.asarray([row["charge_hall_angle"] for row in group])
            rxy = np.asarray([row["Rxy_h_over_e2"] for row in group])
            summary[kind][str(width)] = {
                "sampled_starts": sorted({int(row0["probe_start"]) for row0 in rows
                                           if row0["kind"] == kind
                                           and row0["probe_width"] == width}),
                "mean_charge_hall_angle": float(np.mean(angles)),
                "mirror_spread_charge_hall_angle": float(np.ptp(angles)),
                "mean_Rxy_h_over_e2": float(np.mean(rxy)),
                "mirror_spread_Rxy_h_over_e2": float(np.ptp(rxy)),
                "max_current_conservation_error": float(max(
                    row["current_conservation_error"] for row in group)),
                "max_probe_current_error": float(max(
                    row["probe_current_error"] for row in group)),
                "max_gauge_invariance_error": float(max(
                    row["gauge_invariance_error"] for row in group)),
                "max_scattering_unitarity_error": float(max(
                    row["scattering_unitarity_error"] for row in group)),
                "all_valid": bool(all(row["valid_hall_point"] for row in group)),
            }
    widths = np.arange(1, 17)
    angles = {
        kind: np.asarray([summary[kind][str(w)]["mean_charge_hall_angle"]
                          for w in widths]) for kind in KINDS
    }
    denominator = 0.5 * (np.abs(angles["skyrmion_q_plus"])
                         + np.abs(angles["skyrmion_q_minus"]))
    compensation = np.abs(angles["skyrmionium_q_zero"]) / denominator
    minimum_index = int(np.argmin(compensation))
    assessment = {
        "observable_definition": (
            "C_rel(w_p)=|mean theta_TH(Q=0)| / "
            "{[|mean theta_TH(Q=+1)|+|mean theta_TH(Q=-1)|]/2}"
        ),
        "summary": summary,
        "compensation_ratio": compensation.tolist(),
        "minimum": {
            "probe_width": int(widths[minimum_index]),
            "C": float(compensation[minimum_index]),
        },
        "wide_probe": {"probe_width": 16, "C": float(compensation[-1])},
        "monotonic_decrease": bool(np.all(np.diff(compensation) <= 0.0)),
        "interpretation": (
            "The systematic scan demonstrates a probe-dependent crossover but "
            "not a monotonic approach to zero.  Intermediate and wide windows "
            "strongly suppress Q=0 relative to Q=+-1, while coherent finite-device "
            "interference produces width-to-width rebounds.  C_rel is a comparative "
            "Hall-suppression metric relative to matched Q=+-1 textures, not an "
            "intrinsic decomposition of the two skyrmionium rings.  Probe width "
            "changes both spatial coverage and the coherent contact boundary "
            "condition.  The result supports probe-dependent local compensation, "
            "not a bulk Hall limit."
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )

    coupling_folder = RESULTS / "peer_review_v4_probe_coupling"
    coupling_assessment = json.loads(
        (coupling_folder / "assessment.json").read_text(encoding="utf-8")
    )
    coupling_rows = [json.loads(line) for line in
                     (coupling_folder / "raw.jsonl").read_text(
                         encoding="utf-8").splitlines() if line.strip()]

    configure_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    axes = axes.ravel()
    fig.set_constrained_layout_pads(
        w_pad=0.055, h_pad=0.055, wspace=0.08, hspace=0.10
    )
    for kind in KINDS:
        axes[0].plot(
            widths, angles[kind], lw=0.9, color=COLORS[kind],
            label=LABELS[kind], **PLOT_STYLES[kind]
        )
    axes[0].axhline(0.0, color="0.35", lw=0.7)
    axes[0].set(xlabel=r"probe width $w_p/a$",
                ylabel=r"charge Hall angle $\theta_{\rm TH}$",
                title="signed Hall response",
                xticks=(1, 4, 8, 12, 16))
    axes[0].legend(
        loc="upper right", ncol=2, fontsize=LEGEND_SIZE,
        handlelength=1.25, columnspacing=0.7, borderpad=0.28,
        labelspacing=0.25, framealpha=0.92,
    )
    outside_panel_label(axes[0], "(a)")
    axes[1].semilogy(widths, compensation, "o-", color="#756bb1", ms=3.0)
    axes[1].axhline(0.1, color="0.35", ls="--", lw=0.8,
                   label=r"$C_{\rm rel}=0.1$")
    axes[1].set(xlabel=r"probe width $w_p/a$",
                ylabel=r"relative suppression $C_{\rm rel}$",
                title="width dependence",
                xticks=(1, 4, 8, 12, 16))
    axes[1].legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.92)
    outside_panel_label(axes[1], "(b)")

    selected_widths = np.asarray((2, 8, 16))
    colors = {0.5: "#3182bd", 0.75: "#756bb1", 1.0: "#e6550d"}
    styles = {
        0.5: {
            "marker": "o", "linestyle": "-", "markersize": 4.6,
            "markerfacecolor": "none", "markeredgewidth": 1.0, "zorder": 4,
        },
        0.75: {
            "marker": "s", "linestyle": "--", "markersize": 3.9,
            "markerfacecolor": "none", "markeredgewidth": 0.9, "zorder": 5,
        },
        1.0: {
            "marker": "^", "linestyle": "-", "markersize": 3.2,
            "markeredgewidth": 0.6, "zorder": 6,
        },
    }
    for coupling in (0.5, 0.75, 1.0):
        coupling_summary = coupling_assessment["couplings"][str(coupling)]
        ratios = [coupling_summary["width_scan"][str(width)][
            "relative_hall_suppression_ratio"] for width in selected_widths]
        axes[2].semilogy(selected_widths, ratios, lw=0.9,
                        color=colors[coupling], **styles[coupling],
                        label=rf"${coupling:g}$")
        position = sorted([
            row for row in coupling_rows
            if row["case"]["scan"] == "position_coupling"
            and row["case"]["probe_coupling"] == coupling
        ], key=lambda row: row["case"]["probe_start"])
        centers = np.asarray([row["case"]["probe_start"] + 0.5
                              for row in position])
        hall = np.asarray([row["result"]["Rxy_h_over_e2"] for row in position])
        axes[3].plot(centers, hall, lw=0.8,
                     color=colors[coupling], **styles[coupling],
                     label=rf"${coupling:g}$")
    axes[2].axhline(0.1, color="0.35", ls="--", lw=0.8)
    axes[2].set(xlabel=r"probe width $w_p/a$",
                ylabel=r"relative suppression $C_{\rm rel}$",
                title="interface coupling",
                xticks=selected_widths)
    axes[3].axhline(0.0, color="0.35", lw=0.7)
    axes[3].set(xlabel=r"probe-window centre $x/a$",
                ylabel=r"$R_{xy}$ ($h/e^2$)",
                title="local sign sequence")
    coupling_handles, coupling_labels = axes[2].get_legend_handles_labels()
    axes[2].legend(
        coupling_handles, coupling_labels, title=r"$t_c/t$ for (c),(d)",
        loc="upper center", ncol=3, fontsize=LEGEND_SIZE,
        handlelength=1.3, columnspacing=0.75, borderpad=0.28,
        labelspacing=0.22, framealpha=0.92, title_fontsize=LEGEND_SIZE,
    )
    outside_panel_label(axes[2], "(c)")
    outside_panel_label(axes[3], "(d)")
    for ax in axes:
        ax.title.set_fontsize(TITLE_SIZE)
        ax.grid(alpha=0.22, which="both")
    for suffix in ("png", "pdf"):
        fig.savefig(
            folder / f"probe_width_crossover_v2.{suffix}", dpi=300,
            bbox_inches="tight", pad_inches=0.03,
        )
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
