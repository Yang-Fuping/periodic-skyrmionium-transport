"""Generate the manuscript figures from frozen production datasets."""

from __future__ import annotations

import csv
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get(
    "SKYRMIONIUM_RESULTS", REPOSITORY / "data" / "results"
)).resolve()
FIGURES = Path(os.environ.get(
    "SKYRMIONIUM_FIGURES", REPOSITORY / "generated_figures"
)).resolve()
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from analyze_qpm_disorder_temperature import load_joined  # noqa: E402
from skyrmion_transport.bloch import find_global_gaps  # noqa: E402
from skyrmion_transport.textures import (  # noqa: E402
    lattice_topological_charge,
    make_texture,
)
from paper_plot_style import (  # noqa: E402
    ANNOTATION_SIZE,
    LEGEND_SIZE,
    configure_paper_style,
)


E_CENTER = 1.0997714941836594
GAP = (1.0771431126351807, 1.1223998757321378)
TEXTURES = (
    ("skyrmionium_q_zero", r"Skyrmionium, $Q=0$"),
    ("skyrmion_q_plus", r"Skyrmion, $Q=+1$"),
    ("skyrmion_q_minus", r"Skyrmion, $Q=-1$"),
)


def configure_style() -> None:
    configure_paper_style()


def panel_label(ax, label: str) -> None:
    """Prepend the panel letter to the title for a uniform manuscript style."""
    ax.set_title(f"{label} {ax.get_title()}")


def outside_panel_label(ax, label: str, x: float = -0.025) -> None:
    """Backward-compatible wrapper for title-prefixed panel letters."""
    panel_label(ax, label)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=450)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def copy_processed_figure(source_dir: Path, source_stem: str,
                          target_stem: str) -> None:
    """Collect a figure produced by a dedicated numerical-analysis script."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        source = source_dir / f"{source_stem}.{suffix}"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing processed figure {source}; run its analysis script first"
            )
        shutil.copyfile(source, FIGURES / f"{target_stem}.{suffix}")


def regenerate_supplementary_analyses() -> None:
    """Rebuild processed supplementary panels from archived numerical arrays."""
    modules = (
        "analyze_texture_gap_controls",
        "analyze_chern_flux_cancellation",
        "analyze_peer_review_convergence",
        "analyze_complex_band_validation",
        "analyze_texture_disorder",
        "analyze_fixed_filling_gap_scan",
        "analyze_mz_form_factor_v2",
        "analyze_probe_width_crossover_v2",
    )
    for name in modules:
        module = importlib.import_module(name)
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()


def gap_candidate(kind: str, A: int, R: float, J: float, nk: int,
                  n_occ: int) -> dict:
    """Load one identified full-zone minigap from a frozen scan."""
    report_path = (RESULTS / "gap_scan" / kind /
                   f"R{R:g}_J{J:g}_nk{nk}" / "report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cell = next((item for item in report["cells"] if item["A"] == A), None)
    if cell is not None:
        return next(item for item in cell["largest_interior_candidates"]
                    if item["n_occ"] == n_occ)
    # A focused rerun may replace report.json while leaving the immutable
    # full-zone eigenvalue archive for the other periods in the same folder.
    with np.load(report_path.parent / f"A{A}.npz") as archive:
        eigenvalues = archive["eigenvalues"]
    item = next(gap for gap in find_global_gaps(eigenvalues, 1e-4)
                if gap["n_occ"] == n_occ)
    item = dict(item)
    item["midgap_energy"] = 0.5 * (item["valence_max"] + item["conduction_min"])
    return item


def figure1() -> None:
    loaded = []
    for kind, title in TEXTURES:
        m = make_texture(kind, 18, 18, 8.0)
        q_total, q_density = lattice_topological_charge(m, periodic=True)
        loaded.append((kind, title, m, q_density, q_total))
    qlim = max(float(np.max(np.abs(item[3]))) for item in loaded)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), constrained_layout=True)
    im_m = im_q = None
    for col, (_, title, m, q_density, q_total) in enumerate(loaded):
        ax = axes[0, col]
        im_m = ax.imshow(m[..., 2].T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
                         interpolation="nearest")
        step = 2
        x, y = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        ax.quiver(x[::step, ::step], y[::step, ::step],
                  m[::step, ::step, 0], m[::step, ::step, 1],
                  color="k", pivot="mid", scale=22, width=0.0042,
                  headwidth=3.0, headlength=3.8)
        ax.set_title(title)
        ax.text(0.985, 0.985, rf"$Q_{{\rm lat}}={q_total:+.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                      "pad": 1.0})
        ax.set_aspect("equal")
        panel_label(ax, f"({'abc'[col]})")

        ax = axes[1, col]
        im_q = ax.imshow(q_density.T, origin="lower", cmap="RdBu_r", vmin=-qlim,
                         vmax=qlim, interpolation="nearest")
        ax.axhline((q_density.shape[1] - 1) / 2, color="0.25", lw=0.45, alpha=0.45)
        ax.set_title(r"lattice solid-angle density $q_p$")
        ax.set_aspect("equal")
        panel_label(ax, f"({'def'[col]})")

    for ax in axes.flat:
        ax.set_xlabel(r"$x/a$")
        ax.set_ylabel(r"$y/a$")
        ax.set_xticks([0, 8, 17])
        ax.set_yticks([0, 8, 17])
    fig.colorbar(im_m, ax=axes[0, :], label=r"$m_z$", shrink=0.82, pad=0.015)
    fig.colorbar(im_q, ax=axes[1, :], label=r"$q_p$", shrink=0.82, pad=0.015)
    save_figure(fig, "figure1_textures_topology")


def figure2() -> None:
    path_data = RESULTS / "paper_main_figure_v1" / "bloch_path_A18_R8_J5.npz"
    with np.load(path_data) as data:
        distance, ticks, bands = data["distance"], data["ticks"], data["bands"]
    length = json.loads((RESULTS / "length_scaling" /
                         "skyrmionium_q_zero_A18_Ny2.json").read_text(encoding="utf-8"))
    length_extended = json.loads((RESULTS / "length_scaling" /
                                  "peer_review_A18_R8_J5_Ny2_Nx1_2_3_4_6_8.json")
                                 .read_text(encoding="utf-8"))
    temp = json.loads((RESULTS / "temperature_length_scaling_v1" /
                       "assessment.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.05), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05,
                                    wspace=0.08, hspace=0.10)

    ax = axes[0, 0]
    for band in range(324, 327):
        ax.plot(distance, bands[:, band], color="#225ea8", lw=1.0)
    ax.axhspan(*GAP, color="#fdae6b", alpha=0.35)
    ax.axhline(E_CENTER, color="0.15", ls=":", lw=0.9)
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_ylim(1.045, 1.200)
    ax.set_ylabel(r"energy $E/t$")
    ax.set_title("zero-Chern minigap")
    ax.text(0.50, E_CENTER, rf"$\Delta/t={GAP[1]-GAP[0]:.4f}$",
            transform=ax.get_yaxis_transform(), ha="center", va="center",
            fontsize=ANNOTATION_SIZE,
            bbox={"facecolor": "white", "edgecolor": "#fdae6b",
                  "alpha": 0.88, "pad": 1.5})
    outside_panel_label(ax, "(a)")

    control_report = json.loads((RESULTS / "texture_gap_controls" /
                                 "A18_R8_J5_n325_nk21" / "report.json")
                                .read_text(encoding="utf-8"))
    control_kinds = ("uniform", "skyrmion_q_plus", "skyrmion_q_minus",
                     "skyrmionium_q_zero", "skyrmionium_q_zero_quintic",
                     "nonwinding_double_wall_q_zero")
    control_labels = ("FM", r"$Q=+1$", r"$Q=-1$", "cosine", "quintic",
                      "nonwinding")
    control_gaps = [control_report["textures"][kind]["locally_refined"]
                    ["indirect_gap"] for kind in control_kinds]
    ax = axes[0, 1]
    bars = ax.bar(np.arange(6), control_gaps,
                  color=("0.55", "#de2d26", "#3182bd", "#756bb1", "#fd8d3c",
                         "#31a354"), width=0.72)
    ax.axhline(0.0, color="0.2", lw=0.65)
    ax.set_xticks(np.arange(6), control_labels, rotation=18, ha="right")
    ax.set_ylabel(r"full-zone gap $\Delta/t$")
    ax.set_title("same-filling texture controls")
    ax.set_ylim(min(control_gaps) - 0.006, max(control_gaps) + 0.008)
    for bar, value in zip(bars, control_gaps):
        offset = 0.0012 if value >= 0 else -0.0012
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                f"{value:.3f}", ha="center", va=va,
                fontsize=ANNOTATION_SIZE)
    outside_panel_label(ax, "(b)")

    ax = axes[1, 0]
    nx = np.asarray(length["parameters"]["Nx"], dtype=float)
    energies = np.asarray(length["parameters"]["energy"], dtype=float)
    transmissions = np.asarray(length["transmissions"], dtype=float)
    styles = (("o-", "#3182bd"), ("s-", "#de2d26"), ("^-", "#31a354"))
    for j, (style, color) in enumerate(styles):
        label = rf"$E/t={energies[j]:.4f}$"
        if j == 1:
            extended_nx = np.asarray(length_extended["parameters"]["Nx"], dtype=float)
            extended_t = np.asarray(length_extended["transmissions"], dtype=float)[:, 0]
            fit = next(iter(length_extended["fits"].values()))
            label += rf", $R^2={fit['r2']:.5f}$"
            ax.semilogy(extended_nx, extended_t, style, color=color, ms=4,
                        label=label)
        else:
            ax.semilogy(nx, transmissions[:, j], style, color=color, ms=4,
                        label=label)
    dense_nx = np.linspace(1, 8, 120)
    physical_length = length_extended["parameters"]["A"] * dense_nx
    ax.semilogy(dense_nx,
                np.exp(fit["intercept"] + fit["slope"] * physical_length),
                color="#de2d26", ls="--", lw=0.9,
                label=rf"fit: $\xi={fit['decay_length']:.2f}a$")
    ax.set_xticks(extended_nx.astype(int))
    ax.set_xlabel(r"array length $N_x$")
    ax.set_ylabel(r"transmission $T_{xx}$")
    ax.set_title("finite-array length scaling")
    ax.legend(loc="lower left", handlelength=1.25, fontsize=LEGEND_SIZE,
              handletextpad=0.4, borderpad=0.25, labelspacing=0.15,
              markerscale=0.75, framealpha=0.90)
    outside_panel_label(ax, "(c)")

    ax = axes[1, 1]
    zero = np.asarray(temp["zero_temperature_reference"]["q0_transmission"])
    ax.semilogy(nx, zero, "o-", color="#225ea8", ms=4, label=r"$T=0$, $Q=0$")
    for entry, marker, color in zip(temp["finite_temperature"], ("s", "^"),
                                    ("#fd8d3c", "#e31a1c")):
        kbt = entry["kBT"]
        q0 = np.asarray(entry["conductance"]["skyrmionium_q_zero"])
        ax.semilogy(nx, q0, marker + "--", color=color, ms=4,
                    label=rf"$k_BT/t={kbt:g}$, $Q=0$")
    qpm = np.asarray(temp["finite_temperature"][1]["conductance"]["skyrmion_q_plus"])
    ax.semilogy(nx, qpm, "d:", color="#756bb1", ms=4,
                label=r"$k_BT/t=0.01$, $Q=\pm1$")
    ax.set_xticks(nx.astype(int))
    ax.set_xlabel(r"array length $N_x$")
    ax.set_ylabel(r"$G/(e^2/h)$")
    ax.set_title("Fermi-window averaging")
    ax.legend(loc="lower left", handlelength=1.25, fontsize=LEGEND_SIZE,
              handletextpad=0.4, borderpad=0.25, labelspacing=0.15,
              markerscale=0.75, framealpha=0.90)
    outside_panel_label(ax, "(d)")

    for ax in axes.flat:
        ax.grid(alpha=0.22, which="both")
    save_figure(fig, "figure2_minigap_transport")


def hall_position_rows() -> list[dict[str, str]]:
    rows = []
    path = RESULTS / "hall_mechanism_v1" / "summary.csv"
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row["scan"] == "position_w2"
                    and row["kind"] == "skyrmionium_q_zero"
                    and abs(float(row["energy"]) - E_CENTER) < 1e-10):
                rows.append(row)
    return sorted(rows, key=lambda row: int(row["probe_start"]))


def figure3() -> None:
    rows, _, _ = load_joined()
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 paired disorder records, found {len(rows)}")
    fig = plt.figure(figsize=(7.2, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)

    ax_local = fig.add_subplot(grid[0, 0:2])
    ax = ax_local
    hall_rows = hall_position_rows()
    x = np.asarray([int(row["probe_start"]) + 0.5 for row in hall_rows])
    hall = np.asarray([float(row["Rxy_h_over_e2"]) for row in hall_rows])
    qwin = np.asarray([float(row["windowed_topological_charge"]) for row in hall_rows])
    scale = float(np.dot(qwin, hall) / np.dot(qwin, qwin))
    ax.plot(x, hall, "o-", color="#225ea8", ms=4, label=r"$R_{xy}$")
    ax.plot(x, scale * qwin, "s--", color="#e6550d", ms=3.5,
            label=r"scaled $Q_{\rm window}$")
    ax.axhline(0, color="0.35", lw=0.7)
    ax.set_xlabel(r"probe-window center $x/a$")
    ax.set_ylabel(r"$R_{xy}$ ($h/e^2$)")
    ax.set_title("local Hall compensation")
    ax.set_ylim(min(hall.min(), (scale * qwin).min()) - 0.001,
                max(hall.max(), (scale * qwin).max()) + 0.006)
    ax.legend(loc="upper center", ncol=2, fontsize=LEGEND_SIZE,
              handlelength=1.4, columnspacing=0.9,
              borderpad=0.3, labelspacing=0.2, framealpha=0.93)
    outside_panel_label(ax, "(a)", x=-0.075)

    width_assessment = json.loads((RESULTS / "probe_width_crossover_v2" /
                                   "assessment.json").read_text(encoding="utf-8"))
    probe_widths = np.arange(1, 17)
    compensation = np.asarray(width_assessment["compensation_ratio"])
    ax_width = fig.add_subplot(grid[0, 2:4])
    ax = ax_width
    ax.semilogy(probe_widths, compensation, "o-", ms=3.2, lw=1.0,
                color="#756bb1")
    ax.axhline(0.1, color="0.35", ls="--", lw=0.7)
    width_bottom, _ = ax.get_ylim()
    ax.set_ylim(width_bottom, 0.14)
    ax.text(0.97, 0.1, r"$C_{\rm rel}=0.1$",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=ANNOTATION_SIZE)
    ax.set_xticks((1, 4, 8, 12, 16))
    ax.set_xlabel(r"probe width $w_p/a$")
    ax.set_ylabel(r"relative suppression $C_{\rm rel}(w_p)$")
    ax.set_title("coherent width dependence")
    outside_panel_label(ax, "(b)", x=-0.075)

    ax_spectra = fig.add_subplot(grid[0, 4:6])
    ax = ax_spectra
    group = [row for row in rows if row["Wd"] == 0.5]
    energy = np.asarray(group[0]["energy"])
    q0_spectra = np.asarray([
        row["result"]["skyrmionium_q_zero"]["8"]["transmission"] for row in group
    ])
    qp_spectra = np.asarray([
        row["result"]["skyrmion_q_plus"]["8"]["transmission"] for row in group
    ])
    for spectra, color, label in (
        (q0_spectra, "#225ea8", r"$Q=0$"),
        (qp_spectra, "#de2d26", r"$Q=\pm1$"),
    ):
        median = np.median(spectra, axis=0)
        q25, q75 = np.quantile(spectra, (0.25, 0.75), axis=0)
        ax.semilogy(energy, np.maximum(median, 1e-18), color=color, label=label)
        ax.fill_between(energy, np.maximum(q25, 1e-18), np.maximum(q75, 1e-18),
                        color=color, alpha=0.17, linewidth=0)
    ax.axvspan(*GAP, color="0.65", alpha=0.20)
    ax.set_xlabel(r"energy $E/t$")
    ax.set_ylabel(r"median $T_{xx}$")
    ax.set_title("disorder spectra")
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, handlelength=1.4,
              borderpad=0.3, labelspacing=0.25)
    outside_panel_label(ax, "(c)", x=-0.075)

    ax_box = fig.add_subplot(grid[1, 0:3])
    ax = ax_box
    box_data, labels, colors = [], [], []
    for wd in (0.25, 0.5):
        group = [row for row in rows if row["Wd"] == wd]
        q0 = np.asarray([
            row["result"]["skyrmionium_q_zero"]["8"]["thermal"]["0.01"]
            for row in group
        ])
        qp = np.asarray([
            row["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
            for row in group
        ])
        box_data.extend((q0, qp))
        labels.extend((f"$Q=0$\n$W_d/t={wd:g}$", f"$Q=\\pm1$\n$W_d/t={wd:g}$"))
        colors.extend(("#3182bd", "#de2d26"))
    boxes = ax.boxplot(box_data, tick_labels=labels, showfliers=True, patch_artist=True,
                       flierprops={"markersize": 2, "alpha": 0.45})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.35)
    ax.set_yscale("log")
    ax.set_ylabel(r"$G_8/(e^2/h)$ at $k_BT/t=0.01$")
    ax.set_title("100 paired disorder samples")
    outside_panel_label(ax, "(d)", x=-0.075)

    ax_ratio = fig.add_subplot(grid[1, 3:6])
    ax = ax_ratio
    ratio_data = []
    ratio_labels = []
    for wd in (0.25, 0.5):
        group = [row for row in rows if row["Wd"] == wd]
        q0 = np.asarray([
            row["result"]["skyrmionium_q_zero"]["8"]["thermal"]["0.01"]
            for row in group
        ])
        qp = np.asarray([
            row["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
            for row in group
        ])
        ratio = q0 / qp
        ratio_data.append(ratio)
        ratio_labels.append(rf"$W_d/t={wd:g}$")
    ax.boxplot(ratio_data, tick_labels=ratio_labels, showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.45})
    ax.axhline(1, color="0.25", ls=":", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylabel(r"$G_{Q=0}/G_{Q=\pm1}$")
    ax.set_title("same-disorder suppression ratio")
    outside_panel_label(ax, "(e)", x=-0.075)

    for ax in (ax_local, ax_width, ax_spectra, ax_box, ax_ratio):
        ax.grid(alpha=0.22, which="both", axis="y")
    save_figure(fig, "figure3_hall_disorder")


def figure4() -> None:
    """Tunability slices and a data-backed texture-switching metric."""
    j_values = np.asarray([4.0, 4.25, 4.5, 4.75, 5.0])
    j_gaps = [gap_candidate("skyrmionium_q_zero", 18, 8, j, 11, 325)
              for j in j_values]

    a_values = np.asarray([18, 20, 24])
    a_gaps = [gap_candidate("skyrmionium_q_zero", int(a), 8, 5, 11,
                            int(a * a + 1)) for a in a_values]

    r_values = np.asarray([5.0, 6.0, 7.0, 8.0])
    r_gaps = [gap_candidate("skyrmionium_q_zero", 18, r, 5, 11, 325)
              for r in r_values]

    assessment = json.loads((RESULTS / "disorder_topology_comparison_v1" /
                             "assessment.json").read_text(encoding="utf-8"))
    summary = {float(item["Wd"]): item for item in assessment["summary"]}
    disorder = np.asarray([0.25, 0.5])
    temperature = "0.01"
    contrast = np.empty(2)
    ci_low = np.empty_like(contrast)
    ci_high = np.empty_like(contrast)
    for iw, strength in enumerate(disorder):
        comparison = summary[strength]["temperature"][temperature][
            "paired_topology_comparison"]
        ratio = comparison["Q0_over_mean_Qpm"]["median"]
        ratio_ci = comparison["Q0_over_mean_Qpm_median_bootstrap_95ci"]
        contrast[iw] = 1.0 / ratio
        ci_low[iw] = 1.0 / ratio_ci[1]
        ci_high[iw] = 1.0 / ratio_ci[0]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06,
                                    wspace=0.08, hspace=0.12)

    ax = axes[0, 0]
    centers = np.asarray([item["midgap_energy"] for item in j_gaps]) - j_values
    lower = np.asarray([item["valence_max"] for item in j_gaps]) - j_values
    upper = np.asarray([item["conduction_min"] for item in j_gaps]) - j_values
    ax.fill_between(j_values, lower, upper, color="#9ecae1", alpha=0.65,
                    label="full-zone minigap")
    ax.plot(j_values, centers, "o-", color="#08519c", label="gap center")
    ax.set(xlabel=r"exchange coupling $J/t$",
           ylabel=r"shifted energy $(E-J)/t$")
    ax.set_title("exchange-shifted window")
    ax.legend(loc="lower right")
    outside_panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(a_values, [item["indirect_gap"] for item in a_gaps], "o-",
            color="#cb181d")
    ax.set(xlabel=r"array period $A/a$", ylabel=r"minigap $\Delta/t$")
    ax.set_title("period-controlled gap width")
    ax.set_xticks(a_values)
    outside_panel_label(ax, "(b)")

    ax = axes[1, 0]
    ax.plot(r_values, [item["indirect_gap"] for item in r_gaps], "o-",
            color="#238b45")
    ax.set(xlabel=r"texture radius $R/a$", ylabel=r"minigap $\Delta/t$")
    ax.set_title("radius-controlled gap width")
    ax.set_xticks(r_values)
    outside_panel_label(ax, "(c)")

    ax = axes[1, 1]
    x = np.arange(len(disorder))
    width = 0.52
    yerr = np.vstack((contrast - ci_low, ci_high - contrast))
    ax.bar(x, contrast, width=width, color="#e6550d", alpha=0.80,
           label=rf"$k_BT/t={temperature}$")
    ax.errorbar(x, contrast, yerr=yerr, fmt="none", ecolor="0.15",
                elinewidth=0.8, capsize=2.3)
    ax.axhline(1, color="0.25", ls=":", lw=0.9)
    ax.set_xticks(x, [rf"${value:g}$" for value in disorder])
    ax.set_yscale("log")
    ax.set_ylim(0.8, 40.0)
    ax.set(xlabel=r"disorder $W_d/t$",
           ylabel="conductance contrast\n" + r"$G_{Q=\pm1}/G_{Q=0}$")
    ax.set_title("texture-selective conductance switching")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.985),
              fontsize=LEGEND_SIZE, handlelength=1.35,
              framealpha=0.93, borderpad=0.3, labelspacing=0.2)
    outside_panel_label(ax, "(d)")

    for ax in axes.flat:
        ax.grid(alpha=0.22, axis="y")
    save_figure(fig, "figure4_tunability_applications")


def supplementary_full_bz_gap() -> None:
    """Full-zone band-edge maps supporting the global-gap claim."""
    root = (RESULTS / "full_bz_gap" /
            "skyrmionium_q_zero_A18_R8_J5_n325")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    entry = next(item for item in report["convergence"] if item["nk"] == 31)
    with np.load(root / "nk31.npz") as data:
        ks = data["k_values"] * 18.0 / np.pi
        valence = data["valence"]
        conduction = data["conduction"]
        direct = data["direct_gap_map"]

    extent = [float(ks[0]), float(ks[-1]), float(ks[0]), float(ks[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.04, wspace=0.10)
    panels = (
        (valence, "valence edge", "viridis", r"$E_{324}/t$"),
        (conduction, "conduction edge", "viridis", r"$E_{325}/t$"),
        (direct, "direct gap", "magma",
         r"$(E_{325}-E_{324})/t$"),
    )
    for index, (ax, (values, title, cmap, colorbar_label)) in enumerate(
            zip(axes, panels)):
        image = ax.imshow(values.T, origin="lower", extent=extent,
                          interpolation="nearest", cmap=cmap, aspect="equal")
        ax.set_xlim(-1.06, 1.0)
        ax.set_ylim(-1.06, 1.0)
        ax.set_xlabel(r"$k_xA/\pi$")
        ax.set_ylabel(r"$k_yA/\pi$")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, pad=0.02, shrink=0.86,
                     label=colorbar_label)
        panel_label(ax, f"({'abc'[index]})")
    kv = np.asarray(entry["valence_max_k"]) * 18.0 / np.pi
    kc = np.asarray(entry["conduction_min_k"]) * 18.0 / np.pi
    axes[0].plot(kv[0], kv[1], marker="*", ms=7, mfc="white", mec="0.15",
                 mew=0.6, label=r"$\max E_{324}$")
    axes[1].plot(kc[0], kc[1], marker="*", ms=7, mfc="white", mec="0.15",
                 mew=0.6, label=r"$\min E_{325}$")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    save_figure(fig, "supplementary_figure_s1_full_bz_gap")


def supplementary_r7_length_scaling() -> None:
    """Independent non-baseline length scaling used as a robustness check."""
    path = (RESULTS / "length_scaling" /
            "skyrmionium_q_zero_A18_R7_J5_Ny2.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    nx = np.asarray(result["parameters"]["Nx"], dtype=float)
    energies = np.asarray(result["parameters"]["energy"], dtype=float)
    transmission = np.asarray(result["transmissions"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    for index, (marker, color) in enumerate(zip(("o", "s", "^"),
                                                ("#3182bd", "#e6550d", "#31a354"))):
        fit = result["fits"][str(energies[index])]
        ax.semilogy(nx, transmission[:, index], marker + "-", color=color,
                    ms=4, label=rf"$E/t={energies[index]:.4f}$, "
                                rf"$R^2={fit['r2']:.4f}$")
    ax.set_xticks(nx.astype(int))
    ax.set_xlabel(r"array length $N_x$")
    ax.set_ylabel(r"transmission $T_{xx}$")
    ax.set_title(r"non-baseline scaling, $R=7a$")
    ax.grid(alpha=0.22, which="both")
    ax.legend(loc="lower left")
    save_figure(fig, "supplementary_figure_s5_r7_length_scaling")


def main() -> None:
    if not RESULTS.is_dir():
        raise SystemExit(
            "Paper dataset not found. Extract the Zenodo dataset to "
            f"{RESULTS} or set SKYRMIONIUM_RESULTS to its results directory."
        )
    configure_style()
    regenerate_supplementary_analyses()
    figure1()
    figure2()
    figure3()
    figure4()
    supplementary_full_bz_gap()
    copy_processed_figure(
        RESULTS / "peer_review_texture_controls",
        "texture_gap_controls",
        "supplementary_figure_s2_texture_gap_controls",
    )
    copy_processed_figure(
        RESULTS / "peer_review_chern_validation",
        "chern_flux_cancellation",
        "supplementary_figure_s3_chern_flux_cancellation",
    )
    copy_processed_figure(
        RESULTS / "peer_review_convergence",
        "length_width_convergence",
        "supplementary_figure_s4_length_width_convergence",
    )
    supplementary_r7_length_scaling()
    copy_processed_figure(
        RESULTS / "peer_review_complex_band",
        "complex_band_validation",
        "supplementary_figure_s6_complex_band_validation",
    )
    copy_processed_figure(
        RESULTS / "texture_disorder",
        "texture_disorder_robustness",
        "supplementary_figure_s7_texture_disorder",
    )
    copy_processed_figure(
        RESULTS / "fixed_filling_gap_scan" / "ratio_0.44444444_J5_nk11",
        "fixed_filling_gap_scaling",
        "supplementary_figure_s8_fixed_filling_scaling",
    )
    copy_processed_figure(
        RESULTS / "peer_review_mz_form_factor_v2",
        "mz_form_factor_v2",
        "supplementary_figure_s9_mz_form_factor",
    )
    copy_processed_figure(
        RESULTS / "probe_width_crossover_v2",
        "probe_width_crossover_v2",
        "supplementary_figure_s10_probe_width_crossover",
    )
    print(json.dumps({
        "figures": [
            str(FIGURES / "figure1_textures_topology.pdf"),
            str(FIGURES / "figure2_minigap_transport.pdf"),
            str(FIGURES / "figure3_hall_disorder.pdf"),
            str(FIGURES / "figure4_tunability_applications.pdf"),
            str(FIGURES / "supplementary_figure_s1_full_bz_gap.pdf"),
            str(FIGURES / "supplementary_figure_s2_texture_gap_controls.pdf"),
            str(FIGURES / "supplementary_figure_s3_chern_flux_cancellation.pdf"),
            str(FIGURES / "supplementary_figure_s4_length_width_convergence.pdf"),
            str(FIGURES / "supplementary_figure_s5_r7_length_scaling.pdf"),
            str(FIGURES / "supplementary_figure_s6_complex_band_validation.pdf"),
            str(FIGURES / "supplementary_figure_s7_texture_disorder.pdf"),
            str(FIGURES / "supplementary_figure_s8_fixed_filling_scaling.pdf"),
            str(FIGURES / "supplementary_figure_s9_mz_form_factor.pdf"),
            str(FIGURES / "supplementary_figure_s10_probe_width_crossover.pdf"),
        ]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
