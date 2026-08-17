"""Assemble a six-panel main-figure draft from the converged production datasets."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from analyze_qpm_disorder_temperature import KINDS, LABELS, COLORS, load_joined
from skyrmion_transport.bloch import band_structure, high_symmetry_path
from skyrmion_transport.textures import make_texture


E_CENTER = 1.0997714941836594
GAP = (1.0771431126351807, 1.1223998757321378)


def bloch_path(out):
    path = out / "bloch_path_A18_R8_J5.npz"
    if path.exists():
        with np.load(path) as saved:
            return tuple(saved[key].copy() for key in ("distance", "ticks", "bands"))
    cell = make_texture("skyrmionium_q_zero", 18, 18, 8.0)
    kpath, distance, ticks, _ = high_symmetry_path(18, points_per_segment=24)
    bands = band_structure(cell, kpath, 5.0, 1.0)
    np.savez_compressed(path, kpath=kpath, distance=distance, ticks=ticks, bands=bands)
    return distance, ticks, bands


def hall_position_rows():
    path = ROOT / "results" / "hall_mechanism_v1" / "summary.csv"
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row["scan"] == "position_w2" and row["kind"] == "skyrmionium_q_zero"
                    and abs(float(row["energy"]) - E_CENTER) < 1e-10):
                rows.append(row)
    return sorted(rows, key=lambda row: int(row["probe_start"]))


def main():
    out = ROOT / "results" / "paper_main_figure_v1"
    out.mkdir(parents=True, exist_ok=True)
    rows, _, _ = load_joined()
    if len(rows) != 200:
        raise RuntimeError(f"Need 200 joined disorder samples, found {len(rows)}")
    distance, ticks, bands = bloch_path(out)
    temperature = json.loads((ROOT / "results" / "temperature_length_scaling_v1" /
                              "assessment.json").read_text(encoding="utf-8"))
    thermal = next(row for row in temperature["finite_temperature"] if row["kBT"] == .01)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)

    # (a) Isolated zero-Chern miniband and the confirmed full-zone gap.
    ax = axes[0, 0]
    # Show only the three positive-energy bands surrounding the selected gap.
    # Higher bands 327/328 lie outside this local energy window and previously
    # touched the upper frame, which visually looked like an unclipped curve.
    for band in range(324, 327):
        ax.plot(distance, bands[:, band], color="tab:blue", linewidth=.9)
    ax.axhspan(*GAP, color="tab:orange", alpha=.22, label=rf"$\Delta/t={GAP[1]-GAP[0]:.4f}$")
    ax.axhline(E_CENTER, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"])
    ax.set_ylim(1.045, 1.200); ax.set_ylabel("Energy E/t")
    ax.set_title("(a) zero-Chern Q=0 miniband gap")
    ax.grid(alpha=.2); ax.legend(fontsize=8)

    # (b) Clean tunnelling-to-thermal crossover.
    ax = axes[0, 1]
    nx = np.asarray([1, 2, 4, 8])
    zero = np.asarray(temperature["zero_temperature_reference"]["q0_transmission"])
    ax.semilogy(nx, zero, "o-", color="tab:blue", label=r"$T=0$, Q=0")
    ax.semilogy(nx, thermal["conductance"]["skyrmionium_q_zero"], "s--",
                color="tab:orange", label=r"$k_BT/t=0.01$, Q=0")
    ax.semilogy(nx, thermal["conductance"]["skyrmion_q_plus"], "^:",
                color="tab:red", label=r"$k_BT/t=0.01$, Q=+1")
    ax.set_xticks(nx); ax.set_xlabel(r"array length $N_x$")
    ax.set_ylabel(r"$T(E_F)$ or $G/(e^2/h)$")
    ax.set_title("(b) clean finite-array crossover")
    ax.grid(alpha=.25); ax.legend(fontsize=7)

    # (c) Median disordered spectra at the stronger selected disorder.
    ax = axes[0, 2]
    group = [row for row in rows if row["Wd"] == .5]
    energy = np.asarray(group[0]["energy"])
    for kind in KINDS:
        spectra = np.asarray([row["result"][kind]["8"]["transmission"] for row in group])
        median = np.median(spectra, axis=0)
        q25, q75 = np.quantile(spectra, (.25, .75), axis=0)
        linestyle = "--" if kind == "skyrmion_q_minus" else "-"
        ax.semilogy(energy, np.maximum(median, 1e-18), color=COLORS[kind],
                    linestyle=linestyle, label=LABELS[kind])
        ax.fill_between(energy, np.maximum(q25, 1e-18), np.maximum(q75, 1e-18),
                        color=COLORS[kind], alpha=.12)
    ax.axvspan(*GAP, color="gray", alpha=.12)
    ax.set_xlabel("Energy E/t"); ax.set_ylabel("median transmission")
    ax.set_title(r"(c) disorder spectra, $W_d/t=0.5$")
    ax.grid(alpha=.2); ax.legend(fontsize=8)

    # (d) Same-disorder topology-resolved ensemble conductance.
    ax = axes[1, 0]
    data, labels, colors = [], [], []
    for wd in (.25, .5):
        group = [row for row in rows if row["Wd"] == wd]
        for kind in KINDS:
            data.append([row["result"][kind]["8"]["thermal"]["0.01"] for row in group])
            labels.append(f"{LABELS[kind]}\nW={wd:g}"); colors.append(COLORS[kind])
    boxes = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(.35)
    ax.set_yscale("log"); ax.set_ylabel(r"$G_8/(e^2/h)$")
    ax.set_title(r"(d) 100-sample topology comparison")
    ax.grid(axis="y", alpha=.25)

    # (e) Direct paired suppression relative to the matched Q=+/-1 mean.
    ax = axes[1, 1]
    ratio_data = []
    for wd in (.25, .5):
        group = [row for row in rows if row["Wd"] == wd]
        q0 = np.asarray([row["result"]["skyrmionium_q_zero"]["8"]["thermal"]["0.01"]
                         for row in group])
        qpm = .5 * np.asarray([
            row["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
            + row["result"]["skyrmion_q_minus"]["8"]["thermal"]["0.01"]
            for row in group])
        ratio_data.append(q0 / qpm)
    ax.boxplot(ratio_data, tick_labels=[r"$W_d/t=0.25$", r"$W_d/t=0.5$"], showfliers=True)
    ax.axhline(1, color="black", linestyle=":")
    ax.set_yscale("log"); ax.set_ylabel(r"$G_{Q=0}/\overline{G}_{Q=\pm1}$")
    ax.set_title("(e) same-disorder Q=0 suppression")
    ax.grid(axis="y", alpha=.25)

    # (f) Local Hall signal follows the signed windowed topological charge.
    ax = axes[1, 2]
    hall_rows = hall_position_rows()
    x = np.asarray([int(row["probe_start"]) + .5 for row in hall_rows])
    hall = np.asarray([float(row["Rxy_h_over_e2"]) for row in hall_rows])
    qwin = np.asarray([float(row["windowed_topological_charge"]) for row in hall_rows])
    scale = np.dot(qwin, hall) / np.dot(qwin, qwin)
    ax.plot(x, hall, "o-", label=r"$R_{xy}$")
    ax.plot(x, scale * qwin, "s--", label=r"scaled $Q_{window}$")
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set_xlabel("probe-window center x"); ax.set_ylabel(r"$R_{xy}$ ($h/e^2$)")
    ax.set_title("(f) local Hall compensation")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    fig.savefig(out / "paper_main_figure_draft.png", dpi=260)
    fig.savefig(out / "paper_main_figure_draft.pdf")
    plt.close(fig)
    print(json.dumps({"joined_samples": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
