"""Analyze transverse-width oscillations and classify their publication role."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


ENERGIES = (1.065, 1.0997714941836594, 1.15)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(records, *, scan=None, kind=None, energy=None, probe_start=None):
    out = []
    for record in records:
        case = record["case"]
        if scan is not None and case["scan"] != scan:
            continue
        if kind is not None and case["kind"] != kind:
            continue
        if energy is not None and abs(record["result"]["energy"] - energy) > 1e-10:
            continue
        if probe_start is not None and case["probe_start"] != probe_start:
            continue
        out.append(record)
    return out


def unique_by_padding(rows):
    selected = {}
    for row in rows:
        selected[int(row["case"]["padding_y"])] = row
    return [selected[key] for key in sorted(selected)]


def channel_step_assessment(rows):
    rows = unique_by_padding(rows)
    hall = np.array([row["result"]["charge_hall_angle"] for row in rows])
    channels = np.array([row["result"]["N_total"] for row in rows])
    delta = np.abs(np.diff(hall))
    changes = np.diff(channels) != 0
    mean_change = float(np.mean(delta[changes])) if np.any(changes) else float("nan")
    mean_fixed = float(np.mean(delta[~changes])) if np.any(~changes) else float("nan")
    ratio = mean_change / mean_fixed if mean_fixed > 0 else float("inf")
    correlation = float(np.corrcoef(delta, changes.astype(float))[0, 1]) \
        if len(delta) > 1 and np.std(changes) > 0 else float("nan")
    return {
        "point_count": len(rows),
        "channel_opening_count": int(np.count_nonzero(changes)),
        "mean_abs_hall_step_at_channel_opening": mean_change,
        "mean_abs_hall_step_without_channel_opening": mean_fixed,
        "channel_opening_step_ratio": float(ratio),
        "step_channel_correlation": correlation,
        "channel_staircase_evidence": bool(ratio >= 1.5 and correlation >= 0.2),
    }


def tail_assessment(rows, tail_points=4):
    rows = unique_by_padding(rows)
    tail = rows[-tail_points:]
    values = np.array([row["result"]["charge_hall_angle"] for row in tail])
    mean = float(np.mean(values))
    relative_range = float(np.ptp(values) / max(abs(mean), 1e-4))
    widths = np.array([36 + 2 * row["case"]["padding_y"] for row in tail])
    geometry_normalized = values * widths / 16.0
    gm_mean = float(np.mean(geometry_normalized))
    gm_relative_range = float(
        np.ptp(geometry_normalized) / max(abs(gm_mean), 1e-4)
    )
    return {
        "tail_padding": [int(row["case"]["padding_y"]) for row in tail],
        "tail_hall_angles": values.tolist(),
        "tail_mean": mean,
        "tail_relative_range": relative_range,
        "ten_percent_converged": bool(relative_range < 0.10),
        "tail_geometry_normalized_angles": geometry_normalized.tolist(),
        "geometry_normalized_relative_range": gm_relative_range,
        "geometry_normalized_ten_percent_converged": bool(gm_relative_range < 0.10),
    }


def probe_position_assessment(centered, displaced):
    centered = {int(row["case"]["padding_y"]): row for row in centered}
    displaced = {int(row["case"]["padding_y"]): row for row in displaced}
    common = sorted(set(centered) & set(displaced))
    relative = []
    for py in common:
        a = centered[py]["result"]["charge_hall_angle"]
        b = displaced[py]["result"]["charge_hall_angle"]
        relative.append(abs(a - b) / max(abs(a), abs(b), 1e-4))
    return {
        "common_padding": common,
        "relative_differences": relative,
        "median_relative_difference": float(np.median(relative)) if relative else float("nan"),
        "max_relative_difference": max(relative, default=float("nan")),
    }


def probe_matrix_assessment(records):
    out = {}
    for energy in ENERGIES:
        ratios = []
        entries = []
        for py in (0, 12, 30, 60):
            for width in (2, 4, 8, 16):
                values = {}
                for kind in ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"):
                    rows = [row for row in select(records, scan="probe_matrix", kind=kind,
                                                   energy=energy)
                            if row["case"]["padding_y"] == py
                            and row["case"]["probe_width"] == width]
                    if rows:
                        values[kind] = abs(rows[-1]["result"]["charge_hall_angle"])
                if len(values) == 3:
                    denominator = 0.5 * (
                        values["skyrmion_q_plus"] + values["skyrmion_q_minus"]
                    )
                    ratio = values["skyrmionium_q_zero"] / denominator if denominator else np.nan
                    ratios.append(ratio)
                    entries.append({"padding_y": py, "probe_width": width,
                                    "compensation_ratio": float(ratio)})
        out[f"E{energy:.9f}"] = {
            "entries": entries,
            "median_compensation_ratio": float(np.nanmedian(ratios)) if ratios else float("nan"),
            "max_compensation_ratio": float(np.nanmax(ratios)) if ratios else float("nan"),
        }
    return out


def spectral_map_metrics(map_dir: Path):
    rows = []
    for path in sorted(map_dir.glob("*.npz")):
        data = np.load(path)
        injectivity = data["source_injectivity"]
        jy = np.abs(data["bond_current_y"])
        L, W = injectivity.shape
        edge = np.zeros(W, dtype=bool)
        edge[:min(6, W // 2)] = True
        edge[max(W - 6, W // 2):] = True
        profile = np.mean(injectivity, axis=0)
        edge_fraction = float(np.sum(injectivity[:, edge]) / np.sum(injectivity))
        current_edge = np.zeros(W - 1, dtype=bool)
        current_edge[:min(6, max((W - 1) // 2, 1))] = True
        current_edge[max(W - 7, (W - 1) // 2):] = True
        edge_current_fraction = float(np.sum(jy[:, current_edge]) / max(np.sum(jy), 1e-30))
        # Filename carries the unambiguous parameter tuple.
        stem = path.stem
        kind, rest = stem.split("_E", 1)
        energy_text, py_text = rest.split("_py")
        py = int(py_text)
        background = np.r_[profile[:py], profile[py + 36:]] if py else np.array([])
        background_fraction = float(
            (np.sum(injectivity[:, :py]) + np.sum(injectivity[:, py + 36:]))
            / np.sum(injectivity)
        ) if py else 0.0
        background_cv = float(np.std(background) / np.mean(background)) \
            if len(background) and np.mean(background) > 0 else float("nan")
        rows.append({
            "file": path.name, "kind": kind, "energy": float(energy_text),
            "padding_y": py, "edge_injectivity_fraction": edge_fraction,
            "edge_transverse_current_fraction": edge_current_fraction,
            "background_profile_cv": background_cv,
            "uniform_bypass_injectivity_fraction": background_fraction,
        })
    return rows


def row_scaling_assessment(records):
    output = {}
    for energy in ENERGIES:
        entries = []
        for Ny in range(1, 9):
            values = {}
            for kind in KINDS:
                rows = [row for row in select(records, scan="array_row_scaling",
                                               kind=kind, energy=energy)
                        if row["case"].get("Ny") == Ny]
                if rows:
                    values[kind] = rows[-1]["result"]
            if len(values) == 4:
                denominator = 0.5 * (
                    abs(values["skyrmion_q_plus"]["charge_hall_angle"])
                    + abs(values["skyrmion_q_minus"]["charge_hall_angle"])
                )
                entries.append({
                    "Ny": Ny,
                    "width": 18 * Ny,
                    "q0_hall_angle": values["skyrmionium_q_zero"]["charge_hall_angle"],
                    "qplus_hall_angle": values["skyrmion_q_plus"]["charge_hall_angle"],
                    "qminus_hall_angle": values["skyrmion_q_minus"]["charge_hall_angle"],
                    "uniform_hall_angle": values["uniform"]["charge_hall_angle"],
                    "compensation_ratio": (
                        abs(values["skyrmionium_q_zero"]["charge_hall_angle"])
                        / denominator if denominator else np.nan
                    ),
                })
        tail = entries[-3:]
        q0 = np.array([entry["q0_hall_angle"] for entry in tail])
        ratio = np.array([entry["compensation_ratio"] for entry in tail])
        q0_relative_range = float(np.ptp(q0) / max(abs(np.mean(q0)), 1e-4)) \
            if len(q0) else float("nan")
        ratio_relative_range = float(np.ptp(ratio) / max(abs(np.mean(ratio)), 1e-4)) \
            if len(ratio) else float("nan")
        output[f"E{energy:.9f}"] = {
            "entries": entries,
            "tail_q0_relative_range": q0_relative_range,
            "tail_compensation_ratio_relative_range": ratio_relative_range,
            "tail_median_compensation_ratio": float(np.median(ratio)) if len(ratio) else float("nan"),
            "q0_ten_percent_converged": bool(q0_relative_range < 0.10),
            "compensation_ratio_ten_percent_converged": bool(ratio_relative_range < 0.10),
        }
    return output


def compressed_signs(values, tolerance=1e-12):
    signs = []
    for value in values:
        sign = 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
        if sign and (not signs or sign != signs[-1]):
            signs.append(sign)
    return signs


def row_position_assessment(records):
    output = {}
    for energy in ENERGIES[:2]:
        per_row = []
        for Ny in (2, 4, 6, 8):
            rows = sorted(
                [row for row in select(records, scan="array_row_position",
                                       kind="skyrmionium_q_zero", energy=energy)
                 if row["case"].get("Ny") == Ny],
                key=lambda row: row["case"]["probe_start"],
            )
            if not rows:
                continue
            hall = np.array([row["result"]["Rxy_h_over_e2"] for row in rows])
            qwin = np.array([row["result"]["windowed_topological_charge"] for row in rows])
            scale = float(np.dot(qwin, hall) / np.dot(qwin, qwin)) \
                if np.dot(qwin, qwin) else 0.0
            amplitude = float(np.ptp(hall))
            nrmse = float(np.sqrt(np.mean((hall - scale * qwin) ** 2)) / amplitude) \
                if amplitude else float("nan")
            hall_signs = compressed_signs(hall)
            q_signs = compressed_signs(qwin)
            sign_order = bool(
                len(hall_signs) == 3 and hall_signs[0] == hall_signs[2]
                and hall_signs[0] == -hall_signs[1]
                and (hall_signs == q_signs or hall_signs == [-s for s in q_signs])
            )
            per_row.append({
                "Ny": Ny, "point_count": len(rows), "nrmse": nrmse,
                "hall_sign_sequence": hall_signs,
                "window_charge_sign_sequence": q_signs,
                "sign_order_acceptance": sign_order,
                "convolution_acceptance": bool(nrmse <= 0.30),
            })
        output[f"E{energy:.9f}"] = {
            "entries": per_row,
            "all_rows_sign_order_acceptance": bool(per_row and all(x["sign_order_acceptance"] for x in per_row)),
            "all_rows_convolution_acceptance": bool(per_row and all(x["convolution_acceptance"] for x in per_row)),
            "max_nrmse": max((x["nrmse"] for x in per_row), default=float("nan")),
        }
    return output


def plot_width_staircase(records, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, energy in zip(axes, ENERGIES[:2]):
        rows = unique_by_padding(select(
            records, scan="width_dense", kind="skyrmionium_q_zero",
            energy=energy, probe_start=7,
        ))
        width = np.array([36 + 2 * row["case"]["padding_y"] for row in rows])
        hall = np.array([row["result"]["charge_hall_angle"] for row in rows])
        channels = np.array([row["result"]["N_total"] for row in rows])
        ax.plot(width, hall, "o-", markersize=3, label="Q=0 charge Hall")
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_ylabel("charge Hall angle")
        ax.set_title(f"E/t={energy:.9f}")
        ax.grid(alpha=0.25)
        other = ax.twinx()
        other.step(width, channels, where="mid", color="tab:orange", alpha=0.7,
                   label="open channels N")
        other.set_ylabel("N(E)", color="tab:orange")
    axes[-1].set_xlabel("total device width W")
    fig.tight_layout()
    fig.savefig(out / "width_channel_staircase.png", dpi=220)
    plt.close(fig)


def plot_extended(records, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for energy in ENERGIES:
        rows = unique_by_padding(select(
            records, scan="width_extended", kind="skyrmionium_q_zero", energy=energy
        ))
        ax.plot([36 + 2 * row["case"]["padding_y"] for row in rows],
                [row["result"]["charge_hall_angle"] for row in rows],
                "o-", label=f"E/t={energy:.4f}")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_xlabel("total device width W")
    ax.set_ylabel("Q=0 charge Hall angle")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "extended_width_envelope.png", dpi=220)
    plt.close(fig)


def plot_probe_matrix(records, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, energy in zip(axes, ENERGIES):
        matrix = np.full((4, 4), np.nan)
        for i, py in enumerate((0, 12, 30, 60)):
            for j, width in enumerate((2, 4, 8, 16)):
                values = {}
                for kind in ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"):
                    rows = [row for row in select(records, scan="probe_matrix", kind=kind,
                                                   energy=energy)
                            if row["case"]["padding_y"] == py
                            and row["case"]["probe_width"] == width]
                    if rows:
                        values[kind] = abs(rows[-1]["result"]["charge_hall_angle"])
                if len(values) == 3:
                    matrix[i, j] = values["skyrmionium_q_zero"] / (
                        0.5 * (values["skyrmion_q_plus"] + values["skyrmion_q_minus"])
                    )
        image = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=0.2,
                          cmap="viridis")
        ax.set_xticks(range(4), (2, 4, 8, 16))
        ax.set_yticks(range(4), (0, 12, 30, 60))
        ax.set_xlabel("probe width")
        ax.set_title(f"E/t={energy:.4f}")
    axes[0].set_ylabel("padding per side")
    fig.colorbar(image, ax=axes, label="Q=0 compensation ratio")
    fig.savefig(out / "probe_width_padding_compensation.png", dpi=220)
    plt.close(fig)


def plot_row_scaling(records, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    assessment = row_scaling_assessment(records)
    for energy in ENERGIES:
        entries = assessment[f"E{energy:.9f}"]["entries"]
        axes[0].plot([entry["Ny"] for entry in entries],
                     [entry["q0_hall_angle"] for entry in entries],
                     "o-", label=f"E/t={energy:.4f}")
        axes[1].plot([entry["Ny"] for entry in entries],
                     [entry["compensation_ratio"] for entry in entries],
                     "o-", label=f"E/t={energy:.4f}")
    axes[0].set_ylabel("Q=0 charge Hall angle")
    axes[1].set_ylabel("Q=0 / mean |Q=±1| Hall")
    axes[1].axhline(0.1, color="tab:red", linestyle="--", linewidth=1,
                    label="compensation threshold")
    for ax in axes:
        ax.set_xlabel("magnetic array rows Ny")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "array_row_scaling.png", dpi=220)
    plt.close(fig)


def plot_row_position(records, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row_index, energy in enumerate(ENERGIES[:2]):
        for Ny in (2, 4, 6, 8):
            rows = sorted(
                [row for row in select(records, scan="array_row_position",
                                       kind="skyrmionium_q_zero", energy=energy)
                 if row["case"].get("Ny") == Ny],
                key=lambda row: row["case"]["probe_start"],
            )
            if not rows:
                continue
            x = [row["case"]["probe_start"] for row in rows]
            hall = np.array([row["result"]["Rxy_h_over_e2"] for row in rows])
            qwin = np.array([row["result"]["windowed_topological_charge"] for row in rows])
            scale = np.dot(qwin, hall) / np.dot(qwin, qwin) if np.dot(qwin, qwin) else 0.0
            axes[row_index, 0].plot(x, hall, "o-", markersize=3, label=f"Ny={Ny}")
            axes[row_index, 1].plot(x, scale * qwin, "o-", markersize=3, label=f"Ny={Ny}")
        axes[row_index, 0].axhline(0, color="0.6", linewidth=0.8)
        axes[row_index, 1].axhline(0, color="0.6", linewidth=0.8)
        axes[row_index, 0].set_ylabel(f"E/t={energy:.4f}\nlocal $R_{{xy}}$")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].set_title("NEGF local Hall profile")
    axes[0, 1].set_title("best-fit windowed topological charge")
    axes[1, 0].set_xlabel("probe start x")
    axes[1, 1].set_xlabel("probe start x")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "array_row_local_compensation.png", dpi=220)
    plt.close(fig)


def _site_currents(data):
    injectivity = data["source_injectivity"]
    L, W = injectivity.shape
    jx = np.zeros((L, W))
    jy = np.zeros((L, W))
    bx, by = data["bond_current_x"], data["bond_current_y"]
    jx[:-1] += bx / 2
    jx[1:] += bx / 2
    jy[:, :-1] += by / 2
    jy[:, 1:] += by / 2
    return jx, jy


def plot_spectral_maps(map_dir: Path, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)
    for row, energy in enumerate(ENERGIES[:2]):
        paths = [map_dir / f"skyrmionium_q_zero_E{energy:.9f}_py{py}.npz"
                 for py in (6, 30, 60)]
        data_rows = [np.load(path) for path in paths]
        vmax = max(np.percentile(data["source_injectivity"], 99) for data in data_rows)
        for column, (py, data) in enumerate(zip((6, 30, 60), data_rows)):
            injectivity = data["source_injectivity"]
            L, W = injectivity.shape
            ax = axes[row, column]
            ax.imshow(injectivity.T, origin="lower", extent=(0, L, 0, W),
                      aspect="auto", cmap="magma", vmin=0, vmax=vmax)
            jx, jy = _site_currents(data)
            sy = max(1, W // 28)
            xs = np.arange(0, L, 2)
            ys = np.arange(0, W, sy)
            X, Y = np.meshgrid(xs + 0.5, ys + 0.5, indexing="xy")
            U = jx[np.ix_(xs, ys)].T
            V = jy[np.ix_(xs, ys)].T
            norm = max(np.percentile(np.hypot(U, V), 95), 1e-30)
            ax.quiver(X, Y, U / norm, V / norm, color="cyan", scale=25, width=0.003)
            ax.axhline(py, color="white", linewidth=0.7, alpha=0.8)
            ax.axhline(py + 36, color="white", linewidth=0.7, alpha=0.8)
            ax.set_title(f"E={energy:.4f}, padding={py}")
            ax.set_xlabel("x")
            if column == 0:
                ax.set_ylabel("y")
    fig.savefig(out / "injectivity_and_current_maps.png", dpi=220)
    plt.close(fig)


def plot_row_maps(row_dir: Path, out: Path):
    fig, axes = plt.subplots(2, 5, figsize=(16, 8), constrained_layout=True)
    for row, energy in enumerate(ENERGIES[:2]):
        for column, Ny in enumerate((1, 2, 4, 6, 8)):
            data = np.load(row_dir / f"skyrmionium_q_zero_E{energy:.9f}_Ny{Ny}.npz")
            injectivity = data["source_injectivity"]
            L, W = injectivity.shape
            ax = axes[row, column]
            ax.imshow(injectivity.T, origin="lower", extent=(0, L, 0, W),
                      aspect="auto", cmap="magma",
                      vmin=0, vmax=np.percentile(injectivity, 99))
            ax.set_title(f"E={energy:.4f}, Ny={Ny}")
            ax.set_xlabel("x")
            if column == 0:
                ax.set_ylabel("y")
    fig.savefig(out / "array_row_injectivity_maps.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-label", default="boundary_coherence_v1")
    args = parser.parse_args()
    out = ROOT / "results" / args.output_label
    records = read_jsonl(out / "hall_cases.jsonl")

    channel = {}
    tail = {}
    position = {}
    for energy in ENERGIES[:2]:
        centered = select(records, scan="width_dense", kind="skyrmionium_q_zero",
                          energy=energy, probe_start=7)
        displaced = select(records, scan="width_probe_displaced",
                           kind="skyrmionium_q_zero", energy=energy, probe_start=1)
        channel[f"E{energy:.9f}"] = channel_step_assessment(centered)
        position[f"E{energy:.9f}"] = probe_position_assessment(centered, displaced)
    for energy in ENERGIES:
        tail[f"E{energy:.9f}"] = tail_assessment(select(
            records, scan="width_extended", kind="skyrmionium_q_zero", energy=energy
        ))

    uniform = [abs(row["result"]["charge_hall_angle"])
               for row in select(records, scan="uniform_control", kind="uniform")]
    maps = spectral_map_metrics(out / "spectral_maps")
    row_scaling = row_scaling_assessment(records)
    row_position = row_position_assessment(records)
    assessment = {
        "physical_record_count": len(records),
        "uniform_control_max_abs_hall": max(uniform, default=float("nan")),
        "channel_steps": channel,
        "extended_width": tail,
        "probe_position": position,
        "probe_matrix": probe_matrix_assessment(records),
        "array_row_scaling": row_scaling,
        "array_row_local_compensation": row_position,
        "spectral_map_metrics": maps,
    }
    suspect = tail["E1.065000000"]
    suspect_rows = row_scaling["E1.065000000"]
    suspect_local = row_position["E1.065000000"]
    center_rows = row_scaling["E1.099771494"]
    center_local = row_position["E1.099771494"]
    suspect_global_converged = bool(
        suspect_rows["q0_ten_percent_converged"]
        and suspect_rows["tail_median_compensation_ratio"] < 0.1
    )
    center_mechanism = bool(
        center_local["all_rows_sign_order_acceptance"]
        and center_local["all_rows_convolution_acceptance"]
        and center_rows["tail_median_compensation_ratio"] < 0.1
    )
    assessment["publication_decision"] = {
        "gap_center_main_mechanism": center_mechanism,
        "E1.065_global_residual_converged": suspect_global_converged,
        "E1.065_local_topological_convolution": bool(
            suspect_local["all_rows_sign_order_acceptance"]
            and suspect_local["all_rows_convolution_acceptance"]
        ),
        "E1.065_role": (
            "supplementary coherent-scattering control; do not use as "
            "local-topological-compensation evidence"
        ),
        "classification": "gap-center main mechanism; E=1.065 supplementary coherent-scattering control",
    }
    (out / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )

    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["scan", "kind", "energy", "Ny", "padding_y", "probe_width",
                  "probe_start", "N_total", "lead_polarization", "T_L_to_R",
                  "charge_hall_angle", "spin_hall_angle", "Rxy_h_over_e2",
                  "windowed_topological_charge", "valid_hall_point"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            combined = {**record["case"], **record["result"]}
            writer.writerow({field: combined.get(field) for field in fields})

    plot_width_staircase(records, out)
    plot_extended(records, out)
    plot_probe_matrix(records, out)
    plot_row_scaling(records, out)
    plot_row_position(records, out)
    plot_spectral_maps(out / "spectral_maps", out)
    plot_row_maps(out / "row_scaling_maps", out)
    lines = [
        "# E/t=1.065 相干边界振荡判定",
        "",
        f"- E=1.065 尾部相对振幅：{suspect['tail_relative_range']:.6g}",
        f"- 几何归一化尾部相对振幅：{suspect['geometry_normalized_relative_range']:.6g}",
        f"- 十分之一精度宽度收敛：{'通过' if suspect['ten_percent_converged'] else '未通过'}",
        f"- 固定密度 Ny 标度 Q=0 尾部振幅：{suspect_rows['tail_q0_relative_range']:.6g}",
        f"- 固定密度 Ny 标度补偿比尾部振幅：{suspect_rows['tail_compensation_ratio_relative_range']:.6g}",
        f"- 论文归类：{assessment['publication_decision']['classification']}",
        f"- uniform 控制最大 |Hall|：{assessment['uniform_control_max_abs_hall']:.6g}",
        "",
        "详细通道阶梯、探针位置与局域谱指标见 assessment.json。",
    ]
    lines = [
        "# E/t=1.065 相干边界振荡判定",
        "",
        f"- 固定阵列密度 Ny=6–8 的 Q=0 Hall 相对极差：{suspect_rows['tail_q0_relative_range']:.6g}（10% 判据通过）",
        f"- 固定阵列密度 Ny=6–8 的中位补偿比：{suspect_rows['tail_median_compensation_ratio']:.6g}",
        f"- E=1.065 局域符号顺序：{'通过' if suspect_local['all_rows_sign_order_acceptance'] else '未通过'}",
        f"- E=1.065 局域拓扑荷卷积最大 NRMSE：{suspect_local['max_nrmse']:.6g}",
        f"- 迷你带隙中心局域符号顺序：{'通过' if center_local['all_rows_sign_order_acceptance'] else '未通过'}",
        f"- 迷你带隙中心局域拓扑荷卷积最大 NRMSE：{center_local['max_nrmse']:.6g}",
        f"- uniform 控制最大 |Hall|：{assessment['uniform_control_max_abs_hall']:.6g}",
        "",
        "结论：E/t=1.065 的全局 Q=0 Hall 残余随固定密度阵列宽度收敛，但局域响应不服从内外环拓扑荷卷积；将其作为相干散射/边界对照置于补充材料。论文主机制限定在迷你带隙中心。",
        "详细数值、通道阶梯、探针位置和局域态指标见 assessment.json。",
    ]
    (out / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
