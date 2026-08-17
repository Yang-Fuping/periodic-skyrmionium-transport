"""Analyze production Hall scans and create publication-audit figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LABELS = {
    "uniform": "uniform",
    "skyrmionium_q_zero": "Q=0",
    "skyrmion_q_plus": "Q=+1",
    "skyrmion_q_minus": "Q=-1",
}
COLORS = {
    "uniform": "black",
    "skyrmionium_q_zero": "tab:blue",
    "skyrmion_q_plus": "tab:red",
    "skyrmion_q_minus": "tab:green",
}
E_CENTER = 1.0997714941836594


def read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return abs(a - b) <= tolerance


def rows_for(records, *, scan=None, kind=None, J=None, energy=None, **case_filters):
    selected = []
    for record in records:
        case, result = record["case"], record["result"]
        if scan is not None and case["scan"] != scan:
            continue
        if kind is not None and case["kind"] != kind:
            continue
        if J is not None and not close(float(case["J"]), J):
            continue
        if energy is not None and not close(float(result["energy"]), energy):
            continue
        if any(case.get(key) != value for key, value in case_filters.items()):
            continue
        selected.append(record)
    return selected


def compressed_signs(values: list[float], tolerance=1e-12) -> list[int]:
    signs = []
    for value in values:
        sign = 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
        if sign and (not signs or sign != signs[-1]):
            signs.append(sign)
    return signs


def candidate_windows(rows: list[dict]) -> list[tuple[float, float]]:
    ordered = sorted(rows, key=lambda row: row["result"]["energy"])
    all_energies = np.array([row["result"]["energy"] for row in ordered])
    step = float(np.median(np.diff(all_energies))) if len(all_energies) > 1 else 0.0
    selected = []
    for row in ordered:
        result = row["result"]
        charge = abs(result["charge_hall_angle"])
        spin = abs(result["spin_hall_angle"])
        if (result["valid_hall_point"] and spin >= 0.01
                and charge <= min(0.005, 0.1 * spin)):
            selected.append(float(result["energy"]))
    groups: list[list[float]] = []
    for energy in selected:
        if groups and step and energy - groups[-1][-1] <= 1.5 * step:
            groups[-1].append(energy)
        else:
            groups.append([energy])
    return [(g[0], g[-1]) for g in groups
            if len(g) >= 2 and g[-1] - g[0] >= 0.02 - 1e-12]


def write_csv(records: list[dict], path: Path):
    fields = [
        "scan", "kind", "A", "R", "Nx", "Ny", "J", "probe_width",
        "probe_start", "padding_x", "padding_y", "eta", "energy",
        "T_L_to_R", "N_up", "N_down", "N_total", "lead_polarization",
        "topological_charge", "windowed_topological_charge",
        "Rxx_h_over_e2", "Rxy_h_over_e2", "charge_hall_angle",
        "spin_hall_angle", "source_current_fraction", "valid_hall_point",
        "scattering_unitarity_error", "scattering_charge_mismatch",
        "current_conservation_error", "probe_current_error",
        "gauge_invariance_error", "transmission_bound_ok",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            combined = {**record["case"], **record["result"]}
            writer.writerow({key: combined.get(key) for key in fields})


def numerical_assessment(records: list[dict]) -> dict:
    results = [record["result"] for record in records]
    metrics = {
        "raw_max_current_conservation_error": max(r["current_conservation_error"] for r in results),
        "raw_max_probe_current_error": max(r["probe_current_error"] for r in results),
        "raw_max_gauge_invariance_error": max(r["gauge_invariance_error"] for r in results),
        "raw_max_scattering_unitarity_error": max(r["scattering_unitarity_error"] for r in results),
        "raw_max_scattering_charge_mismatch": max(r["scattering_charge_mismatch"] for r in results),
        "all_transmissions_within_channel_bound": all(r["transmission_bound_ok"] for r in results),
    }
    effective = {}
    for record in records:
        case = record["case"]
        normalized_scan = case["scan"].replace("numerical_validation__", "")
        key = (
            normalized_scan, case["kind"], case["A"], case["R"], case["Nx"],
            case["Ny"], case["J"], case["probe_width"], case["probe_start"],
            case["padding_x"], case["padding_y"], record["result"]["energy"],
        )
        if key not in effective or case["eta"] < effective[key]["case"]["eta"]:
            effective[key] = record
    effective_results = [record["result"] for record in effective.values()]
    metrics.update({
        "effective_point_count": len(effective_results),
        "validation_point_count": sum(
            record["case"]["scan"].startswith("numerical_validation__")
            for record in records
        ),
        "max_current_conservation_error": max(r["current_conservation_error"] for r in effective_results),
        "max_probe_current_error": max(r["probe_current_error"] for r in effective_results),
        "max_gauge_invariance_error": max(r["gauge_invariance_error"] for r in effective_results),
        "max_scattering_unitarity_error": max(r["scattering_unitarity_error"] for r in effective_results),
        "max_scattering_charge_mismatch": max(r["scattering_charge_mismatch"] for r in effective_results),
    })
    metrics["landauer_acceptance"] = bool(
        metrics["max_current_conservation_error"] < 1e-9
        and metrics["max_probe_current_error"] < 1e-9
        and metrics["max_gauge_invariance_error"] < 1e-9
    )
    metrics["scattering_acceptance"] = bool(
        metrics["max_scattering_unitarity_error"] < 5e-6
        and metrics["max_scattering_charge_mismatch"] < 5e-6
    )
    uniform = [record["result"] for record in effective.values()
               if record["case"]["kind"] == "uniform"
               and record["result"]["valid_hall_point"]]
    metrics["uniform_max_abs_charge_hall"] = max(
        (abs(r["charge_hall_angle"]) for r in uniform), default=float("nan")
    )
    metrics["uniform_max_abs_spin_hall"] = max(
        (abs(r["spin_hall_angle"]) for r in uniform), default=float("nan")
    )
    return metrics


def symmetry_assessment(records: list[dict]) -> dict:
    grouped: dict[tuple, dict[str, dict]] = {}
    for record in records:
        case = record["case"]
        if case["kind"] not in {"skyrmion_q_plus", "skyrmion_q_minus"}:
            continue
        key = tuple((name, json.dumps(value, sort_keys=True)) for name, value in case.items()
                    if name not in {"kind", "energies"}) + (("energy", record["result"]["energy"]),)
        grouped.setdefault(key, {})[case["kind"]] = record["result"]
    errors = []
    for pair in grouped.values():
        if len(pair) == 2:
            errors.append(abs(
                pair["skyrmion_q_plus"]["charge_hall_angle"]
                + pair["skyrmion_q_minus"]["charge_hall_angle"]
            ))
    return {
        "paired_points": len(errors),
        "max_q_sign_reversal_error": max(errors, default=float("nan")),
        "q_sign_reversal_acceptance": bool(errors and max(errors) < 1e-8),
    }


def compensation_assessment(records: list[dict]) -> dict:
    position = sorted(
        rows_for(records, scan="position_w2", kind="skyrmionium_q_zero",
                 energy=E_CENTER),
        key=lambda row: row["case"]["probe_start"],
    )
    hall = np.array([row["result"]["Rxy_h_over_e2"] for row in position])
    qwin = np.array([row["result"]["windowed_topological_charge"] for row in position])
    scale = float(np.dot(qwin, hall) / np.dot(qwin, qwin)) if np.dot(qwin, qwin) else 0.0
    amplitude = float(np.ptp(hall))
    nrmse = float(np.sqrt(np.mean((hall - scale * qwin) ** 2)) / amplitude) \
        if len(hall) and amplitude else float("nan")
    hall_signs = compressed_signs(hall.tolist())
    q_signs = compressed_signs(qwin.tolist())
    direct_or_global_flip = hall_signs == q_signs or hall_signs == [-s for s in q_signs]
    sign_order_ok = bool(
        len(hall_signs) == 3 and hall_signs[0] == hall_signs[2]
        and hall_signs[0] == -hall_signs[1]
        and direct_or_global_flip
    )

    ratios = []
    for energy in (1.065, E_CENTER, 1.15):
        values = {}
        for kind in ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"):
            rows = rows_for(records, scan="probe_width", kind=kind, energy=energy,
                            probe_width=16)
            if rows and rows[0]["result"]["valid_hall_point"]:
                values[kind] = abs(rows[0]["result"]["charge_hall_angle"])
        if len(values) == 3:
            denominator = 0.5 * (
                values["skyrmion_q_plus"] + values["skyrmion_q_minus"]
            )
            if denominator:
                ratios.append(values["skyrmionium_q_zero"] / denominator)

    padding_changes = []
    for energy in (1.065, E_CENTER, 1.15):
        values = {}
        for py in (12, 18):
            rows = rows_for(records, scan="transverse_padding",
                            kind="skyrmionium_q_zero", energy=energy, padding_y=py)
            if rows:
                values[py] = rows[0]["result"]["charge_hall_angle"]
        if len(values) == 2:
            padding_changes.append(
                abs(values[12] - values[18]) / max(abs(values[12]), abs(values[18]), 1e-12)
            )
    extended_padding_changes = []
    for energy in (1.065, E_CENTER, 1.15):
        values = {}
        for py in (24, 30):
            rows = rows_for(records, scan="transverse_padding_extended",
                            kind="skyrmionium_q_zero", energy=energy, padding_y=py)
            if rows:
                values[py] = rows[0]["result"]["charge_hall_angle"]
        if len(values) == 2:
            extended_padding_changes.append(
                abs(values[24] - values[30]) / max(abs(values[24]), abs(values[30]), 1e-12)
            )
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    max_padding_change = max(padding_changes, default=float("nan"))
    max_extended_change = max(extended_padding_changes, default=float("nan"))
    return {
        "position_points": len(position),
        "hall_sign_sequence": hall_signs,
        "window_charge_sign_sequence": q_signs,
        "position_sign_order_acceptance": sign_order_ok,
        "window_convolution_scale": scale,
        "window_convolution_nrmse": nrmse,
        "window_convolution_acceptance": bool(nrmse <= 0.30),
        "full_width_compensation_ratios": ratios,
        "full_width_median_compensation_ratio": median_ratio,
        "full_width_compensation_acceptance": bool(median_ratio < 0.1),
        "padding_relative_changes": padding_changes,
        "padding_max_relative_change": max_padding_change,
        "padding_12_18_acceptance": bool(max_padding_change < 0.10),
        "extended_padding_relative_changes": extended_padding_changes,
        "extended_padding_max_relative_change": max_extended_change,
        "padding_acceptance": bool(max_extended_change < 0.10),
    }


def spin_assessment(records: list[dict]) -> dict:
    out = {}
    any_window = False
    for J in (1.5, 3.0):
        for kind in KINDS:
            rows = rows_for(records, scan="mixed_spin_coarse", kind=kind, J=J)
            windows = candidate_windows(rows)
            valid = [row["result"] for row in rows if row["result"]["valid_hall_point"]]
            key = f"J{J:g}_{kind}"
            out[key] = {
                "candidate_windows": windows,
                "max_abs_spin_hall": max((abs(r["spin_hall_angle"]) for r in valid), default=float("nan")),
                "min_abs_charge_hall": min((abs(r["charge_hall_angle"]) for r in valid), default=float("nan")),
            }
            any_window = any_window or bool(windows)
    out["coarse_pure_spin_window_found"] = any_window
    return out


def plot_mixed(records: list[dict], out: Path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    for column, J in enumerate((1.5, 3.0)):
        for kind in KINDS:
            rows = sorted(rows_for(records, scan="mixed_spin_coarse", kind=kind, J=J),
                          key=lambda row: row["result"]["energy"])
            if not rows:
                continue
            energy = [row["result"]["energy"] for row in rows]
            axes[0, column].plot(energy, [row["result"]["charge_hall_angle"] for row in rows],
                                 label=LABELS[kind], color=COLORS[kind])
            axes[1, column].plot(energy, [row["result"]["spin_hall_angle"] for row in rows],
                                 label=LABELS[kind], color=COLORS[kind])
            axes[2, column].plot(energy, [row["result"]["lead_polarization"] for row in rows],
                                 label=LABELS[kind], color=COLORS[kind])
        axes[0, column].set_title(f"J/t={J:g}")
        axes[2, column].set_xlabel("Energy E/t")
    axes[0, 0].set_ylabel("charge Hall angle")
    axes[1, 0].set_ylabel("spin Hall angle")
    axes[2, 0].set_ylabel("lead polarization")
    for ax in axes.flat:
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mixed_spin_hall_spectra.png", dpi=220)
    plt.close(fig)


def plot_position(records: list[dict], out: Path):
    rows = sorted(rows_for(records, scan="position_w2", kind="skyrmionium_q_zero",
                           energy=E_CENTER), key=lambda row: row["case"]["probe_start"])
    if not rows:
        return
    x = np.array([row["case"]["probe_start"] + 0.5 for row in rows])
    hall = np.array([row["result"]["Rxy_h_over_e2"] for row in rows])
    qwin = np.array([row["result"]["windowed_topological_charge"] for row in rows])
    scale = np.dot(qwin, hall) / np.dot(qwin, qwin) if np.dot(qwin, qwin) else 0.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, hall, "o-", label=r"$R_{xy}$")
    ax.plot(x, scale * qwin, "s--", label=r"scaled $Q_{window}$")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_xlabel("probe-window center x")
    ax.set_ylabel(r"$R_{xy}$ ($h/e^2$)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "probe_position_topological_convolution.png", dpi=220)
    plt.close(fig)


def plot_convergence(records: list[dict], out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind in ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"):
        rows = sorted(rows_for(records, scan="probe_width", kind=kind, energy=E_CENTER),
                      key=lambda row: row["case"]["probe_width"])
        axes[0].plot([row["case"]["probe_width"] for row in rows],
                     [row["result"]["charge_hall_angle"] for row in rows],
                     "o-", label=LABELS[kind], color=COLORS[kind])
    rows = (
        rows_for(records, scan="transverse_padding",
                 kind="skyrmionium_q_zero", energy=E_CENTER)
        + rows_for(records, scan="transverse_padding_extended",
                   kind="skyrmionium_q_zero", energy=E_CENTER)
    )
    rows = sorted(rows, key=lambda row: row["case"]["padding_y"])
    axes[1].plot([row["case"]["padding_y"] for row in rows],
                 [row["result"]["charge_hall_angle"] for row in rows], "o-")
    axes[0].set_xlabel("probe width")
    axes[0].set_ylabel("charge Hall angle")
    axes[1].set_xlabel("transverse padding per side")
    axes[1].set_ylabel("Q=0 charge Hall angle")
    axes[1].set_title("gap-center convergence")
    for ax in axes:
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "probe_width_and_padding_convergence.png", dpi=220)
    plt.close(fig)


def plot_walls(records: list[dict], out: Path):
    kinds = ("skyrmionium_inner_wall", "skyrmionium_outer_wall", "skyrmionium_q_zero")
    energies = (1.065, E_CENTER, 1.15)
    values = np.full((len(kinds), len(energies)), np.nan)
    for i, kind in enumerate(kinds):
        for j, energy in enumerate(energies):
            rows = rows_for(records, scan="wall_counterfactual", kind=kind, energy=energy)
            if rows:
                values[i, j] = rows[0]["result"]["charge_hall_angle"]
    x = np.arange(len(energies))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, kind in enumerate(kinds):
        ax.bar(x + (i - 1) * 0.24, values[i], width=0.24,
               label=kind.replace("skyrmionium_", ""))
    ax.set_xticks(x, [f"{energy:.4f}" for energy in energies])
    ax.set_xlabel("Energy E/t")
    ax.set_ylabel("charge Hall angle")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "inner_outer_wall_counterfactual.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-label", default="hall_mechanism_v1")
    args = parser.parse_args()
    out = ROOT / "results" / args.output_label
    records = read_records(out / "raw_cases.jsonl")
    write_csv(records, out / "summary.csv")
    assessment = {
        "record_count": len(records),
        "numerical": numerical_assessment(records),
        "symmetry": symmetry_assessment(records),
        "compensation": compensation_assessment(records),
        "spin_hall": spin_assessment(records),
    }
    (out / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    plot_mixed(records, out)
    plot_position(records, out)
    plot_convergence(records, out)
    plot_walls(records, out)
    numerical = assessment["numerical"]
    symmetry = assessment["symmetry"]
    compensation = assessment["compensation"]
    spin = assessment["spin_hall"]
    lines = [
        "# Hall 补偿与自旋 Hall 数值验收",
        "",
        f"- 原始能点数：{len(records)}",
        f"- Landauer–Büttiker 验收：{'通过' if numerical['landauer_acceptance'] else '未通过'}",
        f"- 完整散射矩阵验收：{'通过' if numerical['scattering_acceptance'] else '未通过'}",
        f"- Q=±1 Hall 反号验收：{'通过' if symmetry['q_sign_reversal_acceptance'] else '未通过'}",
        f"- 局域窗口符号顺序：{'通过' if compensation['position_sign_order_acceptance'] else '未通过'}",
        f"- 拓扑荷窗口卷积 NRMSE：{compensation['window_convolution_nrmse']:.6g}",
        f"- 最大宽度补偿比中位数：{compensation['full_width_median_compensation_ratio']:.6g}",
        f"- padding 12→18 最大相对变化：{compensation['padding_max_relative_change']:.6g}",
        f"- padding 24→30 最大相对变化：{compensation['extended_padding_max_relative_change']:.6g}",
        f"- 三个能点横向边界整体收敛：{'通过' if compensation['padding_acceptance'] else '未通过'}",
        f"- 粗网格纯自旋 Hall 窗口：{'发现候选' if spin['coarse_pure_spin_window_found'] else '未发现'}",
        "",
        "详细数值与各参数点见 assessment.json 和 summary.csv。",
    ]
    (out / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
