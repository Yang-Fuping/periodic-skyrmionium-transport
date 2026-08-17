"""Assess clean geometry convergence of the Nx=8 upper-edge Hall window."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def group_observables(rows, fields):
    grouped = {}
    for row in rows:
        case = row["case"]
        key = tuple(case[field] for field in fields)
        grouped.setdefault(key, {})[case["kind"]] = row["result"]
    output = []
    for key, values in sorted(grouped.items()):
        if len(values) != len(KINDS):
            continue
        denominator = 0.5 * (
            abs(values["skyrmion_q_plus"]["charge_hall_angle"])
            + abs(values["skyrmion_q_minus"]["charge_hall_angle"])
        )
        item = dict(zip(fields, key))
        item.update({
            "all_texture_points_valid": all(values[k]["valid_hall_point"] for k in KINDS[1:]),
            "q0_valid": values["skyrmionium_q_zero"]["valid_hall_point"],
            "q0_source_current_fraction": values["skyrmionium_q_zero"]["source_current_fraction"],
            "q0_hall_angle": values["skyrmionium_q_zero"]["charge_hall_angle"],
            "qplus_hall_angle": values["skyrmion_q_plus"]["charge_hall_angle"],
            "qminus_hall_angle": values["skyrmion_q_minus"]["charge_hall_angle"],
            "compensation_ratio": abs(values["skyrmionium_q_zero"]["charge_hall_angle"]) / denominator if denominator else np.nan,
            "uniform_abs_hall": abs(values["uniform"]["charge_hall_angle"]),
            "q_reversal_error": abs(values["skyrmion_q_plus"]["charge_hall_angle"] + values["skyrmion_q_minus"]["charge_hall_angle"]),
        })
        item["passes"] = bool(item["all_texture_points_valid"] and item["compensation_ratio"] < 0.1)
        output.append(item)
    return output


def contiguous_windows(energies, step=0.0005):
    groups = []
    for energy in sorted(energies):
        if not groups or energy - groups[-1][-1] > step * 1.01:
            groups.append([energy])
        else:
            groups[-1].append(energy)
    return [{"low": group[0], "high": group[-1], "width": group[-1] - group[0],
             "point_count": len(group)} for group in groups]


def energy_assessment(entries):
    by_width = {}
    for width in (4, 16):
        rows = [row for row in entries if row["probe_width"] == width]
        by_width[str(width)] = {
            "entries": rows,
            "passing_windows": contiguous_windows([row["energy"] for row in rows if row["passes"]]),
        }
    common = []
    for energy in sorted({row["energy"] for row in entries}):
        rows = [row for row in entries if abs(row["energy"] - energy) < 1e-10]
        common.append({
            "energy": energy, "all_widths_pass": bool(len(rows) == 2 and all(row["passes"] for row in rows)),
            "max_compensation_ratio": max((row["compensation_ratio"] for row in rows), default=np.nan),
            "min_current_fraction": min((row["q0_source_current_fraction"] for row in rows), default=np.nan),
        })
    return {
        "by_probe_width": by_width,
        "common_entries": common,
        "common_windows": contiguous_windows([row["energy"] for row in common if row["all_widths_pass"]]),
    }


def geometry_summary(entries, varied_field):
    output = []
    for energy in sorted({row["energy"] for row in entries}):
        rows = [row for row in entries if row["energy"] == energy]
        output.append({
            "energy": energy, "case_count": len(rows),
            "passing_count": sum(row["passes"] for row in rows),
            "all_cases_pass": bool(rows and all(row["passes"] for row in rows)),
            "passing_values": sorted({row[varied_field] for row in rows if row["passes"]}),
            "max_compensation_ratio": max((row["compensation_ratio"] for row in rows), default=np.nan),
            "min_current_fraction": min((row["q0_source_current_fraction"] for row in rows), default=np.nan),
        })
    return output


def ny_refinement_assessment(entries):
    rows = []
    for energy in sorted({row["energy"] for row in entries}):
        selected = [row for row in entries if abs(row["energy"] - energy) < 1e-10]
        rows.append({
            "energy": energy, "case_count": len(selected),
            "passing_count": sum(row["passes"] for row in selected),
            "all_ny_width_cases_pass": bool(len(selected) == 8 and all(row["passes"] for row in selected)),
            "max_compensation_ratio": max((row["compensation_ratio"] for row in selected), default=np.nan),
            "min_current_fraction": min((row["q0_source_current_fraction"] for row in selected), default=np.nan),
        })
    return {
        "entries": rows,
        "common_windows": contiguous_windows(
            [row["energy"] for row in rows if row["all_ny_width_cases_pass"]]
        ),
    }


def quality(rows):
    results = [row["result"] for row in rows]
    return {
        "max_current_conservation_error": max(r["current_conservation_error"] for r in results),
        "max_probe_current_error": max(r["probe_current_error"] for r in results),
        "max_gauge_error": max(r["gauge_invariance_error"] for r in results),
        "max_unitarity_error": max(r["scattering_unitarity_error"] for r in results),
        "max_scattering_charge_mismatch": max(r["scattering_charge_mismatch"] for r in results),
        "all_transmission_bounds_ok": all(r["transmission_bound_ok"] for r in results),
    }


def plot_energy(entries, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for width, style in ((4, "o-"), (16, "s--")):
        rows = [row for row in entries if row["probe_width"] == width]
        axes[0].semilogy([row["energy"] for row in rows],
                         [row["q0_source_current_fraction"] for row in rows], style,
                         label=f"probe width {width}")
        axes[1].semilogy([row["energy"] for row in rows],
                         [row["compensation_ratio"] for row in rows], style,
                         label=f"probe width {width}")
    axes[0].axhline(1e-6, color="tab:red", linestyle=":")
    axes[1].axhline(0.1, color="tab:red", linestyle=":")
    axes[0].set(xlabel="Energy E/t", ylabel="$|I_L|/N$", title="current validity")
    axes[1].set(xlabel="Energy E/t", ylabel="Q=0 / mean |Q=±1| Hall",
                title="half-step compensation window")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "half_step_window.png", dpi=240)
    plt.close(fig)


def plot_geometry(ny_entries, position_entries, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for energy in sorted({row["energy"] for row in ny_entries}):
        rows = [row for row in ny_entries if row["energy"] == energy and row["probe_width"] == 4]
        axes[0].semilogy([row["Ny"] for row in rows], [row["compensation_ratio"] for row in rows],
                         "o-", label=f"E/t={energy:.4f}, w=4")
    axes[0].axhline(0.1, color="tab:red", linestyle=":")
    axes[0].set(xlabel="$N_y$", ylabel="compensation ratio", title="transverse-row dependence")
    for energy in sorted({row["energy"] for row in position_entries}):
        rows = [row for row in position_entries if row["energy"] == energy]
        axes[1].semilogy([row["start_in_cell"] for row in rows],
                         [row["compensation_ratio"] for row in rows], "o-",
                         label=f"E/t={energy:.4f}")
    axes[1].axhline(0.1, color="tab:red", linestyle=":")
    axes[1].set(xlabel="probe start within central cell", ylabel="compensation ratio",
                title="probe-position dependence")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "ny_probe_position_robustness.png", dpi=240)
    plt.close(fig)


def plot_ny_refinement(entries, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ny in (1, 2, 3, 4):
        for width, style in ((4, "-"), (16, "--")):
            rows = [row for row in entries if row["Ny"] == ny and row["probe_width"] == width]
            axes[0].semilogy([row["energy"] for row in rows],
                             [row["compensation_ratio"] for row in rows],
                             marker="o", linestyle=style, markersize=3,
                             label=f"Ny={ny}, w={width}")
            axes[1].semilogy([row["energy"] for row in rows],
                             [row["q0_source_current_fraction"] for row in rows],
                             marker="o", linestyle=style, markersize=3,
                             label=f"Ny={ny}, w={width}")
    axes[0].axhline(0.1, color="tab:red", linestyle=":")
    axes[1].axhline(1e-6, color="tab:red", linestyle=":")
    axes[0].set(xlabel="Energy E/t", ylabel="compensation ratio",
                title="Ny and probe-width compensation")
    axes[1].set(xlabel="Energy E/t", ylabel="$|I_L|/N$", title="current validity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(out / "ny_refined_common_window.png", dpi=240)
    plt.close(fig)


def main():
    out = ROOT / "results" / "upper_edge_robustness_v1"
    rows = read_jsonl(out / "clean_cases.jsonl")
    energy = group_observables([row for row in rows if row["case"]["scan"] == "energy_half_step"],
                               ("energy", "probe_width"))
    ny = group_observables([row for row in rows if row["case"]["scan"] == "ny_scaling"],
                           ("energy", "Ny", "probe_width"))
    position = group_observables([row for row in rows if row["case"]["scan"] == "probe_position"],
                                 ("energy", "start_in_cell", "probe_width"))
    ny_refinement = group_observables(
        [row for row in rows if row["case"]["scan"] == "ny_refinement"],
        ("energy", "Ny", "probe_width"),
    )
    assessment = {
        "physical_record_count": len(rows),
        "energy_half_step": energy_assessment(energy),
        "ny_scaling_entries": ny,
        "ny_scaling_summary": geometry_summary(ny, "Ny"),
        "probe_position_entries": position,
        "probe_position_summary": geometry_summary(position, "start_in_cell"),
        "ny_refinement": ny_refinement_assessment(ny_refinement),
        "quality": quality(rows),
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    plot_energy(energy, out)
    plot_geometry(ny, position, out)
    plot_ny_refinement(ny_refinement, out)
    print(json.dumps({
        "physical_record_count": len(rows),
        "common_windows": assessment["energy_half_step"]["common_windows"],
        "ny_scaling_summary": assessment["ny_scaling_summary"],
        "probe_position_summary": assessment["probe_position_summary"],
        "ny_refinement_common_windows": assessment["ny_refinement"]["common_windows"],
        "quality": assessment["quality"],
    }, indent=2))


if __name__ == "__main__":
    main()
