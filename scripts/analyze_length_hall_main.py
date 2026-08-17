"""Analyze the longitudinal Hall campaign and produce paper-candidate figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


E_CENTER = 1.0997714941836594
NX_VALUES = (1, 2, 4, 8)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LABELS = {
    "uniform": "uniform",
    "skyrmionium_q_zero": "Q=0 skyrmionium",
    "skyrmion_q_plus": "Q=+1 skyrmion",
    "skyrmion_q_minus": "Q=-1 skyrmion",
}


def read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_for(records, scan, *, nx=None, kind=None, width=None, energy=None):
    rows = []
    for row in records:
        case = row["case"]
        if case["scan"] != scan:
            continue
        if nx is not None and case["Nx"] != nx:
            continue
        if kind is not None and case["kind"] != kind:
            continue
        if width is not None and case["probe_width"] != width:
            continue
        if energy is not None and abs(row["result"]["energy"] - energy) > 1e-10:
            continue
        rows.append(row)
    return rows


def compressed_signs(values, tolerance=1e-12):
    signs = []
    for value in values:
        sign = 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
        if sign and (not signs or sign != signs[-1]):
            signs.append(sign)
    return signs


def baseline_assessment(records):
    entries = []
    for nx in NX_VALUES:
        for width in (4, 16):
            values = {}
            for kind in KINDS:
                rows = rows_for(records, "length_baseline", nx=nx, kind=kind, width=width)
                if rows:
                    values[kind] = rows[-1]["result"]
            if len(values) != len(KINDS):
                continue
            denominator = 0.5 * (
                abs(values["skyrmion_q_plus"]["charge_hall_angle"])
                + abs(values["skyrmion_q_minus"]["charge_hall_angle"])
            )
            entries.append({
                "Nx": nx, "probe_width": width,
                "all_texture_points_valid": all(values[k]["valid_hall_point"] for k in KINDS[1:]),
                "q0_hall_angle": values["skyrmionium_q_zero"]["charge_hall_angle"],
                "qplus_hall_angle": values["skyrmion_q_plus"]["charge_hall_angle"],
                "qminus_hall_angle": values["skyrmion_q_minus"]["charge_hall_angle"],
                "q0_source_current_fraction": values["skyrmionium_q_zero"]["source_current_fraction"],
                "compensation_ratio": abs(values["skyrmionium_q_zero"]["charge_hall_angle"]) / denominator if denominator else np.nan,
                "q_reversal_error": abs(values["skyrmion_q_plus"]["charge_hall_angle"] + values["skyrmion_q_minus"]["charge_hall_angle"]),
                "uniform_abs_hall": abs(values["uniform"]["charge_hall_angle"]),
                "max_current_error": max(values[k]["current_conservation_error"] for k in KINDS),
                "max_probe_error": max(values[k]["probe_current_error"] for k in KINDS),
                "max_unitarity_error": max(values[k]["scattering_unitarity_error"] for k in KINDS),
                "max_scattering_mismatch": max(values[k]["scattering_charge_mismatch"] for k in KINDS),
            })
    return entries


def profile_assessment(records, scan="length_local_profile"):
    entries = []
    for nx in NX_VALUES:
        rows = sorted(
            rows_for(records, scan, nx=nx,
                     kind="skyrmionium_q_zero", width=2),
            key=lambda row: row["case"]["start_in_cell"],
        )
        if not rows:
            continue
        hall = np.array([row["result"]["Rxy_h_over_e2"] for row in rows])
        qwin = np.array([row["result"]["windowed_topological_charge"] for row in rows])
        scale = float(np.dot(qwin, hall) / np.dot(qwin, qwin)) if np.dot(qwin, qwin) else 0.0
        amplitude = float(np.ptp(hall))
        nrmse = float(np.sqrt(np.mean((hall - scale * qwin) ** 2)) / amplitude) if amplitude else np.nan
        hall_signs = compressed_signs(hall)
        q_signs = compressed_signs(qwin)
        sign_ok = bool(
            len(hall_signs) == 3 and hall_signs[0] == hall_signs[2]
            and hall_signs[0] == -hall_signs[1]
            and (hall_signs == q_signs or hall_signs == [-s for s in q_signs])
        )
        entries.append({
            "Nx": nx, "all_points_valid": all(row["result"]["valid_hall_point"] for row in rows),
            "min_source_current_fraction": min(row["result"]["source_current_fraction"] for row in rows),
            "hall_sign_sequence": hall_signs, "window_charge_sign_sequence": q_signs,
            "sign_order_acceptance": sign_ok, "nrmse": nrmse,
            "convolution_acceptance": bool(nrmse <= 0.30),
            "scale": scale, "hall_amplitude": amplitude,
        })
    return entries


def mirror_profile_assessment(records):
    entries = []
    for nx in (2, 4):
        left = sorted(rows_for(records, "length_local_profile", nx=nx),
                      key=lambda row: row["case"]["start_in_cell"])
        right = sorted(rows_for(records, "length_local_profile_right", nx=nx),
                       key=lambda row: row["case"]["start_in_cell"])
        if len(left) != 15 or len(right) != 15:
            continue
        left_hall = np.array([row["result"]["Rxy_h_over_e2"] for row in left])
        right_mirrored = np.array([row["result"]["Rxy_h_over_e2"] for row in right])[::-1]
        sym = 0.5 * (left_hall + right_mirrored)
        antisym = 0.5 * (left_hall - right_mirrored)
        qwin = np.array([row["result"]["windowed_topological_charge"] for row in left])
        scale = float(np.dot(qwin, sym) / np.dot(qwin, qwin)) if np.dot(qwin, qwin) else 0.0
        amplitude = float(np.ptp(sym))
        nrmse = float(np.sqrt(np.mean((sym - scale * qwin) ** 2)) / amplitude) if amplitude else np.nan
        hall_signs = compressed_signs(sym)
        q_signs = compressed_signs(qwin)
        sign_ok = bool(
            len(hall_signs) == 3 and hall_signs[0] == hall_signs[2]
            and hall_signs[0] == -hall_signs[1]
            and (hall_signs == q_signs or hall_signs == [-s for s in q_signs])
        )
        entries.append({
            "Nx": nx, "nrmse": nrmse, "convolution_acceptance": bool(nrmse <= 0.30),
            "hall_sign_sequence": hall_signs, "sign_order_acceptance": sign_ok,
            "antisymmetric_to_symmetric_rms_ratio": float(
                np.sqrt(np.mean(antisym ** 2)) / max(np.sqrt(np.mean(sym ** 2)), 1e-30)
            ),
            "scale": scale, "symmetrized_hall": sym.tolist(),
        })
    return entries


def spectrum_assessment(records, scan="length_gap_spectrum"):
    output = []
    for nx in NX_VALUES:
        energies = sorted({row["result"]["energy"] for row in rows_for(records, scan, nx=nx)})
        for energy in energies:
            values = {}
            for kind in KINDS:
                rows = rows_for(records, scan, nx=nx, kind=kind, energy=energy)
                if rows:
                    values[kind] = rows[-1]["result"]
            if len(values) != len(KINDS):
                continue
            denominator = 0.5 * (abs(values["skyrmion_q_plus"]["charge_hall_angle"])
                                 + abs(values["skyrmion_q_minus"]["charge_hall_angle"]))
            output.append({
                "Nx": nx, "energy": energy,
                "all_texture_points_valid": all(values[k]["valid_hall_point"] for k in KINDS[1:]),
                "q0_valid": values["skyrmionium_q_zero"]["valid_hall_point"],
                "q0_hall_angle": values["skyrmionium_q_zero"]["charge_hall_angle"],
                "qplus_hall_angle": values["skyrmion_q_plus"]["charge_hall_angle"],
                "qminus_hall_angle": values["skyrmion_q_minus"]["charge_hall_angle"],
                "q0_source_current_fraction": values["skyrmionium_q_zero"]["source_current_fraction"],
                "compensation_ratio": abs(values["skyrmionium_q_zero"]["charge_hall_angle"]) / denominator if denominator else np.nan,
            })
    return output


def refinement_assessment(narrow, wide):
    narrow = {round(row["energy"], 9): row for row in narrow if row["Nx"] == 8}
    wide = {round(row["energy"], 9): row for row in wide if row["Nx"] == 8}
    common = []
    for energy in sorted(set(narrow) & set(wide)):
        a, b = narrow[energy], wide[energy]
        robust = bool(
            a["all_texture_points_valid"] and b["all_texture_points_valid"]
            and a["compensation_ratio"] < 0.1 and b["compensation_ratio"] < 0.1
        )
        common.append({
            "energy": energy, "narrow_valid": a["all_texture_points_valid"],
            "wide_valid": b["all_texture_points_valid"],
            "narrow_compensation_ratio": a["compensation_ratio"],
            "wide_compensation_ratio": b["compensation_ratio"],
            "robust_compensation": robust,
        })
    robust_energies = [row["energy"] for row in common if row["robust_compensation"]]
    groups = []
    for energy in robust_energies:
        if not groups or energy - groups[-1][-1] > 0.00101:
            groups.append([energy])
        else:
            groups[-1].append(energy)
    windows = [{"low": group[0], "high": group[-1], "width": group[-1] - group[0],
                "point_count": len(group)} for group in groups]
    return {"entries": common, "robust_windows": windows}


def spectrum_summary(spectrum):
    output = []
    for nx in NX_VALUES:
        rows = [row for row in spectrum if row["Nx"] == nx]
        valid = [row for row in rows if row["all_texture_points_valid"]]
        gap = [row for row in valid if 1.077143 <= row["energy"] <= 1.1224]
        compensated = [row for row in valid if row["compensation_ratio"] < 0.1]
        gap_compensated = [row for row in gap if row["compensation_ratio"] < 0.1]
        output.append({
            "Nx": nx, "energy_point_count": len(rows), "all_valid_count": len(valid),
            "compensated_count": len(compensated), "gap_valid_count": len(gap),
            "gap_compensated_count": len(gap_compensated),
            "valid_energies": [row["energy"] for row in valid],
            "gap_compensated_energies": [row["energy"] for row in gap_compensated],
            "minimum_valid_compensation_ratio": min(
                (row["compensation_ratio"] for row in valid), default=np.nan),
            "median_valid_compensation_ratio": float(np.median(
                [row["compensation_ratio"] for row in valid])) if valid else np.nan,
        })
    return output


def quality_assessment(records):
    results = [row["result"] for row in records]
    return {
        "max_current_conservation_error": max(r["current_conservation_error"] for r in results),
        "max_probe_current_error": max(r["probe_current_error"] for r in results),
        "max_gauge_error": max(r["gauge_invariance_error"] for r in results),
        "max_unitarity_error": max(r["scattering_unitarity_error"] for r in results),
        "max_scattering_charge_mismatch": max(r["scattering_charge_mismatch"] for r in results),
        "all_transmission_bounds_ok": all(r["transmission_bound_ok"] for r in results),
    }


def validation_assessment(out: Path):
    path = out / "eta_validation.jsonl"
    if not path.exists():
        return {}
    rows = read_records(path)
    best = {}
    for row in rows:
        current = best.get(row["physical_id"])
        if current is None or row["validation_eta"] < current["validation_eta"]:
            best[row["physical_id"]] = row
    selected = list(best.values())
    valid = [row for row in selected if row["physical_result"]["valid_hall_point"]]
    return {
        "validated_point_count": len(selected),
        "minimum_eta": min((row["validation_eta"] for row in selected), default=np.nan),
        "max_unitarity_error": max((row["result"]["scattering_unitarity_error"] for row in selected), default=np.nan),
        "max_gauge_error": max((row["result"]["gauge_invariance_error"] for row in selected), default=np.nan),
        "max_scattering_charge_mismatch": max((row["result"]["scattering_charge_mismatch"] for row in selected), default=np.nan),
        "max_valid_hall_absolute_change": max((
            abs(row["result"]["charge_hall_angle"] - row["physical_result"]["charge_hall_angle"])
            for row in valid), default=np.nan),
        "max_valid_hall_relative_change": max((
            abs(row["result"]["charge_hall_angle"] - row["physical_result"]["charge_hall_angle"])
            / max(abs(row["result"]["charge_hall_angle"]), 1e-4)
            for row in valid), default=np.nan),
    }


def spectral_map_assessment(map_dir: Path):
    entries = []
    for path in sorted(map_dir.glob("q0_*.npz")):
        data = np.load(path)
        injectivity = data["source_injectivity"]
        jx = np.abs(data["bond_current_x"])
        jy = np.abs(data["bond_current_y"])
        nx = injectivity.shape[0] // 18
        label = "center" if "_center_" in path.name else "upper"
        cells = np.array([np.sum(injectivity[18*i:18*(i+1)]) for i in range(nx)])
        cells /= max(np.sum(cells), 1e-30)
        entries.append({
            "file": path.name, "energy_label": label, "Nx": nx,
            "cell_injectivity_fractions": cells.tolist(),
            "last_to_first_cell_injectivity_ratio": float(cells[-1] / max(cells[0], 1e-30)),
            "transverse_to_longitudinal_abs_current_ratio": float(
                np.sum(jy) / max(np.sum(jx), 1e-30)),
        })
    return entries


def plot_length_summary(baseline, out: Path):
    length = json.loads((ROOT / "results" / "length_scaling" /
                         "skyrmionium_q_zero_A18_Ny2.json").read_text(encoding="utf-8"))
    nx = np.asarray(length["parameters"]["Nx"])
    transmissions = np.asarray(length["transmissions"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for index, energy in enumerate(length["parameters"]["energy"]):
        axes[0].semilogy(nx, transmissions[:, index], "o-", label=f"E/t={energy:.4f}")
    axes[0].set(xlabel="$N_x$", ylabel="$T_{xx}$", title="two-terminal length scaling")
    axes[0].legend(fontsize=8)
    rows = [row for row in baseline if row["probe_width"] == 4]
    axes[1].semilogy([row["Nx"] for row in rows],
                     [row["q0_source_current_fraction"] for row in rows], "o-")
    axes[1].axhline(1e-6, color="tab:red", linestyle="--", label="validity threshold")
    axes[1].set(xlabel="$N_x$", ylabel="$|I_L|/N$", title="Hall-point current validity")
    axes[1].legend(fontsize=8)
    for key, label in (("q0_hall_angle", "Q=0"), ("qplus_hall_angle", "Q=+1"),
                       ("qminus_hall_angle", "Q=-1")):
        x = [row["Nx"] for row in rows if row["all_texture_points_valid"]]
        y = [row[key] for row in rows if row["all_texture_points_valid"]]
        axes[2].plot(x, y, "o-", label=label)
    axes[2].axhline(0, color="0.6", linewidth=0.8)
    axes[2].set(xlabel="$N_x$", ylabel="charge Hall angle",
                title="valid minigap-center Hall points")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(NX_VALUES)
    fig.savefig(out / "main_length_transport_hall.png", dpi=240)
    plt.close(fig)


def plot_profiles(records, assessment, out: Path, *, scan="length_local_profile",
                  filename="local_compensation_vs_length.png", energy_label="minigap center"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for index, nx in enumerate(NX_VALUES):
        ax = axes.flat[index]
        rows = sorted(rows_for(records, scan, nx=nx),
                      key=lambda row: row["case"]["start_in_cell"])
        item = next(entry for entry in assessment if entry["Nx"] == nx)
        x = [row["case"]["start_in_cell"] for row in rows]
        hall = np.array([row["result"]["Rxy_h_over_e2"] for row in rows])
        qwin = np.array([row["result"]["windowed_topological_charge"] for row in rows])
        ax.plot(x, hall, "o-", label="NEGF local $R_{xy}$")
        ax.plot(x, item["scale"] * qwin, "s--", label="fitted window $Q$")
        ax.axhline(0, color="0.6", linewidth=0.8)
        state = "valid" if item["all_points_valid"] else "invalid current"
        ax.set_title(f"Nx={nx}: NRMSE={item['nrmse']:.3f}, {state}")
        ax.set(xlabel="probe start within central cell", ylabel="local response")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(energy_label)
    fig.savefig(out / filename, dpi=240)
    plt.close(fig)


def plot_mirror_profiles(records, assessment, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, item in zip(axes, assessment):
        nx = item["Nx"]
        left = sorted(rows_for(records, "length_local_profile", nx=nx),
                      key=lambda row: row["case"]["start_in_cell"])
        right = sorted(rows_for(records, "length_local_profile_right", nx=nx),
                       key=lambda row: row["case"]["start_in_cell"])
        x = np.arange(1, 16)
        left_hall = np.array([row["result"]["Rxy_h_over_e2"] for row in left])
        right_hall = np.array([row["result"]["Rxy_h_over_e2"] for row in right])[::-1]
        qwin = np.array([row["result"]["windowed_topological_charge"] for row in left])
        ax.plot(x, left_hall, color="0.65", linestyle="--", label="left central cell")
        ax.plot(x, right_hall, color="0.35", linestyle=":", label="mirrored right cell")
        ax.plot(x, item["symmetrized_hall"], "o-", label="inversion average")
        ax.plot(x, item["scale"] * qwin, "s--", label="fitted window $Q$")
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_title(f"Nx={nx}, averaged NRMSE={item['nrmse']:.3f}")
        ax.set(xlabel="probe start within cell", ylabel="local $R_{xy}$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "mirror_symmetrized_local_profiles.png", dpi=240)
    plt.close(fig)


def plot_spectrum(spectrum, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, nx in zip(axes.flat, NX_VALUES):
        rows = [row for row in spectrum if row["Nx"] == nx]
        for key, label in (("q0_hall_angle", "Q=0"), ("qplus_hall_angle", "Q=+1"),
                           ("qminus_hall_angle", "Q=-1")):
            ax.plot([row["energy"] for row in rows], [row[key] for row in rows], "o-", label=label)
        invalid = [row for row in rows if not row["all_texture_points_valid"]]
        if invalid:
            ax.scatter([row["energy"] for row in invalid], [row["q0_hall_angle"] for row in invalid],
                       facecolors="none", edgecolors="red", s=70, label="invalid current")
        ax.axvspan(1.077143, 1.1224, color="0.7", alpha=0.2)
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_title(f"Nx={nx}")
        ax.set(ylabel="charge Hall angle")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Energy E/t")
    axes[1, 1].set_xlabel("Energy E/t")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "hall_gap_spectrum_vs_length.png", dpi=240)
    plt.close(fig)


def plot_phase_maps(spectrum, out: Path):
    energies = sorted({row["energy"] for row in spectrum})
    current = np.full((len(NX_VALUES), len(energies)), np.nan)
    compensation = np.full_like(current, np.nan)
    valid = np.zeros_like(current, dtype=bool)
    for row in spectrum:
        i = NX_VALUES.index(row["Nx"])
        j = energies.index(row["energy"])
        current[i, j] = row["q0_source_current_fraction"]
        valid[i, j] = row["all_texture_points_valid"]
        if valid[i, j]:
            compensation[i, j] = row["compensation_ratio"]
    extent = (min(energies), max(energies), 0.5, 4.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(np.log10(np.maximum(current, 1e-16)), origin="lower",
                         aspect="auto", extent=extent, cmap="magma", vmin=-8, vmax=0)
    im1 = axes[1].imshow(np.clip(compensation, 0, 0.5), origin="lower",
                         aspect="auto", extent=extent, cmap="viridis", vmin=0, vmax=0.5)
    invalid_y, invalid_x = np.where(~valid)
    axes[1].scatter([energies[j] for j in invalid_x], [i + 1 for i in invalid_y],
                    marker="x", color="red", s=18, label="invalid current")
    for ax in axes:
        ax.axvline(1.077143, color="white", linestyle="--", linewidth=1)
        ax.axvline(1.1224, color="white", linestyle="--", linewidth=1)
        ax.set_yticks(range(1, 5), NX_VALUES)
        ax.set_xlabel("Energy E/t")
        ax.set_ylabel("array length $N_x$")
    axes[0].set_title("Q=0 longitudinal current fraction")
    axes[1].set_title("Q=0 / mean |Q=±1| Hall ratio")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.colorbar(im0, ax=axes[0], label="$\\log_{10}(|I_L|/N)$")
    fig.colorbar(im1, ax=axes[1], label="compensation ratio (clipped at 0.5)")
    fig.savefig(out / "energy_length_validity_compensation_map.png", dpi=240)
    plt.close(fig)


def plot_refinement(narrow, wide, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for rows, label, style in ((narrow, "probe width 4", "o-"),
                               (wide, "probe width 16", "s--")):
        rows = [row for row in rows if row["Nx"] == 8]
        axes[0].semilogy([row["energy"] for row in rows],
                         [row["q0_source_current_fraction"] for row in rows], style, label=label)
        axes[1].semilogy([row["energy"] for row in rows],
                         [row["compensation_ratio"] for row in rows], style, label=label)
    axes[0].axhline(1e-6, color="tab:red", linestyle=":", label="current threshold")
    axes[1].axhline(0.1, color="tab:red", linestyle=":", label="compensation threshold")
    axes[0].set(xlabel="Energy E/t", ylabel="$|I_L|/N$", title="Nx=8 current validity")
    axes[1].set(xlabel="Energy E/t", ylabel="Q=0 / mean |Q=±1| Hall",
                title="probe-width robustness")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "upper_edge_refined_window.png", dpi=240)
    plt.close(fig)


def plot_spectral_maps(map_dir: Path, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for row, label in enumerate(("center", "upper")):
        for column, nx in enumerate(NX_VALUES):
            data = np.load(map_dir / f"q0_{label}_Nx{nx}.npz")
            injectivity = data["source_injectivity"]
            ax = axes[row, column]
            ax.imshow(injectivity.T, origin="lower", aspect="auto", cmap="magma",
                      vmin=0, vmax=np.percentile(injectivity, 99))
            energy = "1.09977" if label == "center" else "1.1224"
            ax.set_title(f"E/t={energy}, Nx={nx}")
            ax.set_xlabel("x")
            if column == 0:
                ax.set_ylabel("y")
    fig.savefig(out / "injectivity_length_crossover.png", dpi=240)
    plt.close(fig)


def main():
    out = ROOT / "results" / "length_hall_main_v1"
    records = read_records(out / "hall_cases.jsonl")
    baseline = baseline_assessment(records)
    profiles = profile_assessment(records)
    upper_edge_profiles = profile_assessment(records, scan="upper_edge_local_profile")
    mirror_profiles = mirror_profile_assessment(records)
    spectrum = spectrum_assessment(records)
    refinement_narrow = spectrum_assessment(records, scan="upper_edge_refinement")
    refinement_wide = spectrum_assessment(records, scan="upper_edge_refinement_wide")
    refinement = refinement_assessment(refinement_narrow, refinement_wide)
    spectrum_stats = spectrum_summary(spectrum)
    valid_center_nx = [row["Nx"] for row in baseline
                       if row["probe_width"] == 4 and row["all_texture_points_valid"]]
    valid_common_nx8_energies = [row["energy"] for row in spectrum
                                 if row["Nx"] == 8 and row["all_texture_points_valid"]]
    map_metrics = spectral_map_assessment(out / "spectral_maps")
    assessment = {
        "physical_record_count": len(records),
        "baseline": baseline,
        "local_profiles": profiles,
        "upper_edge_local_profiles": upper_edge_profiles,
        "mirror_symmetrized_profiles": mirror_profiles,
        "gap_spectrum": spectrum,
        "upper_edge_refinement": refinement,
        "gap_spectrum_summary": spectrum_stats,
        "quality": quality_assessment(records),
        "targeted_eta_validation": validation_assessment(out),
        "spectral_map_metrics": map_metrics,
        "valid_minigap_center_Nx": valid_center_nx,
        "valid_common_Nx8_energies": valid_common_nx8_energies,
        "publication_decision": {
            "use_Nx8_center_Hall": False,
            "reason": "longitudinal current is below the preregistered validity threshold",
            "use_Nx8_center_transmission": True,
        },
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    plot_length_summary(baseline, out)
    plot_profiles(records, profiles, out)
    plot_profiles(
        records, upper_edge_profiles, out, scan="upper_edge_local_profile",
        filename="upper_edge_local_compensation_vs_length.png",
        energy_label="upper minigap edge E/t=1.1224",
    )
    plot_mirror_profiles(records, mirror_profiles, out)
    plot_spectrum(spectrum, out)
    plot_phase_maps(spectrum, out)
    plot_refinement(refinement_narrow, refinement_wide, out)
    plot_spectral_maps(out / "spectral_maps", out)
    print(json.dumps({
        "physical_record_count": len(records),
        "valid_minigap_center_Nx": valid_center_nx,
        "valid_common_Nx8_energies": valid_common_nx8_energies,
        "profiles": profiles,
        "upper_edge_profiles": upper_edge_profiles,
        "spectrum_summary": spectrum_stats,
        "upper_edge_refinement": refinement,
        "quality": quality_assessment(records),
        "targeted_eta_validation": validation_assessment(out),
        "spectral_map_metrics": map_metrics,
        "mirror_symmetrized_profiles": mirror_profiles,
    }, indent=2))


if __name__ == "__main__":
    main()
