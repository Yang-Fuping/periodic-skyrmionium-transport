"""Finite-temperature longitudinal and four-terminal charge-Hall assessment."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.multiterminal import four_terminal_observables
from skyrmion_transport.statistics import (
    finite_temperature_average, fermi_window_mass,
)


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LABELS = {
    "uniform": "uniform", "skyrmionium_q_zero": "Q=0",
    "skyrmion_q_plus": "Q=+1", "skyrmion_q_minus": "Q=-1",
}
E_CENTER = 1.0997714941836594
E_EDGE = 1.120
LONGITUDINAL_KBT = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.01)
HALL_KBT = (0.0005, 0.001, 0.002, 0.003)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_longitudinal(out):
    data = {}
    for kind in KINDS:
        pieces = []
        for name in (f"longitudinal_{kind}.npz", f"longitudinal_half_step_{kind}.npz"):
            with np.load(out / name) as saved:
                pieces.append({key: saved[key].copy() for key in saved.files})
        order = np.argsort(np.concatenate([piece["energy"] for piece in pieces]))
        data[kind] = {
            "energy": np.concatenate([piece["energy"] for piece in pieces])[order],
            "transmission": np.concatenate([piece["transmission"] for piece in pieces])[order],
            "lead_channels": np.concatenate([piece["lead_channels"] for piece in pieces])[order],
        }
    return data


def longitudinal_assessment(data):
    rows = []
    for kbt in LONGITUDINAL_KBT:
        values = {}
        coarse_values = {}
        for kind in KINDS:
            energy = data[kind]["energy"]
            transmission = data[kind]["transmission"]
            values[kind] = float(finite_temperature_average(energy, transmission, E_CENTER, kbt))
            coarse_values[kind] = float(finite_temperature_average(
                energy[::2], transmission[::2], E_CENTER, kbt
            ))
        topological_mean = 0.5 * (values["skyrmion_q_plus"] + values["skyrmion_q_minus"])
        rows.append({
            "kBT": kbt,
            "fermi_window_mass": fermi_window_mass(data["uniform"]["energy"], E_CENTER, kbt),
            "conductance": values,
            "q0_over_uniform": values["skyrmionium_q_zero"] / values["uniform"],
            "q0_over_mean_qpm": values["skyrmionium_q_zero"] / topological_mean,
            "fine_coarse_relative_difference": {
                k: abs(values[k] - coarse_values[k]) / max(abs(values[k]), 1e-30)
                for k in KINDS
            },
            "coarse_conductance": coarse_values,
        })
    zero = {}
    for kind in KINDS:
        zero[kind] = float(np.interp(
            E_CENTER, data[kind]["energy"], data[kind]["transmission"]
        ))
    return {"zero_temperature_interpolated": zero, "thermal": rows}


def hall_spectra(out):
    records = (
        read_jsonl(out / "hall_spectrum.jsonl")
        + read_jsonl(out / "hall_spectrum_half_step.jsonl")
        + read_jsonl(out / "hall_spectrum_quarter_step.jsonl")
    )
    spectra = {}
    for kind in KINDS:
        rows = sorted((row for row in records if row["case"]["kind"] == kind),
                      key=lambda row: row["case"]["energy"])
        spectra[kind] = {
            "energy": np.asarray([row["case"]["energy"] for row in rows]),
            "T": np.asarray([row["result"]["T"] for row in rows]),
            "N": np.asarray([row["result"]["N_total"] for row in rows], dtype=float),
            "zero_hall": np.asarray([row["result"]["charge_hall_angle"] for row in rows]),
        }
    return spectra


def thermal_hall_one(spectrum, kbt, stride=1):
    energy = spectrum["energy"][::stride]
    matrix = spectrum["T"][::stride]
    channels = spectrum["N"][::stride]
    thermal_matrix = finite_temperature_average(energy, matrix, E_EDGE, kbt)
    thermal_channels = float(finite_temperature_average(energy, channels, E_EDGE, kbt))
    obs = four_terminal_observables(thermal_matrix)
    source_fraction = abs(float(obs["currents"][0])) / max(thermal_channels, 1.0)
    return {
        "charge_hall_angle": float(obs["hall_angle"]),
        "Rxx_h_over_e2": float(obs["Rxx_h_over_e2"]),
        "Rxy_h_over_e2": float(obs["Rxy_h_over_e2"]),
        "source_current": abs(float(obs["currents"][0])),
        "source_current_fraction": source_fraction,
        "valid_charge_hall": bool(np.isfinite(obs["hall_angle"]) and source_fraction >= 1e-6),
        "current_conservation_error": float(obs["current_conservation_error"]),
        "probe_current_error": float(obs["probe_current_error"]),
        "gauge_invariance_error": float(obs["gauge_invariance_error"]),
        "thermal_channels": thermal_channels,
    }


def hall_assessment(spectra):
    rows = []
    for kbt in HALL_KBT:
        fine = {kind: thermal_hall_one(spectra[kind], kbt, 1) for kind in KINDS}
        coarse = {kind: thermal_hall_one(spectra[kind], kbt, 2) for kind in KINDS}
        denominator = 0.5 * (
            abs(fine["skyrmion_q_plus"]["charge_hall_angle"])
            + abs(fine["skyrmion_q_minus"]["charge_hall_angle"])
        )
        ratio = abs(fine["skyrmionium_q_zero"]["charge_hall_angle"]) / denominator
        coarse_denominator = 0.5 * (
            abs(coarse["skyrmion_q_plus"]["charge_hall_angle"])
            + abs(coarse["skyrmion_q_minus"]["charge_hall_angle"])
        )
        coarse_ratio = abs(coarse["skyrmionium_q_zero"]["charge_hall_angle"]) / coarse_denominator
        fine_pass = bool(all(fine[k]["valid_charge_hall"] for k in KINDS[1:]) and ratio < 0.1)
        coarse_pass = bool(all(coarse[k]["valid_charge_hall"] for k in KINDS[1:]) and coarse_ratio < 0.1)
        rows.append({
            "kBT": kbt,
            "fermi_window_mass": fermi_window_mass(spectra["uniform"]["energy"], E_EDGE, kbt),
            "textures": fine,
            "compensation_ratio": ratio,
            "coarse_compensation_ratio": coarse_ratio,
            "passes_compensation": fine_pass,
            "coarse_passes_compensation": coarse_pass,
            "resolution_verdict": (
                "robust_pass" if fine_pass and coarse_pass else
                "robust_fail" if not fine_pass and not coarse_pass else
                "resolution_sensitive"
            ),
            "q_reversal_error": abs(
                fine["skyrmion_q_plus"]["charge_hall_angle"]
                + fine["skyrmion_q_minus"]["charge_hall_angle"]
            ),
            "uniform_abs_hall": abs(fine["uniform"]["charge_hall_angle"]),
            "max_fine_coarse_absolute_hall_difference": max(
                abs(fine[k]["charge_hall_angle"] - coarse[k]["charge_hall_angle"])
                for k in KINDS
            ),
            "max_thermal_gauge_error": max(fine[k]["gauge_invariance_error"] for k in KINDS),
        })

    zero_values = {}
    for kind in KINDS:
        index = int(np.argmin(abs(spectra[kind]["energy"] - E_EDGE)))
        zero_values[kind] = float(spectra[kind]["zero_hall"][index])
    zero_denominator = 0.5 * (abs(zero_values["skyrmion_q_plus"]) + abs(zero_values["skyrmion_q_minus"]))
    return {
        "zero_temperature_nearest_grid": zero_values,
        "zero_temperature_compensation_ratio": abs(zero_values["skyrmionium_q_zero"]) / zero_denominator,
        "thermal": rows,
    }


def plot(longitudinal, hall, spectra, longitudinal_data_for_plot, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    # Show the clean high-resolution spectrum around the confirmed minigap.
    for kind in KINDS:
        axes[0].semilogy(
            longitudinal_data_for_plot[kind]["energy"],
            np.maximum(longitudinal_data_for_plot[kind]["transmission"], 1e-30),
            label=LABELS[kind],
        )
    axes[0].axvline(E_CENTER, color="black", linestyle=":", linewidth=1)
    axes[0].set(xlabel="Energy E/t", ylabel="Transmission", title="Nx=8 spectrum near minigap")
    for kind in KINDS:
        axes[1].semilogy(
            [row["kBT"] for row in longitudinal["thermal"]],
            [row["conductance"][kind] for row in longitudinal["thermal"]],
            "o-", label=LABELS[kind],
        )
    axes[1].set(xlabel="$k_BT/t$", ylabel="$G/(e^2/h)$",
                title="thermal longitudinal conductance")
    axes[1].set_xscale("log")
    axes[2].semilogy(
        [row["kBT"] for row in hall["thermal"]],
        [row["compensation_ratio"] for row in hall["thermal"]], "o-", color="tab:purple",
        label="thermal compensation ratio",
    )
    axes[2].semilogy(
        [row["kBT"] for row in hall["thermal"]],
        [row["coarse_compensation_ratio"] for row in hall["thermal"]],
        "s--", color="tab:gray", label="coarser-grid check",
    )
    axes[2].axhline(0.1, color="tab:red", linestyle=":", label="criterion")
    axes[2].set(xlabel="$k_BT/t$", ylabel="Q=0 / mean |Q=+/-1| Hall",
                title="thermal survival of edge Hall point")
    axes[2].set_xscale("log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "finite_temperature_main.png", dpi=240)
    plt.close(fig)


def main():
    out = ROOT / "results" / "finite_temperature_main_v1"
    longitudinal_data = load_longitudinal(out)
    longitudinal = longitudinal_assessment(longitudinal_data)
    spectra = hall_spectra(out)
    hall = hall_assessment(spectra)
    assessment = {
        "parameters": {
            "A": 18, "R": 8, "J_over_t": 5, "Nx": 8, "Ny": 2,
            "gap_center": E_CENTER, "edge_working_point": E_EDGE,
            "longitudinal_energy_step": 0.00025, "hall_energy_step": 0.000125,
            "thermal_method": "convolve T(E), then solve Landauer-Buttiker voltage probes",
        },
        "longitudinal": longitudinal,
        "charge_hall": hall,
        "scope_note": "Finite-temperature spin Hall is not inferred because spin-resolved transmission matrices were not stored in this campaign.",
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    plot(longitudinal, hall, spectra, longitudinal_data, out)
    print(json.dumps({
        "longitudinal_thermal": longitudinal["thermal"],
        "hall_thermal": hall["thermal"],
        "zero_temperature_compensation_ratio": hall["zero_temperature_compensation_ratio"],
    }, indent=2))


if __name__ == "__main__":
    main()
