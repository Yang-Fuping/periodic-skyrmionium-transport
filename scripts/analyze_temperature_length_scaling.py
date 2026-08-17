"""Assess finite-temperature length scaling at the confirmed minigap center."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.statistics import (
    fermi_derivative, finite_temperature_average, fermi_window_mass,
)
from skyrmion_transport.transport import fit_exponential_length


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LABELS = {
    "uniform": "uniform", "skyrmionium_q_zero": "Q=0",
    "skyrmion_q_plus": "Q=+1", "skyrmion_q_minus": "Q=-1",
}
NX_VALUES = (1, 2, 4, 8)
A = 18
E_CENTER = 1.0997714941836594
KBT_VALUES = (0.005, 0.01)
GAP_LOW = 1.077143
GAP_HIGH = 1.122400


def interval_contribution(energy, transmission, ef, kbt, low, high):
    mask = (energy >= low) & (energy <= high)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapezoid(
        transmission[mask] * fermi_derivative(energy[mask], ef, kbt), energy[mask]
    ))


def load_spectrum(out, nx, kind):
    if nx < 8:
        with np.load(out / f"spectrum_Nx{nx}_{kind}.npz") as saved:
            energy, transmission = saved["energy"].copy(), saved["transmission"].copy()
        refinement = out / f"spectrum_refinement_Nx{nx}_{kind}.npz"
        if refinement.exists():
            with np.load(refinement) as saved:
                energy = np.concatenate((energy, saved["energy"]))
                transmission = np.concatenate((transmission, saved["transmission"]))
            order = np.argsort(energy)
            energy, transmission = energy[order], transmission[order]
        return energy, transmission
    source = ROOT / "results" / "finite_temperature_main_v1"
    pieces = []
    for name in (f"longitudinal_{kind}.npz", f"longitudinal_half_step_{kind}.npz"):
        with np.load(source / name) as saved:
            pieces.append((saved["energy"].copy(), saved["transmission"].copy()))
    energy = np.concatenate([piece[0] for piece in pieces])
    transmission = np.concatenate([piece[1] for piece in pieces])
    order = np.argsort(energy)
    energy, transmission = energy[order], transmission[order]
    refinement = out / f"spectrum_refinement_Nx{nx}_{kind}.npz"
    if refinement.exists():
        with np.load(refinement) as saved:
            energy = np.concatenate((energy, saved["energy"]))
            transmission = np.concatenate((transmission, saved["transmission"]))
        order = np.argsort(energy)
        energy, transmission = energy[order], transmission[order]
    return energy, transmission


def load_zero_temperature_reference():
    path = ROOT / "results" / "length_scaling" / "skyrmionium_q_zero_A18_Ny2.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    energies = report["parameters"]["energy"]
    index = int(np.argmin(abs(np.asarray(energies) - E_CENTER)))
    values = np.asarray(report["transmissions"])[:, index]
    return values, report["fits"][str(E_CENTER)]


def main():
    out = ROOT / "results" / "temperature_length_scaling_v1"
    spectra = {nx: {kind: load_spectrum(out, nx, kind) for kind in KINDS} for nx in NX_VALUES}
    rows = []
    for kbt in KBT_VALUES:
        conductance = {kind: [] for kind in KINDS}
        coarse = {kind: [] for kind in KINDS}
        q0_decomposition = []
        for nx in NX_VALUES:
            for kind in KINDS:
                energy, transmission = spectra[nx][kind]
                conductance[kind].append(float(finite_temperature_average(
                    energy, transmission, E_CENTER, kbt
                )))
                coarse[kind].append(float(finite_temperature_average(
                    energy[::2], transmission[::2], E_CENTER, kbt
                )))
            energy, transmission = spectra[nx]["skyrmionium_q_zero"]
            lower = interval_contribution(
                energy, transmission, E_CENTER, kbt, energy[0], GAP_LOW
            )
            inside = interval_contribution(
                energy, transmission, E_CENTER, kbt, GAP_LOW, GAP_HIGH
            )
            upper = interval_contribution(
                energy, transmission, E_CENTER, kbt, GAP_HIGH, energy[-1]
            )
            total = conductance["skyrmionium_q_zero"][-1]
            q0_decomposition.append({
                "Nx": nx, "below_gap": lower, "inside_gap": inside,
                "above_gap": upper, "resolved_sum": lower + inside + upper,
                "outside_gap_fraction": (lower + upper) / total,
                "inside_gap_fraction": inside / total,
            })
        q0 = np.asarray(conductance["skyrmionium_q_zero"])
        uniform = np.asarray(conductance["uniform"])
        qpm = 0.5 * (
            np.asarray(conductance["skyrmion_q_plus"])
            + np.asarray(conductance["skyrmion_q_minus"])
        )
        fit = fit_exponential_length(np.asarray(NX_VALUES) * A, q0)
        rows.append({
            "kBT": kbt,
            "fermi_window_mass": fermi_window_mass(spectra[8]["uniform"][0], E_CENTER, kbt),
            "conductance": conductance,
            "coarse_conductance": coarse,
            "q0_over_uniform": (q0 / uniform).tolist(),
            "q0_over_mean_qpm": (q0 / qpm).tolist(),
            "q0_length_fit": fit,
            "q0_thermal_decomposition": q0_decomposition,
            "fine_coarse_relative_difference": {
                kind: (
                    abs(np.asarray(conductance[kind]) - np.asarray(coarse[kind]))
                    / np.maximum(abs(np.asarray(conductance[kind])), 1e-30)
                ).tolist() for kind in KINDS
            },
        })

    zero_values, zero_fit = load_zero_temperature_reference()
    assessment = {
        "parameters": {
            "A": A, "R": 8, "J_over_t": 5, "Ny": 2,
            "Nx": list(NX_VALUES), "energy": E_CENTER,
            "energy_step": 0.00025, "refined_q0_step_Nx4_Nx8": 0.000125,
            "kBT": list(KBT_VALUES), "gap_bounds": [GAP_LOW, GAP_HIGH],
        },
        "zero_temperature_reference": {
            "q0_transmission": zero_values.tolist(), "fit": zero_fit,
        },
        "finite_temperature": rows,
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    lengths = np.asarray(NX_VALUES) * A
    axes[0].semilogy(NX_VALUES, zero_values, "o-", label="$T=0$, Q=0")
    fitted = np.exp(zero_fit["intercept"] + zero_fit["slope"] * lengths)
    axes[0].semilogy(NX_VALUES, fitted, "--", label=f"fit, $R^2$={zero_fit['r2']:.5f}")
    axes[0].set(xlabel="$N_x$", ylabel="Transmission", title="zero-temperature exponential decay")

    styles = {0.005: "o-", 0.01: "s--"}
    for row in rows:
        for kind in KINDS:
            axes[1].semilogy(
                NX_VALUES, row["conductance"][kind], styles[row["kBT"]],
                label=f"{LABELS[kind]}, $k_BT/t$={row['kBT']:g}",
            )
        axes[2].semilogy(
            NX_VALUES, row["q0_over_uniform"], styles[row["kBT"]],
            label=f"Q=0 / uniform, {row['kBT']:g}",
        )
        axes[2].semilogy(
            NX_VALUES, row["q0_over_mean_qpm"], styles[row["kBT"]], alpha=0.65,
            label=f"Q=0 / mean Q=+/-1, {row['kBT']:g}",
        )
    axes[1].set(xlabel="$N_x$", ylabel="$G/(e^2/h)$", title="finite-temperature length scaling")
    axes[2].set(xlabel="$N_x$", ylabel="conductance ratio", title="effective thermal suppression")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.savefig(out / "temperature_length_scaling.png", dpi=240)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, constrained_layout=True)
    for ax, row in zip(axes, rows):
        decomposition = row["q0_thermal_decomposition"]
        inside = np.asarray([item["inside_gap_fraction"] for item in decomposition])
        outside = np.asarray([item["outside_gap_fraction"] for item in decomposition])
        ax.bar(NX_VALUES, inside, label="inside-gap tunneling", color="tab:blue")
        ax.bar(NX_VALUES, outside, bottom=inside, label="thermally activated band-edge states",
               color="tab:orange")
        ax.set(xlabel="$N_x$", title=f"$k_BT/t={row['kBT']:g}$")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("fraction of Q=0 thermal conductance")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Crossover from gap tunneling to thermally activated transport")
    fig.savefig(out / "thermal_activation_decomposition.png", dpi=240)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
