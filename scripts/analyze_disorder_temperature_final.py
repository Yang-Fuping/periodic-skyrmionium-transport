"""Final 100-sample disorder-temperature statistics and paper figures."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


STRENGTHS = (0.25, 0.5)
KBT = (0.005, 0.01)
CLEAN_Q0 = {0.005: {4: 0.0031579657, 8: 0.0030204051},
            0.01: {4: 0.0388186443, 8: 0.0387960847}}
CLEAN_UNIFORM = {0.005: 3.0527294, 0.01: 3.1609656}


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "q05": float(np.quantile(values, 0.05)), "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
        "mean_over_median": float(np.mean(values) / np.median(values)),
    }


def bootstrap_median_ci(values, seed, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return [float(low), float(high)]


def wilson_interval(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def combined_rows(out):
    pilot = [row for row in read_jsonl(out / "pilot.jsonl")
             if row["case"]["Wd"] in STRENGTHS]
    extension = read_jsonl(out / "final_extension.jsonl")
    rows = pilot + extension
    unique = {row["id"]: row for row in rows}
    return list(unique.values())


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    rows = combined_rows(out)
    summary = []
    for wd in STRENGTHS:
        group = sorted((row for row in rows if row["case"]["Wd"] == wd),
                       key=lambda row: row["case"]["sample"])
        item = {"Wd": wd, "sample_count": len(group), "temperature": {}}
        for kbt in KBT:
            key = str(kbt)
            g4 = np.asarray([row["result"]["4"]["thermal"][key] for row in group])
            g8 = np.asarray([row["result"]["8"]["thermal"][key] for row in group])
            item["temperature"][key] = {
                "Nx4": distribution(g4), "Nx8": distribution(g8),
                "paired_G8_over_G4": distribution(g8 / g4),
                "Nx8_median_bootstrap_95ci": bootstrap_median_ci(
                    g8, int(100000 * wd + 1000 * kbt)
                ),
                "paired_ratio_median_bootstrap_95ci": bootstrap_median_ci(
                    g8 / g4, int(200000 * wd + 1000 * kbt)
                ),
                "fraction_G8_less_than_G4": float(np.mean(g8 < g4)),
                "fraction_G8_less_than_G4_wilson_95ci": wilson_interval(
                    int(np.count_nonzero(g8 < g4)), len(g8)
                ),
                "fraction_G8_below_clean_Q0": float(np.mean(g8 < CLEAN_Q0[kbt][8])),
                "fraction_G8_below_5pct_uniform": float(np.mean(g8 < 0.05 * CLEAN_UNIFORM[kbt])),
                "median_G8_over_uniform": float(np.median(g8) / CLEAN_UNIFORM[kbt]),
                "q95_G8_over_uniform": float(np.quantile(g8, 0.95) / CLEAN_UNIFORM[kbt]),
            }
        item["T_center"] = {
            "Nx4": distribution([row["result"]["4"]["T_center"] for row in group]),
            "Nx8": distribution([row["result"]["8"]["T_center"] for row in group]),
        }
        summary.append(item)

    assessment = {
        "record_count": len(rows), "expected_record_count": 200,
        "paired_nested_disorder": True,
        "energy_step": 0.001,
        "summary": summary,
        "interpretation_rule": (
            "Robust suppression requires the Nx=8 median and q95 to remain below 5% of the "
            "clean-uniform thermal conductance; the paired length ratio is reported separately."
        ),
        "publication_scope": {
            "quantitative_temperature": 0.01,
            "quantitative_grid_convergence": (
                "Representative-subset median shifts from dE=0.0005 to 0.00025 are "
                "1.1% (Wd=0.25) and 1.5% (Wd=0.5) for Nx=8."
            ),
            "qualitative_only_case": (
                "kBT=0.005, Wd=0.5 is not energy-grid converged because narrow coherent "
                "resonances move the representative-subset median by 32%."
            ),
        },
    }
    (out / "final_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for column, kbt in enumerate(KBT):
        data = []
        labels = []
        for wd in STRENGTHS:
            group = [row for row in rows if row["case"]["Wd"] == wd]
            for nx in (4, 8):
                data.append([row["result"][str(nx)]["thermal"][str(kbt)] for row in group])
                labels.append(f"W={wd:g}\nNx={nx}")
        ax = axes[0, column]
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.axhline(CLEAN_Q0[kbt][8], color="black", linestyle="--", label="clean Q=0, Nx=8")
        ax.set_yscale("log")
        ax.set(ylabel="$G/(e^2/h)$", title=f"$k_BT/t={kbt:g}$")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, column]
        for wd, marker in zip(STRENGTHS, ("o", "s")):
            group = [row for row in rows if row["case"]["Wd"] == wd]
            g4 = np.asarray([row["result"]["4"]["thermal"][str(kbt)] for row in group])
            g8 = np.asarray([row["result"]["8"]["thermal"][str(kbt)] for row in group])
            ax.scatter(g4, g8, alpha=0.45, s=18, marker=marker, label=f"W={wd:g}")
        low, high = ax.get_xlim()
        low = min(low, ax.get_ylim()[0]); high = max(high, ax.get_ylim()[1])
        ax.plot([low, high], [low, high], ":", color="black", label="$G_8=G_4$")
        ax.set(xscale="log", yscale="log", xlabel="$G_{N_x=4}$", ylabel="$G_{N_x=8}$",
               title="paired length response")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.savefig(out / "disorder_temperature_100samples.png", dpi=240)
    plt.close(fig)

    # Ensemble spectra make the fate of the minigap visible rather than reducing
    # the campaign to two thermally averaged scalars.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, nx in zip(axes, (4, 8)):
        for wd in STRENGTHS:
            group = [row for row in rows if row["case"]["Wd"] == wd]
            energy = np.asarray(group[0]["energy"])
            spectra = np.asarray([row["result"][str(nx)]["transmission"] for row in group])
            median = np.median(spectra, axis=0)
            q25 = np.quantile(spectra, 0.25, axis=0)
            q75 = np.quantile(spectra, 0.75, axis=0)
            ax.semilogy(energy, np.maximum(median, 1e-18), label=f"W={wd:g}")
            ax.fill_between(energy, np.maximum(q25, 1e-18), np.maximum(q75, 1e-18), alpha=0.2)
        ax.axvspan(1.077143, 1.1224, color="gray", alpha=0.12, label="Bloch gap")
        ax.set(xlabel="Energy E/t", ylabel="median transmission", title=f"Nx={nx}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "disorder_ensemble_spectra.png", dpi=240)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
