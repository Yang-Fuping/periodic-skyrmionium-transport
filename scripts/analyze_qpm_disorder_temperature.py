"""Analyze paired Q=0,+1,-1 disorder-temperature ensembles."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


KINDS = ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LABELS = {"skyrmionium_q_zero": "Q=0", "skyrmion_q_plus": "Q=+1",
          "skyrmion_q_minus": "Q=-1"}
COLORS = {"skyrmionium_q_zero": "tab:blue", "skyrmion_q_plus": "tab:red",
          "skyrmion_q_minus": "tab:green"}
STRENGTHS = (0.25, 0.5)
KBT = (0.005, 0.01)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return list({row["id"]: row for row in rows}.values())


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)), "q05": float(np.quantile(values, .05)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "q95": float(np.quantile(values, .95)),
            "mean_over_median": float(np.mean(values) / max(np.median(values), 1e-30))}


def bootstrap_median_ci(values, seed, draws=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    return [float(value) for value in np.quantile(medians, (.025, .975))]


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - half), float(center + half)]


def load_joined():
    q0_out = ROOT / "results" / "disorder_temperature_joint_v1"
    q0_rows = read_jsonl(q0_out / "pilot.jsonl") + read_jsonl(q0_out / "final_extension.jsonl")
    q0 = {(row["case"]["Wd"], row["case"]["sample"]): row for row in q0_rows
          if row["case"]["Wd"] in STRENGTHS}
    qpm_out = ROOT / "results" / "disorder_topology_comparison_v1"
    qpm = {(row["case"]["Wd"], row["case"]["sample"]): row
           for row in read_jsonl(qpm_out / "qpm_cases.jsonl")}
    joined = []
    for key in sorted(set(q0) & set(qpm)):
        row = {"Wd": key[0], "sample": key[1], "energy": q0[key]["energy"], "result": {},
               "qminus_mode": qpm[key].get("qminus_mode", "independent")}
        row["result"]["skyrmionium_q_zero"] = q0[key]["result"]
        row["result"].update(qpm[key]["result"])
        joined.append(row)
    return joined, len(q0), len(qpm)


def main():
    out = ROOT / "results" / "disorder_topology_comparison_v1"
    rows, q0_count, qpm_count = load_joined()
    summary = []
    for wd in STRENGTHS:
        group = [row for row in rows if row["Wd"] == wd]
        item = {"Wd": wd, "paired_sample_count": len(group), "temperature": {}}
        for kbt in KBT:
            key = str(kbt)
            thermal = {kind: {nx: np.asarray([
                row["result"][kind][str(nx)]["thermal"][key] for row in group
            ]) for nx in (4, 8)} for kind in KINDS}
            kind_stats = {}
            for kind_index, kind in enumerate(KINDS):
                g4, g8 = thermal[kind][4], thermal[kind][8]
                kind_stats[kind] = {
                    "Nx4": distribution(g4), "Nx8": distribution(g8),
                    "Nx8_median_bootstrap_95ci": bootstrap_median_ci(
                        g8, int(wd * 100000 + kbt * 10000 + kind_index)
                    ),
                    "paired_G8_over_G4": distribution(g8 / g4),
                    "fraction_G8_less_than_G4": float(np.mean(g8 < g4)),
                    "fraction_G8_less_than_G4_wilson_95ci": wilson(
                        int(np.count_nonzero(g8 < g4)), len(g8)
                    ),
                }
            mean_qpm = .5 * (thermal["skyrmion_q_plus"][8] + thermal["skyrmion_q_minus"][8])
            q0 = thermal["skyrmionium_q_zero"][8]
            qplus, qminus = thermal["skyrmion_q_plus"][8], thermal["skyrmion_q_minus"][8]
            independent = np.asarray([row["qminus_mode"] == "independent" for row in group])
            asymmetry = np.abs(qplus[independent] - qminus[independent]) / np.maximum(
                .5 * (qplus[independent] + qminus[independent]), 1e-30
            )
            ratio = q0 / np.maximum(mean_qpm, 1e-30)
            kind_stats["paired_topology_comparison"] = {
                "Q0_over_mean_Qpm": distribution(ratio),
                "Q0_over_mean_Qpm_median_bootstrap_95ci": bootstrap_median_ci(
                    ratio, int(wd * 300000 + kbt * 10000)
                ),
                "fraction_Q0_less_than_mean_Qpm": float(np.mean(q0 < mean_qpm)),
                "fraction_Q0_less_than_mean_Qpm_wilson_95ci": wilson(
                    int(np.count_nonzero(q0 < mean_qpm)), len(q0)
                ),
                "Qplus_Qminus_relative_asymmetry": distribution(asymmetry),
                "Qplus_median_over_Qminus_median": float(np.median(qplus) / np.median(qminus)),
                "independent_Qminus_validation_count": int(np.count_nonzero(independent)),
            }
            item["temperature"][key] = kind_stats
        item["T_center"] = {kind: {str(nx): distribution([
            row["result"][kind][str(nx)]["T_center"] for row in group
        ]) for nx in (4, 8)} for kind in KINDS}
        summary.append(item)

    assessment = {
        "q0_available_count": q0_count, "qpm_record_count": qpm_count,
        "joined_physical_sample_count": len(rows), "expected_joined_count": 200,
        "same_disorder_field_across_Q0_Qplus_Qminus": True,
        "qminus_production_method": (
            "Q- was independently calculated for samples 0,10,...,90 at each Wd; "
            "other Q- longitudinal spectra were copied from Q+ using the validated "
            "two-terminal Q-reversal symmetry. Pre-existing Wd=0.25 samples 1..7 were "
            "also independently calculated before optimization."
        ),
        "parameters": {"A": 18, "R": 8, "J_over_t": 5, "Ny": 2,
                       "Nx": [4, 8], "Wd": list(STRENGTHS), "kBT": list(KBT)},
        "summary": summary,
        "scope": ("kBT/t=0.01 is the quantitative main comparison. kBT/t=0.005 is "
                  "qualitative because the strong-disorder Q=0 integral is grid sensitive."),
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    # Main topology-disorder comparison.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    data, labels, colors = [], [], []
    for wd in STRENGTHS:
        group = [row for row in rows if row["Wd"] == wd]
        for kind in KINDS:
            data.append([row["result"][kind]["8"]["thermal"]["0.01"] for row in group])
            labels.append(f"{LABELS[kind]}\nW={wd:g}")
            colors.append(COLORS[kind])
    boxes = ax.boxplot(data, tick_labels=labels, showfliers=True, patch_artist=True)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color); box.set_alpha(.35)
    ax.set_yscale("log"); ax.set_ylabel(r"$G_{N_x=8}/(e^2/h)$")
    ax.set_title(r"disorder ensembles, $k_BT/t=0.01$"); ax.grid(axis="y", alpha=.25)

    ax = axes[0, 1]
    for wd, marker in zip(STRENGTHS, ("o", "s")):
        group = [row for row in rows if row["Wd"] == wd]
        q0 = np.asarray([row["result"]["skyrmionium_q_zero"]["8"]["thermal"]["0.01"]
                         for row in group])
        qpm = .5 * np.asarray([
            row["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
            + row["result"]["skyrmion_q_minus"]["8"]["thermal"]["0.01"]
            for row in group])
        ax.scatter(qpm, q0, alpha=.45, s=20, marker=marker, label=f"W={wd:g}")
    bounds = [1e-4, 1]
    ax.plot(bounds, bounds, ":", color="black", label="equal conductance")
    ax.set(xscale="log", yscale="log", xlabel=r"mean $G_{Q=\pm1}$",
           ylabel=r"$G_{Q=0}$", title="same-disorder topology comparison")
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = axes[1, 0]
    ratio_data, ratio_labels = [], []
    for wd in STRENGTHS:
        group = [row for row in rows if row["Wd"] == wd]
        for kind in KINDS:
            g4 = np.asarray([row["result"][kind]["4"]["thermal"]["0.01"] for row in group])
            g8 = np.asarray([row["result"][kind]["8"]["thermal"]["0.01"] for row in group])
            ratio_data.append(g8 / g4); ratio_labels.append(f"{LABELS[kind]}\nW={wd:g}")
    ax.boxplot(ratio_data, tick_labels=ratio_labels, showfliers=True)
    ax.axhline(1, color="black", linestyle=":")
    ax.set_yscale("log"); ax.set_ylabel(r"$G_8/G_4$")
    ax.set_title("paired length response"); ax.grid(axis="y", alpha=.25)

    ax = axes[1, 1]
    for wd, marker in zip(STRENGTHS, ("o", "s")):
        group = [row for row in rows if row["Wd"] == wd]
        qp = np.asarray([row["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
                         for row in group])
        qm = np.asarray([row["result"]["skyrmion_q_minus"]["8"]["thermal"]["0.01"]
                         for row in group])
        ax.scatter(qp, qm, alpha=.45, s=20, marker=marker, label=f"W={wd:g}")
    ax.plot(bounds, bounds, ":", color="black")
    ax.set(xscale="log", yscale="log", xlabel=r"$G_{Q=+1}$", ylabel=r"$G_{Q=-1}$",
           title="Q-reversal ensemble symmetry")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.savefig(out / "topology_disorder_comparison.png", dpi=240)
    plt.close(fig)

    # Median spectra expose whether a conductance ratio is a gap or a broad scattering effect.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, wd in zip(axes, STRENGTHS):
        group = [row for row in rows if row["Wd"] == wd]
        energy = np.asarray(group[0]["energy"])
        for kind in KINDS:
            spectra = np.asarray([row["result"][kind]["8"]["transmission"] for row in group])
            median = np.median(spectra, axis=0)
            q25, q75 = np.quantile(spectra, (.25, .75), axis=0)
            ax.semilogy(energy, np.maximum(median, 1e-18), color=COLORS[kind],
                        label=LABELS[kind])
            ax.fill_between(energy, np.maximum(q25, 1e-18), np.maximum(q75, 1e-18),
                            color=COLORS[kind], alpha=.12)
        ax.axvspan(1.077143, 1.1224, color="gray", alpha=.12, label="Q=0 Bloch gap")
        ax.set(xlabel="Energy E/t", ylabel="median transmission",
               title=rf"$W_d/t={wd:g},\ N_x=8$")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(out / "topology_ensemble_spectra.png", dpi=240)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
