"""Summarize the paired disorder-temperature pilot and select final strengths."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "q05": float(np.quantile(values, 0.05)), "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
    }


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    rows = read_jsonl(out / "pilot.jsonl")
    strengths = sorted({row["case"]["Wd"] for row in rows})
    summary = []
    for wd in strengths:
        group = [row for row in rows if row["case"]["Wd"] == wd]
        item = {"Wd": wd, "samples": len(group), "temperature": {}}
        for kbt in (0.005, 0.01):
            key = str(kbt)
            g4 = np.asarray([row["result"]["4"]["thermal"][key] for row in group])
            g8 = np.asarray([row["result"]["8"]["thermal"][key] for row in group])
            item["temperature"][key] = {
                "Nx4": stats(g4), "Nx8": stats(g8), "paired_G8_over_G4": stats(g8 / g4),
                "fraction_G8_below_clean": float(np.mean(g8 < (0.0030204051 if kbt == 0.005 else 0.0387960847))),
                "fraction_G8_below_5pct_uniform": float(np.mean(g8 < (0.05 * (3.0527294 if kbt == 0.005 else 3.1609656)))),
            }
        item["T_center"] = {
            "Nx4": stats([row["result"]["4"]["T_center"] for row in group]),
            "Nx8": stats([row["result"]["8"]["T_center"] for row in group]),
        }
        summary.append(item)

    assessment = {
        "record_count": len(rows), "strengths": strengths, "summary": summary,
        "selection": {
            "preservation_strength": 0.25,
            "crossover_strength": 0.5,
            "rationale": (
                "Wd=0.25 samples retain strong suppression with a relatively compact distribution; "
                "Wd=0.5 develops a broad rare-event-sensitive crossover distribution. Both require "
                "100 samples. Wd=1.0 is retained as an exploratory localization-dominated point."
            ),
        },
    }
    (out / "pilot_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    positions = np.arange(len(strengths))
    for ax, kbt in zip(axes, (0.005, 0.01)):
        data = [[row["result"]["8"]["thermal"][str(kbt)]
                 for row in rows if row["case"]["Wd"] == wd] for wd in strengths]
        ax.boxplot(data, positions=positions, showfliers=True)
        clean = 0.0030204051 if kbt == 0.005 else 0.0387960847
        ax.axhline(clean, color="black", linestyle="--", label="clean Q=0")
        ax.set_yscale("log")
        ax.set_xticks(positions, [f"{wd:g}" for wd in strengths])
        ax.set(xlabel="$W_d/t$", ylabel="$G_{Q=0}/(e^2/h)$",
               title=f"Nx=8, $k_BT/t={kbt:g}$")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "pilot_disorder_temperature.png", dpi=240)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
