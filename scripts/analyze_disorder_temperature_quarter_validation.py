"""Assess convergence from dE=0.0005 to 0.00025 on the representative subset."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


def read_unique(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return list({row["physical_id"]: row for row in rows}.values())


def stats(values):
    values = np.asarray(values)
    return {"median": float(np.median(values)), "q95": float(np.quantile(values, .95)),
            "maximum": float(np.max(values))}


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    rows = read_unique(out / "energy_quarter_validation.jsonl")
    summary = []
    for wd in (0.25, 0.5):
        group = [row for row in rows if row["case"]["Wd"] == wd]
        item = {"Wd": wd, "sample_count": len(group), "convergence": {}}
        for nx in (4, 8):
            nx_item = {}
            for kbt in (0.005, 0.01):
                key = str(kbt)
                refined = np.asarray([row["result"][str(nx)]["refined_thermal"][key]
                                      for row in group])
                ultra = np.asarray([row["result"][str(nx)]["ultrafine_thermal"][key]
                                    for row in group])
                relative = np.abs(ultra - refined) / np.maximum(np.abs(ultra), 1e-30)
                nx_item[key] = {
                    "relative_error": stats(relative),
                    "refined_median": float(np.median(refined)),
                    "ultrafine_median": float(np.median(ultra)),
                    "median_shift_fraction": float(
                        abs(np.median(ultra) - np.median(refined))
                        / max(abs(np.median(ultra)), 1e-30)
                    ),
                }
            item["convergence"][str(nx)] = nx_item
        summary.append(item)
    assessment = {"record_count": len(rows), "expected_record_count": 20,
                  "refined_step": .0005, "ultrafine_step": .00025, "summary": summary}
    (out / "energy_quarter_validation_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, kbt in zip(axes, (0.005, 0.01)):
        x = np.arange(4)
        labels, first, second = [], [], []
        for wd in (0.25, 0.5):
            item = next(entry for entry in summary if entry["Wd"] == wd)
            for nx in (4, 8):
                labels.append(f"W={wd:g}\nNx={nx}")
                first.append(item["convergence"][str(nx)][str(kbt)]["refined_median"])
                second.append(item["convergence"][str(nx)][str(kbt)]["ultrafine_median"])
        width = 0.36
        ax.bar(x - width / 2, first, width, label=r"$\Delta E=5\times10^{-4}t$")
        ax.bar(x + width / 2, second, width, label=r"$\Delta E=2.5\times10^{-4}t$")
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_ylabel(r"subset median $G/(e^2/h)$")
        ax.set_title(rf"$k_BT/t={kbt:g}$")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "energy_grid_convergence.png", dpi=240)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
