"""Quantify paired-disorder robustness of the tuned Nx=8 Hall point."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


KINDS = ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def paired_samples(rows):
    grouped = {}
    for row in rows:
        case = row["case"]
        grouped.setdefault((case["Wd"], case["sample"]), {})[case["kind"]] = row["result"]
    samples = []
    for (strength, sample), values in sorted(grouped.items()):
        if not all(kind in values for kind in KINDS):
            continue
        q0 = values["skyrmionium_q_zero"]
        qp = values["skyrmion_q_plus"]
        qm = values["skyrmion_q_minus"]
        denominator = 0.5 * (abs(qp["charge_hall_angle"]) + abs(qm["charge_hall_angle"]))
        ratio = abs(q0["charge_hall_angle"]) / denominator if denominator else np.nan
        samples.append({
            "Wd": strength,
            "sample": sample,
            "all_valid": bool(q0["valid_hall_point"] and qp["valid_hall_point"] and qm["valid_hall_point"]),
            "compensation_ratio": ratio,
            "passes_ratio_0p1": bool(np.isfinite(ratio) and ratio < 0.1),
            "q0_hall_angle": q0["charge_hall_angle"],
            "qplus_hall_angle": qp["charge_hall_angle"],
            "qminus_hall_angle": qm["charge_hall_angle"],
            "q_reversal_error": abs(qp["charge_hall_angle"] + qm["charge_hall_angle"]),
            "q0_current_fraction": q0["source_current_fraction"],
        })
    return samples


def summarize(samples):
    output = []
    for strength in sorted({row["Wd"] for row in samples}):
        rows = [row for row in samples if row["Wd"] == strength]
        valid = [row for row in rows if row["all_valid"] and np.isfinite(row["compensation_ratio"])]
        ratios = np.asarray([row["compensation_ratio"] for row in valid])
        q0 = np.asarray([row["q0_hall_angle"] for row in valid])
        currents = np.asarray([row["q0_current_fraction"] for row in valid])
        reversal = np.asarray([row["q_reversal_error"] for row in valid])
        output.append({
            "Wd": strength,
            "sample_count": len(rows),
            "valid_sample_count": len(valid),
            "median_compensation_ratio": float(np.median(ratios)),
            "mean_compensation_ratio": float(np.mean(ratios)),
            "q10_compensation_ratio": float(np.quantile(ratios, 0.1)),
            "q90_compensation_ratio": float(np.quantile(ratios, 0.9)),
            "fraction_ratio_below_0p1": float(np.mean(ratios < 0.1)),
            "signed_q0_mean": float(np.mean(q0)),
            "signed_q0_sem": float(np.std(q0, ddof=1) / np.sqrt(len(q0))),
            "mean_abs_q0": float(np.mean(np.abs(q0))),
            "median_q_reversal_error": float(np.median(reversal)),
            "median_current_fraction": float(np.median(currents)),
            "min_current_fraction": float(np.min(currents)),
            # A deliberately strict descriptive gate: both a typical-sample and
            # a population requirement must hold.  Signed ensemble cancellation
            # is not used in this decision.
            "robust_by_strict_gate": bool(np.median(ratios) < 0.1 and np.mean(ratios < 0.1) >= 0.8),
        })
    return output


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


def plot(samples, summary, out):
    strengths = [row["Wd"] for row in summary]
    data = [[s["compensation_ratio"] for s in samples if s["Wd"] == strength and s["all_valid"]]
            for strength in strengths]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].boxplot(data, tick_labels=[f"{x:g}" for x in strengths], showfliers=True)
    axes[0].axhline(0.1, color="tab:red", linestyle=":", label="compensation criterion")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Anderson disorder $W_d/t$", ylabel="per-sample compensation ratio",
                title="paired-disorder distribution")
    axes[0].legend(fontsize=8)
    fractions = [row["fraction_ratio_below_0p1"] for row in summary]
    axes[1].plot(strengths, fractions, "o-", color="tab:purple")
    axes[1].axhline(0.8, color="tab:red", linestyle=":", label="strict population gate")
    axes[1].set_ylim(0, 1.02)
    axes[1].set(xlabel="Anderson disorder $W_d/t$", ylabel="fraction with ratio < 0.1",
                title="sample-level survival fraction")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out / "disorder_compensation_distribution.png", dpi=240)
    plt.close(fig)


def main():
    out = ROOT / "results" / "upper_edge_robustness_v1"
    rows = read_jsonl(out / "disorder_cases.jsonl")
    samples = paired_samples(rows)
    summary = summarize(samples)
    assessment = {
        "physical_record_count": len(rows),
        "paired_sample_count": len(samples),
        "decision_rule": "median ratio < 0.1 and at least 80% of samples have ratio < 0.1",
        "summary": summary,
        "samples": samples,
        "quality": quality(rows),
    }
    (out / "disorder_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    plot(samples, summary, out)
    print(json.dumps({"summary": summary, "quality": assessment["quality"]}, indent=2))


if __name__ == "__main__":
    main()
