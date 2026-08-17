"""Assess half-step energy-grid convergence for the disorder ensemble subset."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT


STRENGTHS = (0.25, 0.5)
KBT = (0.005, 0.01)
NX_VALUES = (4, 8)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return list({row["physical_id"]: row for row in rows}.values())


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    rows = read_jsonl(out / "energy_validation.jsonl")
    summary = []
    for wd in STRENGTHS:
        group = [row for row in rows if row["case"]["Wd"] == wd]
        item = {"Wd": wd, "sample_count": len(group), "convergence": {}}
        for nx in NX_VALUES:
            nx_result = {}
            for kbt in KBT:
                key = str(kbt)
                coarse = np.asarray([
                    row["result"][str(nx)]["coarse_thermal"][key] for row in group
                ])
                refined = np.asarray([
                    row["result"][str(nx)]["refined_thermal"][key] for row in group
                ])
                denominator = np.maximum(np.abs(refined), 1e-30)
                relative = np.abs(refined - coarse) / denominator
                signed = (coarse - refined) / denominator
                nx_result[key] = {
                    "relative_error": statistics(relative),
                    "signed_coarse_minus_refined": statistics(signed),
                    "coarse_median": float(np.median(coarse)),
                    "refined_median": float(np.median(refined)),
                    "median_shift_fraction": float(
                        abs(np.median(refined) - np.median(coarse))
                        / max(abs(np.median(refined)), 1e-30)
                    ),
                }
            item["convergence"][str(nx)] = nx_result
        summary.append(item)

    assessment = {
        "record_count": len(rows),
        "expected_record_count": 20,
        "coarse_step": 0.001,
        "refined_step": 0.0005,
        "summary": summary,
        "acceptance_guidance": (
            "Use the refined subset to qualify ensemble statistics. Median shifts below 5% "
            "are adequate for the central claim; large individual relative errors indicate "
            "narrow coherent resonances and must be reported rather than averaged away."
        ),
    }
    (out / "energy_validation_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
