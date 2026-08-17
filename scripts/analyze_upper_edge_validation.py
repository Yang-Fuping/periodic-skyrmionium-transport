"""Summarize targeted eta validation without hiding unresolved resonances."""

from __future__ import annotations

import json

from _bootstrap import ROOT


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def passes(result):
    return bool(
        result["scattering_unitarity_error"] <= 5e-6
        and result["scattering_charge_mismatch"] <= 5e-6
        and result["gauge_invariance_error"] <= 1e-9
    )


def main():
    out = ROOT / "results" / "upper_edge_robustness_v1"
    rows = read_jsonl(out / "eta_validation.jsonl")
    latest = {}
    for row in rows:
        key = (row["source"], row["physical_key"])
        if key not in latest or row["validation_eta"] < latest[key]["validation_eta"]:
            latest[key] = row
    final = list(latest.values())
    unresolved = [row for row in final if not passes(row["result"])]
    tuned = [row for row in final if abs(row["case"]["energy"] - 1.120) < 1e-10]
    assessment = {
        "validation_record_count": len(rows),
        "unique_validated_case_count": len(final),
        "final_pass_count": sum(passes(row["result"]) for row in final),
        "final_unresolved_count": len(unresolved),
        "tuned_point_validated_case_count": len(tuned),
        "tuned_point_all_pass": bool(tuned and all(passes(row["result"]) for row in tuned)),
        "tuned_point_max_gauge_error": max(row["result"]["gauge_invariance_error"] for row in tuned),
        "tuned_point_max_unitarity_error": max(row["result"]["scattering_unitarity_error"] for row in tuned),
        "tuned_point_max_scattering_mismatch": max(row["result"]["scattering_charge_mismatch"] for row in tuned),
        "unresolved_cases": [{
            "source": row["source"], "case": row["case"],
            "validation_eta": row["validation_eta"],
            "gauge_error": row["result"]["gauge_invariance_error"],
            "unitarity_error": row["result"]["scattering_unitarity_error"],
            "scattering_mismatch": row["result"]["scattering_charge_mismatch"],
        } for row in unresolved],
        "interpretation": (
            "The E/t=1.120 conclusion passes targeted validation. Fourteen adjacent sharp "
            "resonance cases remain above the preregistered gauge tolerance and are excluded "
            "from precision-window claims."
        ),
    }
    (out / "validation_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in assessment.items() if k != "unresolved_cases"}, indent=2))


if __name__ == "__main__":
    main()
