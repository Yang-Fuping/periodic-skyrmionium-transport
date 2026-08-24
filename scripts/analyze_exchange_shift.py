import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def candidate(j_value: float) -> dict:
    report = RESULTS / "gap_scan" / "skyrmionium_q_zero" / f"R8_J{j_value:g}_nk11" / "report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    return next(
        row for row in payload["cells"][0]["largest_interior_candidates"]
        if row["n_occ"] == 325
    )


def main() -> None:
    j_values = np.asarray([4.0, 4.25, 4.5, 4.75, 5.0])
    rows = []
    for j_value in j_values:
        gap = candidate(float(j_value))
        rows.append({
            "J_over_t": float(j_value),
            "midgap_over_t": float(gap["midgap_energy"]),
            "midgap_minus_J_over_t": float(gap["midgap_energy"] - j_value),
            "gap_over_t": float(gap["indirect_gap"]),
        })

    shifted = np.asarray([row["midgap_minus_J_over_t"] for row in rows])
    widths = np.asarray([row["gap_over_t"] for row in rows])
    assessment = {
        "parameters": {
            "A_over_a": 18,
            "R_over_a": 8,
            "n_occ": 325,
            "nk": 11,
        },
        "rows": rows,
        "shifted_center_range_over_t": float(np.ptp(shifted)),
        "gap_mean_over_t": float(np.mean(widths)),
        "gap_relative_range": float(np.ptp(widths) / np.mean(widths)),
        "interpretation": (
            "The positive minigap follows an almost rigid exchange shift over the "
            "sampled strong-coupling slice; this is not claimed as a separate "
            "nontrivial tuning mechanism."
        ),
    }
    output = RESULTS / "peer_review_exchange_shift"
    output.mkdir(parents=True, exist_ok=True)
    (output / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
