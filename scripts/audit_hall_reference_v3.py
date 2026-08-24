"""Recover the exact fixed-geometry Hall statistic reported in the paper.

This audit distinguishes the original three-energy, maximum-width reference
from the later fixed-energy scan over every integer probe width.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT


KINDS = ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    source = ROOT / "results" / "hall_mechanism_v1" / "raw_cases.jsonl"
    output = ROOT / "results" / "peer_review_v3"
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(source)

    selected = [
        row for row in records
        if row["case"]["scan"] == "probe_width"
        and row["case"]["probe_width"] == 16
        and row["case"]["probe_start"] == 1
        and row["case"]["kind"] in KINDS
    ]
    energies = sorted({float(row["result"]["energy"]) for row in selected})
    rows_by_key = {
        (row["case"]["kind"], float(row["result"]["energy"])): row
        for row in selected
    }

    ratios = []
    points = []
    for energy in energies:
        group = {kind: rows_by_key[(kind, energy)] for kind in KINDS}
        valid = all(row["result"]["valid_hall_point"] for row in group.values())
        q0 = abs(group["skyrmionium_q_zero"]["result"]["charge_hall_angle"])
        qp = abs(group["skyrmion_q_plus"]["result"]["charge_hall_angle"])
        qm = abs(group["skyrmion_q_minus"]["result"]["charge_hall_angle"])
        denominator = 0.5 * (qp + qm)
        ratio = q0 / denominator if valid and denominator > 0 else float("nan")
        if np.isfinite(ratio):
            ratios.append(ratio)
        points.append({
            "energy_over_t": energy,
            "all_three_textures_valid": valid,
            "q0_charge_hall_angle": q0,
            "qplus_abs_charge_hall_angle": qp,
            "qminus_abs_charge_hall_angle": qm,
            "compensation_ratio": ratio,
            "source_current_fraction": {
                kind: group[kind]["result"]["source_current_fraction"]
                for kind in KINDS
            },
        })

    reference = selected[0]["case"]
    assessment = {
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "statistic_type": "fixed_geometry_three_energy_median",
        "not_a_continuous_energy_scan": True,
        "geometry": {
            key: reference[key]
            for key in ("A", "R", "Nx", "Ny", "J", "probe_width",
                        "probe_start", "padding_x", "padding_y", "eta")
        },
        "contact_model": {
            "longitudinal_leads": "+z ferromagnetic, J_lead/t=5",
            "side_probes": "normal metal, J_probe=0",
            "device_lead_interface_hopping_over_t": 1.0,
            "lead_internal_hopping_over_t": 1.0,
        },
        "validity_criterion": {
            "minimum_absolute_source_current": 1e-8,
            "minimum_source_current_per_source_channel": 1e-6,
            "applied_to_all_three_textures": True,
        },
        "requested_energy_point_count": len(energies),
        "valid_energy_triple_count": len(ratios),
        "points": points,
        "compensation_ratios": ratios,
        "median_compensation_ratio": float(np.median(ratios)),
    }
    path = output / "hall_reference_metadata.json"
    path.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(path)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
