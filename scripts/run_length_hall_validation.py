"""Targeted smaller-eta validation for numerically demanding length-Hall points."""

from __future__ import annotations

import json

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def key(case):
    return "|".join(str(case[name]) for name in sorted(case) if name != "eta")


def main():
    out = ROOT / "results" / "length_hall_main_v1"
    physical = read_jsonl(out / "hall_cases.jsonl")
    raw = out / "eta_validation.jsonl"
    validation = read_jsonl(raw)
    completed = {(row["physical_id"], row["validation_eta"]) for row in validation}
    selected = [row for row in physical if (
        row["result"]["scattering_unitarity_error"] > 5e-6
        or row["result"]["scattering_charge_mismatch"] > 5e-6
        or row["result"]["gauge_invariance_error"] > 1e-9
    )]
    jobs = [(row, 1e-10) for row in selected]
    by_id = {row["id"]: row for row in physical}
    for checked in validation:
        result = checked["result"]
        if checked["validation_eta"] == 1e-10 and (
            result["scattering_unitarity_error"] > 5e-6
            or result["scattering_charge_mismatch"] > 5e-6
            or result["gauge_invariance_error"] > 1e-9
        ):
            jobs.append((by_id[checked["physical_id"]], 5e-11))
        if checked["validation_eta"] == 5e-11 and (
            result["scattering_unitarity_error"] > 5e-6
            or result["scattering_charge_mismatch"] > 5e-6
            or result["gauge_invariance_error"] > 1e-9
        ):
            jobs.append((by_id[checked["physical_id"]], 2.5e-11))
    for index, (row, eta) in enumerate(jobs, 1):
        case = row["case"]
        physical_id = row["id"]
        if (physical_id, eta) in completed:
            continue
        texture = make_array_texture(case["kind"], 18, case["Nx"], 2, 8.0)
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], case["probe_width"],
            probe_J=0.0, probe_start=case["probe_start"], longitudinal_J=5.0,
        )
        result = evaluate_hall_point(texture, case["energy"], 5.0, 1.0, contacts, eta=eta)
        record = {
            "physical_id": physical_id, "case": case, "physical_result": row["result"],
            "validation_eta": eta, "result": result,
        }
        with raw.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({
            "position": index, "total": len(jobs), "Nx": case["Nx"],
            "unitarity": result["scattering_unitarity_error"],
            "gauge": result["gauge_invariance_error"],
        }), flush=True)


if __name__ == "__main__":
    main()
