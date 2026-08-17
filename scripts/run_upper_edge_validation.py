"""Targeted smaller-eta checks for upper-edge clean and disorder results."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.statistics import anderson_disorder
from skyrmion_transport.textures import make_array_texture


BASE_SEED = 20260813
STRENGTHS = (0.002, 0.005, 0.01)
ETAS = (1e-10, 5e-11, 2.5e-11)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fails(result):
    return bool(
        result["scattering_unitarity_error"] > 5e-6
        or result["scattering_charge_mismatch"] > 5e-6
        or result["gauge_invariance_error"] > 1e-9
    )


def physical_key(case):
    return "|".join(str(case.get(name)) for name in
                    ("kind", "Nx", "Ny", "energy", "probe_width", "probe_start", "Wd", "sample"))


def select_jobs(clean, disorder):
    # Deduplicate calculations repeated in more than one clean scan.
    selected = {}
    for row in clean:
        if fails(row["result"]):
            selected.setdefault(("clean", physical_key(row["case"])), row)

    # Validate the three numerically hardest disorder cases per strength.  This
    # checks the quality envelope without silently reusing ensemble cancellation.
    for strength in STRENGTHS:
        candidates = [row for row in disorder if row["case"]["Wd"] == strength]
        candidates.sort(key=lambda row: max(
            row["result"]["scattering_unitarity_error"],
            row["result"]["scattering_charge_mismatch"],
            row["result"]["gauge_invariance_error"],
        ), reverse=True)
        for row in candidates[:3]:
            selected[("disorder", physical_key(row["case"]))] = row
    return [(source, row) for (source, _), row in selected.items()]


def reproduce_disorder(case, shape):
    strength_index = STRENGTHS.index(case["Wd"])
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, strength_index, case["sample"]]))
    return anderson_disorder(shape, case["Wd"], rng)


def main():
    out = ROOT / "results" / "upper_edge_robustness_v1"
    clean = read_jsonl(out / "clean_cases.jsonl")
    disorder = read_jsonl(out / "disorder_cases.jsonl")
    raw = out / "eta_validation.jsonl"
    stored = read_jsonl(raw)
    completed = {(row["source"], row["physical_key"], row["validation_eta"]) for row in stored}
    base_jobs = select_jobs(clean, disorder)

    for eta in ETAS:
        if eta == ETAS[0]:
            jobs = base_jobs
        else:
            previous_eta = ETAS[ETAS.index(eta) - 1]
            failed_keys = {
                (row["source"], row["physical_key"])
                for row in stored if row["validation_eta"] == previous_eta and fails(row["result"])
            }
            jobs = [(source, row) for source, row in base_jobs
                    if (source, physical_key(row["case"])) in failed_keys]

        for index, (source, row) in enumerate(jobs, 1):
            case = row["case"]
            pkey = physical_key(case)
            if (source, pkey, eta) in completed:
                continue
            texture = make_array_texture(case["kind"], 18, case["Nx"], case["Ny"], 8.0)
            contacts = standard_four_contacts(
                texture.shape[0], texture.shape[1], case["probe_width"],
                probe_J=0.0, probe_start=case["probe_start"], longitudinal_J=5.0,
            )
            onsite = reproduce_disorder(case, texture.shape[:2]) if source == "disorder" else None
            result = evaluate_hall_point(
                texture, case["energy"], 5.0, 1.0, contacts, eta=eta,
                onsite_disorder=onsite,
            )
            record = {
                "source": source, "physical_key": pkey, "case": case,
                "physical_result": row["result"], "validation_eta": eta, "result": result,
            }
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            stored.append(record)
            completed.add((source, pkey, eta))
            print(json.dumps({
                "eta": eta, "position": index, "total": len(jobs), "source": source,
                "kind": case["kind"], "E": case["energy"], "Ny": case["Ny"],
                "unitarity": result["scattering_unitarity_error"],
                "mismatch": result["scattering_charge_mismatch"],
                "gauge": result["gauge_invariance_error"], "passes": not fails(result),
            }), flush=True)


if __name__ == "__main__":
    main()
