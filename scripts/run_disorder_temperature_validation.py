"""Half-step energy-grid validation for representative final ensemble samples."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _bootstrap import ROOT
from run_disorder_temperature_pilot import BASE_SEED, ENERGY, STRENGTHS
from skyrmion_transport.statistics import anderson_disorder, finite_temperature_average
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import paired_prefix_transmission


SELECTED_STRENGTHS = (0.25, 0.5)
SELECTED_SAMPLES = tuple(range(0, 100, 10))
MIDPOINT_ENERGY = np.arange(1.0205, 1.1800001, 0.001)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def all_physical_rows(out):
    rows = read_jsonl(out / "pilot.jsonl") + read_jsonl(out / "final_extension.jsonl")
    return {row["id"]: row for row in rows}


def run_validation(row):
    wd = row["case"]["Wd"]
    sample = row["case"]["sample"]
    strength_index = STRENGTHS.index(wd)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, strength_index, sample]))
    texture8 = make_array_texture("skyrmionium_q_zero", 18, 8, 2, 8.0)
    disorder8 = anderson_disorder(texture8.shape[:2], wd, rng)
    paired_midpoint = paired_prefix_transmission(
        texture8, (18 * 4, 18 * 8), MIDPOINT_ENERGY, 5.0, 1.0, eta=5e-10,
        onsite_disorder=disorder8,
    )
    result = {}
    base_energy = np.asarray(row["energy"])
    for nx in (4, 8):
        midpoint_t = paired_midpoint[18 * nx]
        combined_energy = np.concatenate((base_energy, MIDPOINT_ENERGY))
        combined_t = np.concatenate((np.asarray(row["result"][str(nx)]["transmission"]), midpoint_t))
        order = np.argsort(combined_energy)
        combined_energy, combined_t = combined_energy[order], combined_t[order]
        result[str(nx)] = {
            "midpoint_transmission": midpoint_t.tolist(),
            "refined_thermal": {
                str(kbt): float(finite_temperature_average(
                    combined_energy, combined_t, 1.0997714941836594, kbt
                )) for kbt in (0.005, 0.01)
            },
            "coarse_thermal": row["result"][str(nx)]["thermal"],
        }
    return {"physical_id": row["id"], "case": row["case"], "result": result}


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    physical = all_physical_rows(out)
    raw = out / "energy_validation.jsonl"
    existing = read_jsonl(raw)
    completed = {row["physical_id"] for row in existing}
    selected = [row for row in physical.values()
                if row["case"]["Wd"] in SELECTED_STRENGTHS
                and row["case"]["sample"] in SELECTED_SAMPLES
                and row["id"] not in completed]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_validation, row): row for row in selected}
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed": position, "total": len(selected),
                "Wd": record["case"]["Wd"], "sample": record["case"]["sample"],
                "refined_G8_0.01": record["result"]["8"]["refined_thermal"]["0.01"],
            }), flush=True)


if __name__ == "__main__":
    main()
