"""Refine the representative disorder subset from dE=0.0005 to 0.00025."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _bootstrap import ROOT
from run_disorder_temperature_pilot import BASE_SEED, STRENGTHS
from skyrmion_transport.statistics import anderson_disorder, finite_temperature_average
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import paired_prefix_transmission


QUARTER_ENERGY = np.sort(np.concatenate((
    np.arange(1.02025, 1.18, 0.001),
    np.arange(1.02075, 1.18, 0.001),
)))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_validation(validation_row, physical_row):
    wd = physical_row["case"]["Wd"]
    sample = physical_row["case"]["sample"]
    strength_index = STRENGTHS.index(wd)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, strength_index, sample]))
    texture8 = make_array_texture("skyrmionium_q_zero", 18, 8, 2, 8.0)
    disorder8 = anderson_disorder(texture8.shape[:2], wd, rng)
    paired_quarter = paired_prefix_transmission(
        texture8, (72, 144), QUARTER_ENERGY, 5.0, 1.0, eta=5e-10,
        onsite_disorder=disorder8,
    )
    base_energy = np.asarray(physical_row["energy"])
    midpoint_energy = np.arange(1.0205, 1.1800001, 0.001)
    result = {}
    for nx in (4, 8):
        base_t = np.asarray(physical_row["result"][str(nx)]["transmission"])
        midpoint_t = np.asarray(validation_row["result"][str(nx)]["midpoint_transmission"])
        combined_energy = np.concatenate((base_energy, midpoint_energy, QUARTER_ENERGY))
        combined_t = np.concatenate((base_t, midpoint_t, paired_quarter[18 * nx]))
        order = np.argsort(combined_energy)
        result[str(nx)] = {
            "quarter_transmission": paired_quarter[18 * nx].tolist(),
            "ultrafine_thermal": {
                str(kbt): float(finite_temperature_average(
                    combined_energy[order], combined_t[order], 1.0997714941836594, kbt
                )) for kbt in (0.005, 0.01)
            },
            "refined_thermal": validation_row["result"][str(nx)]["refined_thermal"],
        }
    return {"physical_id": physical_row["id"], "case": physical_row["case"], "result": result}


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    physical = {row["id"]: row for row in (
        read_jsonl(out / "pilot.jsonl") + read_jsonl(out / "final_extension.jsonl")
    )}
    validation = {row["physical_id"]: row for row in read_jsonl(out / "energy_validation.jsonl")}
    raw = out / "energy_quarter_validation.jsonl"
    existing = read_jsonl(raw)
    completed = {row["physical_id"] for row in existing}
    selected = [(row, physical[physical_id]) for physical_id, row in validation.items()
                if physical_id not in completed]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_validation, *job): job for job in selected}
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed": position, "total": len(selected),
                "Wd": record["case"]["Wd"], "sample": record["case"]["sample"],
                "ultrafine_G8_0.005": record["result"]["8"]["ultrafine_thermal"]["0.005"],
            }), flush=True)


if __name__ == "__main__":
    main()
