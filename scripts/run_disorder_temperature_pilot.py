"""Resumable paired Nx=4/8 disorder-temperature pilot at the minigap center."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.statistics import anderson_disorder, finite_temperature_average
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import paired_prefix_transmission


A = 18
R = 8.0
J = 5.0
NY = 2
E_CENTER = 1.0997714941836594
ENERGY = np.asarray(sorted(set(np.arange(1.02, 1.1800001, 0.001).tolist() + [E_CENTER])))
STRENGTHS = (0.05, 0.1, 0.25, 0.5, 1.0)
SAMPLES = 8
KBT = (0.005, 0.01)
BASE_SEED = 20260814


def identifier(strength, sample):
    return f"Wd={strength:g}|sample={sample}|seed={BASE_SEED}"


def run_pair(strength, sample):
    strength_index = STRENGTHS.index(strength)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, strength_index, sample]))
    texture8 = make_array_texture("skyrmionium_q_zero", A, 8, NY, R)
    disorder8 = anderson_disorder(texture8.shape[:2], strength, rng)
    paired = paired_prefix_transmission(
        texture8, (A * 4, A * 8), ENERGY, J, 1.0, eta=5e-10,
        onsite_disorder=disorder8,
    )
    result = {}
    center_index = int(np.argmin(np.abs(ENERGY - E_CENTER)))
    for nx in (4, 8):
        transmission = paired[A * nx]
        result[str(nx)] = {
            "transmission": transmission.tolist(),
            "T_center": float(transmission[center_index]),
            "thermal": {
                str(kbt): float(finite_temperature_average(
                    ENERGY, transmission, E_CENTER, kbt
                )) for kbt in KBT
            },
        }
    return {
        "id": identifier(strength, sample),
        "case": {"Wd": strength, "sample": sample, "seed": BASE_SEED},
        "energy": ENERGY.tolist(), "result": result,
    }


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    out = ROOT / "results" / "disorder_temperature_joint_v1"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "pilot.jsonl"
    existing = read_jsonl(raw)
    completed = {row["id"] for row in existing}
    jobs = [(strength, sample) for strength in STRENGTHS for sample in range(SAMPLES)
            if identifier(strength, sample) not in completed]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_pair, *job): job for job in jobs}
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({
                "completed_new": position, "new_total": len(jobs),
                "stored_total": len(completed) + position,
                "Wd": record["case"]["Wd"], "sample": record["case"]["sample"],
                "G8_kBT0.01": record["result"]["8"]["thermal"]["0.01"],
            }), flush=True)


if __name__ == "__main__":
    main()
