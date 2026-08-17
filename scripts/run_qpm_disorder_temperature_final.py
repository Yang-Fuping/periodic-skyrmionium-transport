"""Paired Q=+/-1 disorder-temperature comparison at the selected minigap point."""

from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _bootstrap import ROOT
from run_disorder_temperature_pilot import (
    A, BASE_SEED, ENERGY, E_CENTER, J, KBT, NY, R, STRENGTHS,
)
from skyrmion_transport.statistics import anderson_disorder, finite_temperature_average
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import paired_prefix_transmission


SELECTED_STRENGTHS = (0.25, 0.5)
KINDS = ("skyrmion_q_plus", "skyrmion_q_minus")
SAMPLES = 100
INDEPENDENT_QMINUS_SAMPLES = tuple(range(0, 100, 10))


def identifier(strength, sample):
    return f"Qpm|Wd={strength:g}|sample={sample}|seed={BASE_SEED}"


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_case(strength, sample):
    strength_index = STRENGTHS.index(strength)
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, strength_index, sample]))
    # This is exactly the field used by the matching Q=0 sample.
    shape = make_array_texture("skyrmion_q_plus", A, 8, NY, R).shape[:2]
    disorder8 = anderson_disorder(shape, strength, rng)
    center_index = int(np.argmin(np.abs(ENERGY - E_CENTER)))
    result = {}
    calculated_kinds = ("skyrmion_q_plus",)
    if sample in INDEPENDENT_QMINUS_SAMPLES:
        calculated_kinds += ("skyrmion_q_minus",)
    for kind in calculated_kinds:
        texture8 = make_array_texture(kind, A, 8, NY, R)
        paired = paired_prefix_transmission(
            texture8, (A * 4, A * 8), ENERGY, J, 1.0, eta=5e-10,
            onsite_disorder=disorder8,
        )
        kind_result = {}
        for nx in (4, 8):
            transmission = paired[A * nx]
            kind_result[str(nx)] = {
                "transmission": transmission.tolist(),
                "T_center": float(transmission[center_index]),
                "thermal": {
                    str(kbt): float(finite_temperature_average(
                        ENERGY, transmission, E_CENTER, kbt
                    )) for kbt in KBT
                },
            }
        result[kind] = kind_result
    qminus_mode = "independent" if "skyrmion_q_minus" in result else "q_reversal_derived"
    if qminus_mode == "q_reversal_derived":
        # For real scalar onsite disorder and two identical longitudinal leads,
        # Q reversal complex-conjugates the scattering problem and leaves the
        # two-terminal transmission invariant.  A preregistered 10% subset is
        # still calculated independently at every disorder strength.
        result["skyrmion_q_minus"] = deepcopy(result["skyrmion_q_plus"])
    return {
        "id": identifier(strength, sample),
        "case": {"Wd": strength, "sample": sample, "seed": BASE_SEED},
        "energy": ENERGY.tolist(), "result": result,
        "qminus_mode": qminus_mode,
    }


def main():
    out = ROOT / "results" / "disorder_topology_comparison_v1"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "qpm_cases.jsonl"
    existing = read_jsonl(raw)
    completed = {row["id"] for row in existing}
    jobs = [(wd, sample) for wd in SELECTED_STRENGTHS for sample in range(SAMPLES)
            if identifier(wd, sample) not in completed]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_case, *job): job for job in jobs}
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            qp = record["result"]["skyrmion_q_plus"]["8"]["thermal"]["0.01"]
            qm = record["result"]["skyrmion_q_minus"]["8"]["thermal"]["0.01"]
            print(json.dumps({
                "completed": position, "total": len(jobs),
                "Wd": record["case"]["Wd"], "sample": record["case"]["sample"],
                "G8_qplus": qp, "G8_qminus": qm, "qminus_mode": record["qminus_mode"],
            }), flush=True)


if __name__ == "__main__":
    main()
