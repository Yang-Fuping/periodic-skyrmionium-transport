"""Paired Anderson-disorder Hall samples at the cross-Ny candidate point."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.statistics import anderson_disorder
from skyrmion_transport.textures import make_array_texture


KINDS = ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
STRENGTHS = (0.002, 0.005, 0.01)
SAMPLES = 50
BASE_SEED = 20260813


def identifier(case):
    return "|".join(str(case[key]) for key in sorted(case))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    out = ROOT / "results" / "upper_edge_robustness_v1"
    raw = out / "disorder_cases.jsonl"
    records = read_jsonl(raw)
    completed = {row["id"] for row in records}
    textures = {kind: make_array_texture(kind, 18, 8, 2, 8.0) for kind in KINDS}
    shape = textures[KINDS[0]].shape[:2]
    contacts = standard_four_contacts(
        shape[0], shape[1], 4, probe_J=0.0, probe_start=61, longitudinal_J=5.0,
    )
    total = len(STRENGTHS) * SAMPLES * len(KINDS)
    position = 0
    for wd_index, wd in enumerate(STRENGTHS):
        for sample in range(SAMPLES):
            rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, wd_index, sample]))
            disorder = anderson_disorder(shape, wd, rng)
            for kind in KINDS:
                position += 1
                case = {
                    "scan": "paired_disorder", "kind": kind, "Nx": 8, "Ny": 2,
                    "energy": 1.120, "probe_width": 4, "probe_start": 61,
                    "Wd": wd, "sample": sample, "seed": BASE_SEED,
                    "eta": 5e-10,
                }
                case_id = identifier(case)
                if case_id in completed:
                    continue
                result = evaluate_hall_point(
                    textures[kind], 1.120, 5.0, 1.0, contacts, eta=5e-10,
                    onsite_disorder=disorder,
                )
                with raw.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"id": case_id, "case": case, "result": result}) + "\n")
                completed.add(case_id)
                print(json.dumps({
                    "stored": len(completed), "position": position, "total": total,
                    "Wd": wd, "sample": sample, "kind": kind,
                    "valid": result["valid_hall_point"],
                }), flush=True)


if __name__ == "__main__":
    main()
