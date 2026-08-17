"""Clean convergence campaign for the Nx=8 upper-edge Hall window."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
NX = 8
A = 18
CELL_OFFSET = ((NX - 1) // 2) * A


def identifier(case):
    return "|".join(str(case[key]) for key in sorted(case))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def energy_cases():
    for energy in np.arange(1.1185, 1.12651, 0.0005):
        for width, start_in_cell in ((4, 7), (16, 1)):
            for kind in KINDS:
                yield {
                    "scan": "energy_half_step", "kind": kind, "Nx": NX, "Ny": 2,
                    "energy": float(energy), "probe_width": width,
                    "probe_start": CELL_OFFSET + start_in_cell,
                    "start_in_cell": start_in_cell, "eta": 5e-10,
                }


def ny_cases():
    for ny in (1, 2, 3, 4):
        for energy in (1.120, 1.1225, 1.125):
            for width, start_in_cell in ((4, 7), (16, 1)):
                for kind in KINDS:
                    yield {
                        "scan": "ny_scaling", "kind": kind, "Nx": NX, "Ny": ny,
                        "energy": energy, "probe_width": width,
                        "probe_start": CELL_OFFSET + start_in_cell,
                        "start_in_cell": start_in_cell, "eta": 5e-10,
                    }


def ny_refinement_cases():
    for ny in (1, 2, 3, 4):
        for energy in np.arange(1.119, 1.12151, 0.0005):
            for width, start_in_cell in ((4, 7), (16, 1)):
                for kind in KINDS:
                    yield {
                        "scan": "ny_refinement", "kind": kind, "Nx": NX, "Ny": ny,
                        "energy": float(energy), "probe_width": width,
                        "probe_start": CELL_OFFSET + start_in_cell,
                        "start_in_cell": start_in_cell, "eta": 5e-10,
                    }


def position_cases():
    for energy in (1.120, 1.1225, 1.125):
        for start_in_cell in (1, 4, 7, 10, 13):
            for kind in KINDS:
                yield {
                    "scan": "probe_position", "kind": kind, "Nx": NX, "Ny": 2,
                    "energy": energy, "probe_width": 4,
                    "probe_start": CELL_OFFSET + start_in_cell,
                    "start_in_cell": start_in_cell, "eta": 5e-10,
                }


def run(cases, out):
    raw = out / "clean_cases.jsonl"
    records = read_jsonl(raw)
    completed = {row["id"] for row in records}
    cases = list(cases)
    for index, case in enumerate(cases, 1):
        case_id = identifier(case)
        if case_id in completed:
            continue
        texture = make_array_texture(case["kind"], A, NX, case["Ny"], 8.0)
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], case["probe_width"],
            probe_J=0.0, probe_start=case["probe_start"], longitudinal_J=5.0,
        )
        result = evaluate_hall_point(
            texture, case["energy"], 5.0, 1.0, contacts, eta=case["eta"]
        )
        with raw.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": case_id, "case": case, "result": result}) + "\n")
        completed.add(case_id)
        print(json.dumps({
            "stored": len(completed), "position": index, "total": len(cases),
            "scan": case["scan"], "Ny": case["Ny"],
            "valid": result["valid_hall_point"],
        }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("energy", "ny", "ny-refinement", "position", "all"), default="all")
    args = parser.parse_args()
    out = ROOT / "results" / "upper_edge_robustness_v1"
    out.mkdir(parents=True, exist_ok=True)
    if args.suite in {"energy", "all"}:
        run(energy_cases(), out)
    if args.suite in {"ny", "all"}:
        run(ny_cases(), out)
    if args.suite in {"ny-refinement", "all"}:
        run(ny_refinement_cases(), out)
    if args.suite in {"position", "all"}:
        run(position_cases(), out)


if __name__ == "__main__":
    main()
