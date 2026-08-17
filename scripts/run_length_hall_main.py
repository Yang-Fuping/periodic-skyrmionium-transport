"""Resumable longitudinal-length Hall campaign at the confirmed minigap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture


A = 18
R = 8.0
J = 5.0
NY = 2
E_CENTER = 1.0997714941836594
E_UPPER_EDGE = 1.1224
GAP_ENERGIES = tuple(sorted(set(
    np.linspace(1.05, 1.15, 21).tolist()
    + [1.065, 1.077143, 1.085, E_CENTER, 1.115, 1.1224, 1.15]
)))
NX_VALUES = (1, 2, 4, 8)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def central_cell_offset(nx: int) -> int:
    """Choose the left member of the central cell pair for even Nx."""
    return ((nx - 1) // 2) * A


def identifier(case: dict) -> str:
    return "|".join(str(case[key]) for key in sorted(case))


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_record(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def baseline_cases():
    for nx in NX_VALUES:
        for width in (4, 16):
            start = central_cell_offset(nx) + max(1, (A - width) // 2)
            for kind in KINDS:
                yield {
                    "scan": "length_baseline", "kind": kind, "Nx": nx,
                    "energy": E_CENTER, "probe_width": width,
                    "probe_start": start, "eta": 5e-10,
                }


def profile_cases():
    for nx in NX_VALUES:
        offset = central_cell_offset(nx)
        for start_in_cell in range(1, 16):
            yield {
                "scan": "length_local_profile", "kind": "skyrmionium_q_zero",
                "Nx": nx, "energy": E_CENTER, "probe_width": 2,
                "probe_start": offset + start_in_cell,
                "start_in_cell": start_in_cell, "eta": 5e-10,
            }


def mirror_profile_cases():
    for nx in (2, 4):
        offset = (nx // 2) * A
        for start_in_cell in range(1, 16):
            yield {
                "scan": "length_local_profile_right", "kind": "skyrmionium_q_zero",
                "Nx": nx, "energy": E_CENTER, "probe_width": 2,
                "probe_start": offset + start_in_cell,
                "start_in_cell": start_in_cell, "eta": 5e-10,
            }


def upper_edge_profile_cases():
    for nx in NX_VALUES:
        offset = central_cell_offset(nx)
        for start_in_cell in range(1, 16):
            yield {
                "scan": "upper_edge_local_profile", "kind": "skyrmionium_q_zero",
                "Nx": nx, "energy": E_UPPER_EDGE, "probe_width": 2,
                "probe_start": offset + start_in_cell,
                "start_in_cell": start_in_cell, "eta": 5e-10,
            }


def spectrum_cases():
    for nx in NX_VALUES:
        start = central_cell_offset(nx) + 7
        for energy in GAP_ENERGIES:
            for kind in KINDS:
                yield {
                    "scan": "length_gap_spectrum", "kind": kind, "Nx": nx,
                    "energy": energy, "probe_width": 4,
                    "probe_start": start, "eta": 5e-10,
                }


def upper_edge_refinement_cases():
    nx = 8
    start = central_cell_offset(nx) + 7
    for energy in np.arange(1.115, 1.1281, 0.001):
        for kind in KINDS:
            yield {
                "scan": "upper_edge_refinement", "kind": kind, "Nx": nx,
                "energy": float(energy), "probe_width": 4,
                "probe_start": start, "eta": 5e-10,
            }


def upper_edge_wide_cases():
    nx = 8
    width = 16
    start = central_cell_offset(nx) + 1
    for energy in np.arange(1.119, 1.1251, 0.001):
        for kind in KINDS:
            yield {
                "scan": "upper_edge_refinement_wide", "kind": kind, "Nx": nx,
                "energy": float(energy), "probe_width": width,
                "probe_start": start, "eta": 5e-10,
            }


def run(cases, out: Path):
    raw = out / "hall_cases.jsonl"
    records = read_records(raw)
    completed = {record["id"] for record in records}
    cases = list(cases)
    for position, case in enumerate(cases, 1):
        case_id = identifier(case)
        if case_id in completed:
            continue
        texture = make_array_texture(case["kind"], A, case["Nx"], NY, R)
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], case["probe_width"],
            probe_J=0.0, probe_start=case["probe_start"], longitudinal_J=J,
        )
        result = evaluate_hall_point(
            texture, case["energy"], J, 1.0, contacts, eta=case["eta"]
        )
        append_record(raw, {"id": case_id, "case": case, "result": result})
        completed.add(case_id)
        status = {
            "stored": len(completed), "position": position, "suite_total": len(cases),
            "Nx": case["Nx"], "kind": case["kind"],
            "valid": result["valid_hall_point"],
            "source_current_fraction": result["source_current_fraction"],
        }
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("baseline", "profile", "mirror", "upper-edge-profile", "spectrum", "refinement", "refinement-wide", "all"),
                        default="all")
    parser.add_argument("--output-label", default="length_hall_main_v1")
    args = parser.parse_args()
    out = ROOT / "results" / args.output_label
    out.mkdir(parents=True, exist_ok=True)
    if args.suite in {"baseline", "all"}:
        run(baseline_cases(), out)
    if args.suite in {"profile", "all"}:
        run(profile_cases(), out)
    if args.suite in {"mirror", "all"}:
        run(mirror_profile_cases(), out)
    if args.suite in {"upper-edge-profile", "all"}:
        run(upper_edge_profile_cases(), out)
    if args.suite in {"spectrum", "all"}:
        run(spectrum_cases(), out)
    if args.suite in {"refinement", "all"}:
        run(upper_edge_refinement_cases(), out)
    if args.suite in {"refinement-wide", "all"}:
        run(upper_edge_wide_cases(), out)


if __name__ == "__main__":
    main()
