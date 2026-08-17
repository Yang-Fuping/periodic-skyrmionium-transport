"""Diagnose coherent transverse-boundary oscillations near the minigap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import contact_spectral_maps, evaluate_hall_point
from skyrmion_transport.multiterminal import (
    four_terminal_observables,
    standard_four_contacts,
    transmission_matrix_sparse,
)
from skyrmion_transport.textures import make_array_texture


ENERGIES = (1.065, 1.0997714941836594, 1.15)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


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


def hall_cases():
    # Dense one-site width scan of Q=0 for the suspect energy and gap center.
    for energy in (ENERGIES[0], ENERGIES[1]):
        for py in range(0, 41):
            yield {
                "scan": "width_dense", "kind": "skyrmionium_q_zero",
                "energy": energy, "padding_y": py, "probe_width": 4,
                "probe_start": 7, "eta": 5e-10,
            }
        # A displaced probe every four sites checks that oscillations are not a
        # peculiarity of the centered longitudinal contact position.
        for py in range(0, 41, 4):
            yield {
                "scan": "width_probe_displaced", "kind": "skyrmionium_q_zero",
                "energy": energy, "padding_y": py, "probe_width": 4,
                "probe_start": 1, "eta": 5e-10,
            }
    # Continue to very wide samples at a coarser step.
    for energy in ENERGIES:
        for py in range(0, 61, 4):
            yield {
                "scan": "width_extended", "kind": "skyrmionium_q_zero",
                "energy": energy, "padding_y": py, "probe_width": 4,
                "probe_start": 7, "eta": 5e-10,
            }
        for py in (0, 20, 40, 60):
            yield {
                "scan": "uniform_control", "kind": "uniform", "energy": energy,
                "padding_y": py, "probe_width": 4,
                "probe_start": 7, "eta": 5e-10,
            }
    # Separate local-probe coupling from transverse width.
    for energy in ENERGIES:
        for py in (0, 12, 30, 60):
            for width in (2, 4, 8, 16):
                for kind in ("skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus"):
                    start = max(1, (18 - width) // 2)
                    yield {
                        "scan": "probe_matrix", "kind": kind, "energy": energy,
                        "padding_y": py, "probe_width": width,
                        "probe_start": start, "eta": 5e-10,
                    }
    # True array-width scaling: increase the number of magnetic rows without
    # opening clean uniform bypass regions.
    for energy in ENERGIES:
        for Ny in range(1, 9):
            for kind in KINDS:
                yield {
                    "scan": "array_row_scaling", "kind": kind,
                    "energy": energy, "Ny": Ny, "padding_y": 0,
                    "probe_width": 4, "probe_start": 7, "eta": 5e-10,
                }
    # Test whether the local outer-inner-outer Hall pattern survives at fixed
    # texture density as the transverse number of magnetic cells grows.
    for energy in ENERGIES[:2]:
        for Ny in (2, 4, 6, 8):
            for start in range(1, 16):
                yield {
                    "scan": "array_row_position", "kind": "skyrmionium_q_zero",
                    "energy": energy, "Ny": Ny, "padding_y": 0,
                    "probe_width": 2, "probe_start": start, "eta": 5e-10,
                }


def run_hall(out: Path):
    raw = out / "hall_cases.jsonl"
    records = read_records(raw)
    completed = {record["id"] for record in records}
    cases = list(hall_cases())
    for position, case in enumerate(cases, 1):
        case_id = identifier(case)
        if case_id in completed:
            continue
        texture = make_array_texture(
            case["kind"], 18, 1, case.get("Ny", 2), 8,
            padding=(0, case["padding_y"]),
        )
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], case["probe_width"],
            probe_J=0.0, probe_start=case["probe_start"], longitudinal_J=5.0,
        )
        result = evaluate_hall_point(
            texture, case["energy"], 5.0, 1.0, contacts, eta=case["eta"]
        )
        record = {"id": case_id, "case": case, "result": result}
        append_record(raw, record)
        completed.add(case_id)
        status = {"completed": len(completed), "position": position, "total": len(cases)}
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)


def run_maps(out: Path):
    map_dir = out / "spectral_maps"
    map_dir.mkdir(exist_ok=True)
    # Representative extrema and converged-width controls from the dense scan.
    cases = []
    for energy in (ENERGIES[0], ENERGIES[1]):
        for py in (0, 6, 12, 24, 30, 40, 60):
            for kind in ("uniform", "skyrmionium_q_zero"):
                cases.append((energy, py, kind))
    for energy, py, kind in cases:
        path = map_dir / f"{kind}_E{energy:.9f}_py{py}.npz"
        if path.exists():
            continue
        texture = make_array_texture(kind, 18, 1, 2, 8, padding=(0, py))
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], 4,
            probe_J=0.0, probe_start=7, longitudinal_J=5.0,
        )
        transmission, diagnostics = transmission_matrix_sparse(
            texture, energy, 5.0, 1.0, contacts, eta=5e-10
        )
        lb = four_terminal_observables(transmission)
        maps = contact_spectral_maps(diagnostics, texture.shape[:2], source_lead=0)
        np.savez_compressed(
            path, texture=texture, transmission=transmission,
            voltages=lb["voltages"], **maps,
        )
        print(json.dumps({"saved_map": str(path)}), flush=True)
    row_dir = out / "row_scaling_maps"
    row_dir.mkdir(exist_ok=True)
    for energy in (ENERGIES[0], ENERGIES[1]):
        for Ny in (1, 2, 4, 6, 8):
            path = row_dir / f"skyrmionium_q_zero_E{energy:.9f}_Ny{Ny}.npz"
            if path.exists():
                continue
            texture = make_array_texture("skyrmionium_q_zero", 18, 1, Ny, 8)
            contacts = standard_four_contacts(
                texture.shape[0], texture.shape[1], 4,
                probe_J=0.0, probe_start=7, longitudinal_J=5.0,
            )
            transmission, diagnostics = transmission_matrix_sparse(
                texture, energy, 5.0, 1.0, contacts, eta=5e-10
            )
            lb = four_terminal_observables(transmission)
            maps = contact_spectral_maps(diagnostics, texture.shape[:2], source_lead=0)
            np.savez_compressed(
                path, texture=texture, transmission=transmission,
                voltages=lb["voltages"], **maps,
            )
            print(json.dumps({"saved_row_map": str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("hall", "maps", "all"), default="all")
    parser.add_argument("--output-label", default="boundary_coherence_v1")
    args = parser.parse_args()
    out = ROOT / "results" / args.output_label
    out.mkdir(parents=True, exist_ok=True)
    if args.suite in {"hall", "all"}:
        run_hall(out)
    if args.suite in {"maps", "all"}:
        run_maps(out)


if __name__ == "__main__":
    main()
