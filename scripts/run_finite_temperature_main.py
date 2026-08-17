"""High-resolution spectra for the finite-temperature main-text assessment."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import clean_lead_modes, two_terminal_transmission


A = 18
R = 8.0
J = 5.0
NX = 8
NY = 2
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
LONGITUDINAL_ENERGY = np.arange(1.02, 1.1800001, 0.0005)
HALL_ENERGY = np.arange(1.10, 1.1400001, 0.0005)
HALL_ENERGY_HALF = np.arange(1.10025, 1.1400001, 0.0005)
LONGITUDINAL_ENERGY_HALF = np.arange(1.02025, 1.1800001, 0.0005)
HALL_ENERGY_QUARTER = np.arange(1.100125, 1.1400001, 0.00025)


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_longitudinal(out):
    for kind in KINDS:
        target = out / f"longitudinal_{kind}.npz"
        if target.exists():
            with np.load(target) as saved:
                if np.array_equal(saved["energy"], LONGITUDINAL_ENERGY):
                    print(json.dumps({"suite": "longitudinal", "kind": kind, "status": "reused"}), flush=True)
                    continue
        texture = make_array_texture(kind, A, NX, NY, R)
        transmission = two_terminal_transmission(
            texture, LONGITUDINAL_ENERGY, J, 1.0, eta=5e-10,
        )
        channels = clean_lead_modes(LONGITUDINAL_ENERGY, texture.shape[1], J, 1.0)
        np.savez_compressed(
            target, energy=LONGITUDINAL_ENERGY, transmission=transmission,
            lead_channels=channels, eta=5e-10,
        )
        print(json.dumps({
            "suite": "longitudinal", "kind": kind, "status": "stored",
            "points": len(LONGITUDINAL_ENERGY), "min_T": float(transmission.min()),
            "max_T": float(transmission.max()),
        }), flush=True)


def run_longitudinal_half_step(out):
    for kind in KINDS:
        target = out / f"longitudinal_half_step_{kind}.npz"
        if target.exists():
            with np.load(target) as saved:
                if np.array_equal(saved["energy"], LONGITUDINAL_ENERGY_HALF):
                    print(json.dumps({"suite": "longitudinal-half-step", "kind": kind, "status": "reused"}), flush=True)
                    continue
        texture = make_array_texture(kind, A, NX, NY, R)
        transmission = two_terminal_transmission(
            texture, LONGITUDINAL_ENERGY_HALF, J, 1.0, eta=5e-10,
        )
        channels = clean_lead_modes(LONGITUDINAL_ENERGY_HALF, texture.shape[1], J, 1.0)
        np.savez_compressed(
            target, energy=LONGITUDINAL_ENERGY_HALF, transmission=transmission,
            lead_channels=channels, eta=5e-10,
        )
        print(json.dumps({
            "suite": "longitudinal-half-step", "kind": kind, "status": "stored",
            "points": len(LONGITUDINAL_ENERGY_HALF), "min_T": float(transmission.min()),
            "max_T": float(transmission.max()),
        }), flush=True)


def run_hall(out):
    raw = out / "hall_spectrum.jsonl"
    records = read_jsonl(raw)
    completed = {row["id"] for row in records}
    textures = {kind: make_array_texture(kind, A, NX, NY, R) for kind in KINDS}
    shape = textures[KINDS[0]].shape[:2]
    contacts = standard_four_contacts(
        shape[0], shape[1], 4, probe_J=0.0, probe_start=61, longitudinal_J=J,
    )
    total = len(KINDS) * len(HALL_ENERGY)
    position = 0
    for energy in HALL_ENERGY:
        for kind in KINDS:
            position += 1
            case_id = f"{kind}|{energy:.10f}|1e-10"
            if case_id in completed:
                continue
            result = evaluate_hall_point(
                textures[kind], float(energy), J, 1.0, contacts, eta=1e-10,
            )
            record = {
                "id": case_id,
                "case": {
                    "kind": kind, "energy": float(energy), "Nx": NX, "Ny": NY,
                    "probe_width": 4, "probe_start": 61, "eta": 1e-10,
                },
                "result": result,
            }
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed.add(case_id)
            print(json.dumps({
                "suite": "hall", "position": position, "total": total,
                "kind": kind, "energy": float(energy),
                "valid": result["valid_hall_point"],
                "gauge": result["gauge_invariance_error"],
            }), flush=True)


def run_hall_half_step(out):
    """Fill the midpoint grid so the merged Hall spectrum has dE=0.00025t."""
    raw = out / "hall_spectrum_half_step.jsonl"
    records = read_jsonl(raw)
    completed = {row["id"] for row in records}
    textures = {kind: make_array_texture(kind, A, NX, NY, R) for kind in KINDS}
    shape = textures[KINDS[0]].shape[:2]
    contacts = standard_four_contacts(
        shape[0], shape[1], 4, probe_J=0.0, probe_start=61, longitudinal_J=J,
    )
    total = len(KINDS) * len(HALL_ENERGY_HALF)
    position = 0
    for energy in HALL_ENERGY_HALF:
        for kind in KINDS:
            position += 1
            case_id = f"{kind}|{energy:.10f}|1e-10"
            if case_id in completed:
                continue
            result = evaluate_hall_point(
                textures[kind], float(energy), J, 1.0, contacts, eta=1e-10,
            )
            record = {
                "id": case_id,
                "case": {
                    "kind": kind, "energy": float(energy), "Nx": NX, "Ny": NY,
                    "probe_width": 4, "probe_start": 61, "eta": 1e-10,
                },
                "result": result,
            }
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed.add(case_id)
            print(json.dumps({
                "suite": "hall-half-step", "position": position, "total": total,
                "kind": kind, "energy": float(energy),
                "valid": result["valid_hall_point"],
                "gauge": result["gauge_invariance_error"],
            }), flush=True)


def run_hall_quarter_step(out):
    """Fill remaining midpoints so the merged Hall spectrum has dE=0.000125t."""
    raw = out / "hall_spectrum_quarter_step.jsonl"
    records = read_jsonl(raw)
    completed = {row["id"] for row in records}
    textures = {kind: make_array_texture(kind, A, NX, NY, R) for kind in KINDS}
    shape = textures[KINDS[0]].shape[:2]
    contacts = standard_four_contacts(
        shape[0], shape[1], 4, probe_J=0.0, probe_start=61, longitudinal_J=J,
    )
    total = len(KINDS) * len(HALL_ENERGY_QUARTER)
    position = 0
    for energy in HALL_ENERGY_QUARTER:
        for kind in KINDS:
            position += 1
            case_id = f"{kind}|{energy:.10f}|1e-10"
            if case_id in completed:
                continue
            result = evaluate_hall_point(
                textures[kind], float(energy), J, 1.0, contacts, eta=1e-10,
            )
            record = {
                "id": case_id,
                "case": {
                    "kind": kind, "energy": float(energy), "Nx": NX, "Ny": NY,
                    "probe_width": 4, "probe_start": 61, "eta": 1e-10,
                },
                "result": result,
            }
            with raw.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed.add(case_id)
            print(json.dumps({
                "suite": "hall-quarter-step", "position": position, "total": total,
                "kind": kind, "energy": float(energy),
                "valid": result["valid_hall_point"],
                "gauge": result["gauge_invariance_error"],
            }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=(
        "longitudinal", "longitudinal-half-step", "hall", "hall-half-step",
        "hall-quarter-step", "all",
    ), default="all")
    args = parser.parse_args()
    out = ROOT / "results" / "finite_temperature_main_v1"
    out.mkdir(parents=True, exist_ok=True)
    if args.suite in {"longitudinal", "all"}:
        run_longitudinal(out)
    if args.suite in {"longitudinal-half-step", "all"}:
        run_longitudinal_half_step(out)
    if args.suite in {"hall", "all"}:
        run_hall(out)
    if args.suite in {"hall-half-step", "all"}:
        run_hall_half_step(out)
    if args.suite in {"hall-quarter-step", "all"}:
        run_hall_quarter_step(out)


if __name__ == "__main__":
    main()
