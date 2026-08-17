"""Resumable production scans for Hall compensation and spin Hall transport."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import (
    make_array_texture,
    make_skyrmionium_wall_array,
)


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
STRONG_ENERGIES = (1.065, 1.0997714941836594, 1.15)


@dataclass(frozen=True)
class Case:
    scan: str
    kind: str
    A: int = 18
    R: float = 8.0
    Nx: int = 1
    Ny: int = 2
    J: float = 5.0
    probe_width: int = 4
    probe_start: int | None = 7
    padding_x: int = 0
    padding_y: int = 0
    eta: float = 1e-8
    energies: tuple[float, ...] = STRONG_ENERGIES

    def identifier(self, energy: float) -> str:
        fields = (
            self.scan, self.kind, self.A, self.R, self.Nx, self.Ny, self.J,
            self.probe_width, self.probe_start, self.padding_x, self.padding_y,
            self.eta, float(energy),
        )
        return "|".join(str(value) for value in fields)


def strong_cases(eta: float) -> list[Case]:
    cases: list[Case] = []
    for kind in KINDS:
        for start in range(1, 16):
            cases.append(Case(
                "position_w2", kind, probe_width=2, probe_start=start, eta=eta
            ))
        for width in (2, 4, 8, 16):
            start = (18 - width) // 2
            cases.append(Case(
                "probe_width", kind, probe_width=width, probe_start=start, eta=eta
            ))
        for padding_y in (0, 6, 12, 18):
            cases.append(Case(
                "transverse_padding", kind, padding_y=padding_y, eta=eta
            ))
        if kind == "skyrmionium_q_zero":
            for padding_y in (24, 30):
                cases.append(Case(
                    "transverse_padding_extended", kind,
                    padding_y=padding_y, eta=eta,
                ))
        for start in (7, 16, 25):
            cases.append(Case(
                "nx2_position", kind, Nx=2, probe_width=4,
                probe_start=start, eta=eta,
            ))
        cases.append(Case(
            "nx2_full_width", kind, Nx=2, probe_width=34,
            probe_start=1, eta=eta,
        ))
    for kind in (
        "skyrmionium_inner_wall", "skyrmionium_outer_wall",
        "skyrmionium_q_zero",
    ):
        cases.append(Case("wall_counterfactual", kind, eta=eta))
    return cases


def mixed_cases(eta: float) -> list[Case]:
    cases: list[Case] = []
    grids = (
        (1.5, tuple(np.linspace(-2.4, 2.4, 61))),
        (3.0, tuple(np.linspace(-0.9, 0.9, 61))),
    )
    for J, energies in grids:
        for kind in KINDS:
            cases.append(Case(
                "mixed_spin_coarse", kind, J=J, eta=eta, energies=energies
            ))
    return cases


def _contiguous_candidate_windows(rows: list[dict]) -> list[tuple[float, float]]:
    rows = sorted(rows, key=lambda row: row["result"]["energy"])
    selected = []
    for row in rows:
        result = row["result"]
        charge = abs(result["charge_hall_angle"])
        spin = abs(result["spin_hall_angle"])
        if (result["valid_hall_point"] and spin >= 0.01
                and charge <= min(0.005, 0.1 * spin)):
            selected.append(float(result["energy"]))
    if not selected:
        return []
    steps = np.diff(sorted({float(row["result"]["energy"]) for row in rows}))
    nominal_step = float(np.median(steps)) if len(steps) else 0.0
    groups = [[selected[0]]]
    for energy in selected[1:]:
        if nominal_step and energy - groups[-1][-1] <= 1.5 * nominal_step:
            groups[-1].append(energy)
        else:
            groups.append([energy])
    return [
        (group[0], group[-1]) for group in groups
        if len(group) >= 2 and group[-1] - group[0] >= 0.02 - 1e-12
    ]


def refinement_cases(records: list[dict]) -> list[Case]:
    cases: list[Case] = []
    grouped: dict[tuple[float, str], list[dict]] = {}
    for record in records:
        case = record["case"]
        if case["scan"] == "mixed_spin_coarse":
            grouped.setdefault((float(case["J"]), case["kind"]), []).append(record)
    for (J, kind), rows in grouped.items():
        windows = _contiguous_candidate_windows(rows)
        ranked = sorted(
            windows,
            key=lambda bounds: max(
                abs(row["result"]["spin_hall_angle"]) for row in rows
                if bounds[0] <= row["result"]["energy"] <= bounds[1]
            ),
            reverse=True,
        )[:3]
        for index, (low, high) in enumerate(ranked):
            energies = tuple(np.arange(low - 0.01, high + 0.0101, 0.002))
            for eta in (1e-8, 1e-9, 5e-10):
                for label, width, start, py in (
                    ("local", 4, 7, 0),
                    ("full", 16, 1, 0),
                    ("padded", 4, 7, 12),
                ):
                    cases.append(Case(
                        f"mixed_refine_{index}_{label}", kind, J=J,
                        probe_width=width, probe_start=start, padding_y=py,
                        eta=eta, energies=energies,
                    ))
    return cases


def numerical_validation_cases(records: list[dict]) -> list[Case]:
    """Re-evaluate only threshold offenders at smaller numerical broadening."""
    cases = []
    seen = set()
    for record in records:
        case, result = record["case"], record["result"]
        if case["scan"].startswith("numerical_validation__"):
            continue
        failed = (
            result["gauge_invariance_error"] >= 1e-9
            or result["scattering_unitarity_error"] >= 5e-6
            or result["scattering_charge_mismatch"] >= 5e-6
        )
        if not failed:
            continue
        for eta in (1e-9, 5e-10):
            key = (case["scan"], case["kind"], case["A"], case["R"],
                   case["Nx"], case["Ny"], case["J"], case["probe_width"],
                   case["probe_start"], case["padding_x"], case["padding_y"],
                   eta, result["energy"])
            if key in seen:
                continue
            seen.add(key)
            cases.append(Case(
                f"numerical_validation__{case['scan']}", case["kind"],
                A=case["A"], R=case["R"], Nx=case["Nx"], Ny=case["Ny"],
                J=case["J"], probe_width=case["probe_width"],
                probe_start=case["probe_start"], padding_x=case["padding_x"],
                padding_y=case["padding_y"], eta=eta,
                energies=(float(result["energy"]),),
            ))
    return cases


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _texture_and_lead(case: Case):
    padding = (case.padding_x, case.padding_y)
    if case.kind in {"skyrmionium_inner_wall", "skyrmionium_outer_wall"}:
        component = "inner" if "inner" in case.kind else "outer"
        texture = make_skyrmionium_wall_array(
            component, case.A, case.Nx, case.Ny, case.R, padding=padding
        )
        lead_J = -abs(case.J) if component == "inner" else abs(case.J)
    else:
        texture = make_array_texture(
            case.kind, case.A, case.Nx, case.Ny, case.R, padding=padding
        )
        lead_J = case.J
    return texture, lead_J


def run_cases(cases: list[Case], out: Path, records: list[dict]) -> list[dict]:
    raw_path = out / "raw_cases.jsonl"
    completed = {record["id"] for record in records}
    total = sum(len(case.energies) for case in cases)
    done = 0
    for case in cases:
        texture, lead_J = _texture_and_lead(case)
        contacts = standard_four_contacts(
            texture.shape[0], texture.shape[1], case.probe_width,
            probe_J=0.0, probe_start=case.probe_start,
            longitudinal_J=lead_J,
        )
        for energy in case.energies:
            identifier = case.identifier(energy)
            done += 1
            if identifier in completed:
                continue
            result = evaluate_hall_point(
                texture, energy, case.J, 1.0, contacts, eta=case.eta
            )
            record = {"id": identifier, "case": asdict(case), "result": result}
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            records.append(record)
            completed.add(identifier)
            status = {
                "completed_unique_points": len(completed),
                "current_batch_position": done,
                "current_batch_total": total,
                "last_id": identifier,
            }
            (out / "status.json").write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            print(json.dumps(status), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=(
        "strong", "mixed", "refine", "validate", "all"
    ),
                        default="all")
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--output-label", default="hall_mechanism_v1")
    args = parser.parse_args()
    out = ROOT / "results" / args.output_label
    out.mkdir(parents=True, exist_ok=True)
    records = _read_records(out / "raw_cases.jsonl")
    if args.suite in {"strong", "all"}:
        records = run_cases(strong_cases(args.eta), out, records)
    if args.suite in {"mixed", "all"}:
        records = run_cases(mixed_cases(args.eta), out, records)
    if args.suite in {"refine", "all"}:
        refine = refinement_cases(records)
        records = run_cases(refine, out, records)
        (out / "refinement_case_count.json").write_text(
            json.dumps({"case_count": len(refine)}, indent=2), encoding="utf-8"
        )
    if args.suite in {"validate", "all"}:
        validation = numerical_validation_cases(records)
        records = run_cases(validation, out, records)
        (out / "numerical_validation_case_count.json").write_text(
            json.dumps({"case_count": len(validation)}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
