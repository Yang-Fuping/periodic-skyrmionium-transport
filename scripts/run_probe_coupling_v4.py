"""V4 sensitivity test for the side-probe/device interface coupling."""

from __future__ import annotations

import json

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture


ENERGY = 1.0997714941836594
COUPLINGS = (0.5, 0.75, 1.0)
WIDTHS = (2, 8, 16)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")


def centred_start(length: int, width: int) -> int:
    return (length - width) // 2


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    out = ROOT / "results" / "peer_review_v4_probe_coupling"
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    rows = read_rows(raw)
    completed = {row["id"] for row in rows}

    cases = []
    for coupling in COUPLINGS:
        for width in WIDTHS:
            for kind in KINDS:
                cases.append({
                    "scan": "width_coupling", "kind": kind,
                    "probe_coupling": coupling, "probe_width": width,
                    "probe_start": centred_start(18, width),
                })
        for start in range(1, 16):
            cases.append({
                "scan": "position_coupling", "kind": "skyrmionium_q_zero",
                "probe_coupling": coupling, "probe_width": 2,
                "probe_start": start,
            })

    textures = {
        kind: make_array_texture(kind, 18, 1, 2, 8.0) for kind in KINDS
    }
    for index, case in enumerate(cases, 1):
        case_id = "|".join(f"{key}={case[key]}" for key in sorted(case))
        if case_id in completed:
            continue
        contacts = standard_four_contacts(
            18, 36, case["probe_width"], probe_J=0.0,
            probe_start=case["probe_start"], longitudinal_J=5.0,
            probe_coupling=case["probe_coupling"],
        )
        result = evaluate_hall_point(
            textures[case["kind"]], ENERGY, 5.0, 1.0, contacts, eta=1e-8
        )
        row = {"id": case_id, "case": case, "energy": ENERGY, "result": result}
        with raw.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        completed.add(case_id)
        print(json.dumps({"completed": len(completed), "total": len(cases),
                          "position": index}), flush=True)

    metadata = {
        "model": {
            "A": 18, "R": 8.0, "J_over_t": 5.0, "Nx": 1, "Ny": 2,
            "energy_over_t": ENERGY, "eta_over_t": 1e-8,
            "longitudinal_lead_J_over_t": 5.0,
            "side_probe_J_over_t": 0.0,
            "longitudinal_interface_hopping_over_t": 1.0,
        },
        "side_probe_interface_hopping_over_t": list(COUPLINGS),
        "probe_widths_over_a": list(WIDTHS),
        "position_scan": {"probe_width_over_a": 2, "starts_over_a": list(range(1, 16))},
        "self_energy_convention": (
            "For a probe-device hopping t_c, Sigma_probe=(t_c/t)^2 Sigma_probe(t)."
        ),
        "record_count": len(read_rows(raw)),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
