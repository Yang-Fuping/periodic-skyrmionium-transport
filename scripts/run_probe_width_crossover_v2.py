"""Systematic Hall-probe-width scan at the baseline minigap centre."""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import ROOT
from skyrmion_transport.hall import evaluate_hall_point
from skyrmion_transport.multiterminal import standard_four_contacts
from skyrmion_transport.textures import make_array_texture


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
ENERGY = 1.0997714941836594


def centred_starts(length: int, width: int) -> tuple[int, ...]:
    """Return one exact or two mirror-related central integer windows."""
    left = (length - width) // 2
    right = length - width - left
    return tuple(sorted({left, right}))


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    output = ROOT / "results" / "probe_width_crossover_v2"
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw.jsonl"
    records = read_records(raw)
    completed = {(row["kind"], row["probe_width"], row["probe_start"])
                 for row in records}
    for kind in KINDS:
        texture = make_array_texture(kind, 18, 1, 2, 8.0)
        for width in range(1, 17):
            for start in centred_starts(texture.shape[0], width):
                key = (kind, width, start)
                if key in completed:
                    continue
                contacts = standard_four_contacts(
                    texture.shape[0], texture.shape[1], width,
                    probe_J=0.0, probe_start=start, longitudinal_J=5.0,
                )
                result = evaluate_hall_point(
                    texture, ENERGY, 5.0, 1.0, contacts, eta=1e-8
                )
                row = {
                    "kind": kind,
                    "probe_width": width,
                    "probe_start": start,
                    "energy": ENERGY,
                    "result": result,
                }
                with raw.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                completed.add(key)
                print(f"{kind}: width={width}, start={start}", flush=True)
    metadata = {
        "parameters": {
            "A": 18, "R": 8.0, "J": 5.0, "Nx": 1, "Ny": 2,
            "energy": ENERGY, "probe_J": 0.0, "eta": 1e-8,
            "probe_widths": list(range(1, 17)),
        },
        "centering": (
            "Even widths use the unique half-integer-centred window; odd widths "
            "use both mirror-related central windows."
        ),
        "record_count": len(read_records(raw)),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
