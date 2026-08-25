"""Regression tests for the portable final-figure entry points."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")


def test_load_joined_uses_explicit_results_root(tmp_path: Path) -> None:
    module = importlib.import_module("analyze_qpm_disorder_temperature")
    write_jsonl(
        tmp_path / "disorder_temperature_joint_v1" / "pilot.jsonl",
        [{"id": "q0", "case": {"Wd": 0.25, "sample": 3},
          "energy": [1.0], "result": {"q0": True}}],
    )
    write_jsonl(
        tmp_path / "disorder_topology_comparison_v1" / "qpm_cases.jsonl",
        [{"id": "qpm", "case": {"Wd": 0.25, "sample": 3},
          "result": {"skyrmion_q_plus": {"qp": True},
                     "skyrmion_q_minus": {"qm": True}},
          "qminus_mode": "independent"}],
    )
    joined, q0_count, qpm_count = module.load_joined(tmp_path)
    assert (q0_count, qpm_count) == (1, 1)
    assert len(joined) == 1
    assert joined[0]["result"]["skyrmionium_q_zero"] == {"q0": True}
    assert joined[0]["qminus_mode"] == "independent"


def test_final_figure_manifest_is_complete() -> None:
    module = importlib.import_module("verify_paper_artifacts")
    assert len(module.STEMS) == 14
    assert module.STEMS[:4] == (
        "figure1_textures_topology",
        "figure2_minigap_transport",
        "figure3_hall_disorder",
        "figure4_tunability_applications",
    )
    assert module.STEMS[-1] == "supplementary_figure_s10_probe_width_crossover"
