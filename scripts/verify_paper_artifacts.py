"""Verify the final paper-figure manifest and its numerical source values."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("SKYRMIONIUM_RESULTS", ROOT / "data" / "results")).resolve()
FIGURES = Path(os.environ.get("SKYRMIONIUM_FIGURES", ROOT / "generated_figures")).resolve()

STEMS = (
    "figure1_textures_topology",
    "figure2_minigap_transport",
    "figure3_hall_disorder",
    "figure4_tunability_applications",
    "supplementary_figure_s1_full_bz_gap",
    "supplementary_figure_s2_texture_gap_controls",
    "supplementary_figure_s3_chern_flux_cancellation",
    "supplementary_figure_s4_length_width_convergence",
    "supplementary_figure_s5_r7_length_scaling",
    "supplementary_figure_s6_complex_band_validation",
    "supplementary_figure_s7_texture_disorder",
    "supplementary_figure_s8_fixed_filling_scaling",
    "supplementary_figure_s9_mz_form_factor",
    "supplementary_figure_s10_probe_width_crossover",
)


def png_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"Not a PNG file: {path}")
        length = struct.unpack(">I", handle.read(4))[0]
        if handle.read(4) != b"IHDR" or length != 13:
            raise AssertionError(f"Missing PNG IHDR: {path}")
        width, height = struct.unpack(">II", handle.read(8))
    return width, height


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"{label}: {actual} != {expected} within {tolerance}")


def main() -> None:
    missing = []
    shapes = {}
    for stem in STEMS:
        for suffix in ("png", "pdf"):
            path = FIGURES / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        png = FIGURES / f"{stem}.png"
        if png.is_file():
            shapes[stem] = png_shape(png)
    if missing:
        raise AssertionError("Missing or empty final figures:\n" + "\n".join(missing))

    gap = json.loads((RESULTS / "full_bz_gap" /
                      "skyrmionium_q_zero_A18_R8_J5_n325" /
                      "report.json").read_text(encoding="utf-8"))
    nk31 = next(row for row in gap["convergence"] if row["nk"] == 31)
    close(nk31["indirect_gap"], 0.045256763096937824, 5e-13,
          "baseline full-zone gap")

    chern = json.loads((RESULTS / "chern" /
                        "skyrmionium_q_zero_A18_R8_J5_n325" /
                        "report.json").read_text(encoding="utf-8"))
    for row in chern["convergence"]:
        if abs(row["chern"]) >= 5e-6:
            raise AssertionError(f"Nonzero occupied-subspace Chern: {row}")

    controls = json.loads((RESULTS / "peer_review_texture_controls" /
                           "assessment.json").read_text(encoding="utf-8"))
    close(controls["same_mz_relative_gap_change"], 0.004024800549746077,
          5e-13, "same-mz relative gap change")

    length = json.loads((RESULTS / "peer_review_convergence" /
                         "assessment.json").read_text(encoding="utf-8"))
    close(length["extended_length_scaling"]["decay_length"],
          6.321486491934364, 5e-12, "six-point decay length")

    disorder = json.loads((RESULTS / "disorder_topology_comparison_v1" /
                           "assessment.json").read_text(encoding="utf-8"))
    if "kBT/t=0.01 is the quantitative" not in disorder["scope"]:
        raise AssertionError("The quantitative temperature scope is missing")
    counts = {float(row["Wd"]): row["temperature"]["0.01"]
              ["paired_topology_comparison"]["independent_Qminus_validation_count"]
              for row in disorder["summary"]}
    if counts != {0.25: 17, 0.5: 10}:
        raise AssertionError(f"Unexpected independent Q- counts: {counts}")

    print(json.dumps({
        "status": "passed",
        "figure_pairs": len(STEMS),
        "png_shapes": shapes,
        "baseline_gap_t": nk31["indirect_gap"],
        "qminus_independent_counts": counts,
        "quantitative_temperature": "kBT/t=0.01",
    }, indent=2))


if __name__ == "__main__":
    main()
