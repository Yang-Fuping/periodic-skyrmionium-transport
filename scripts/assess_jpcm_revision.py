"""Assemble machine-readable acceptance checks for the JPCM revision."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.bloch import direct_indirect_gap


def main() -> None:
    chern_path = (ROOT / "results" / "chern" /
                  "skyrmionium_q_zero_A18_R8_J5_n325" / "report.json")
    chern = json.loads(chern_path.read_text(encoding="utf-8"))
    convergence = {entry["nk"]: entry for entry in chern["convergence"]}
    required_meshes = (11, 21, 31)
    chern_pass = all(
        nk in convergence
        and convergence[nk]["direct_gap"] > 0
        and abs(convergence[nk]["chern"]) < 5e-6
        for nk in required_meshes
    )

    gap_root = ROOT / "results" / "gap_scan" / "skyrmionium_q_zero"
    a20 = {}
    for nk in (11, 21):
        with np.load(gap_root / f"R8_J5_nk{nk}" / "A20.npz") as archive:
            a20[nk] = direct_indirect_gap(archive["eigenvalues"], 401)
    gap_difference = abs(a20[11]["indirect_gap"] - a20[21]["indirect_gap"])
    relative_difference = gap_difference / a20[21]["indirect_gap"]
    a20_pass = gap_difference < 1e-4 or relative_difference < 0.01

    full_bz_path = (ROOT / "results" / "full_bz_gap" /
                    "skyrmionium_q_zero_A18_R8_J5_n325" / "report.json")
    full_bz = json.loads(full_bz_path.read_text(encoding="utf-8"))
    finest = next(entry for entry in full_bz["convergence"] if entry["nk"] == 31)
    same_extremum = (
        finest["valence_max_index"] == finest["conduction_min_index"]
        and abs(finest["direct_gap"] - finest["indirect_gap"]) < 1e-12
    )

    report = {
        "occupied_subspace_chern": {
            "status": "pass" if chern_pass else "fail",
            "criterion": "positive direct gap and |C_occ| < 5e-6 at nk=11,21,31",
            "convergence": [convergence[nk] for nk in required_meshes
                            if nk in convergence],
            "source": str(chern_path.relative_to(ROOT.parent)),
        },
        "A20_gap_convergence": {
            "status": "pass" if a20_pass else "fail",
            "gap_nk11": a20[11]["indirect_gap"],
            "gap_nk21": a20[21]["indirect_gap"],
            "absolute_difference": gap_difference,
            "relative_difference": relative_difference,
        },
        "baseline_gap_character": {
            "status": "corrected",
            "classification": "direct full-zone minigap" if same_extremum
                              else "indirect full-zone minigap",
            "valence_max_k": finest["valence_max_k"],
            "conduction_min_k": finest["conduction_min_k"],
            "direct_gap": finest["direct_gap"],
            "full_zone_gap": finest["indirect_gap"],
            "source": str(full_bz_path.relative_to(ROOT.parent)),
        },
        "figure4_decision": "retain_in_main_text" if a20_pass
                            else "move_to_supplement",
        "zero_chern_wording_allowed": chern_pass,
    }
    output = ROOT / "results" / "jpcm_revision_validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not chern_pass:
        raise SystemExit("Occupied-subspace Chern acceptance check failed")


if __name__ == "__main__":
    main()
