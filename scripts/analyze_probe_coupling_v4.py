"""Analyze the V4 side-probe coupling sensitivity calculation."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT


KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
COLORS = {0.5: "#3182bd", 0.75: "#756bb1", 1.0: "#e6550d"}


def compressed_signs(values: np.ndarray, tolerance: float = 1e-12) -> str:
    signs = np.sign(values[np.abs(values) > tolerance]).astype(int)
    if not len(signs):
        return "0"
    reduced = [int(signs[0])]
    for value in signs[1:]:
        if value != reduced[-1]:
            reduced.append(int(value))
    return ",".join("+" if value > 0 else "-" for value in reduced)


def main() -> None:
    folder = ROOT / "results" / "peer_review_v4_probe_coupling"
    rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    couplings = (0.5, 0.75, 1.0)
    widths = (2, 8, 16)
    assessment = {
        "observable": (
            "C_rel=|theta_TH(Q=0)|/"
            "{[|theta_TH(Q=+1)|+|theta_TH(Q=-1)|]/2}; this is a comparative "
            "Hall-suppression metric, not an intrinsic ring decomposition."
        ),
        "couplings": {},
    }
    coupling_ratios = {}
    for coupling in couplings:
        width_results = {}
        ratios = []
        for width in widths:
            group = {
                row["case"]["kind"]: row["result"] for row in rows
                if row["case"]["scan"] == "width_coupling"
                and row["case"]["probe_coupling"] == coupling
                and row["case"]["probe_width"] == width
            }
            if set(group) != set(KINDS):
                raise RuntimeError(f"Incomplete width group: t_c/t={coupling}, w={width}")
            angle = {kind: float(result["charge_hall_angle"])
                     for kind, result in group.items()}
            denominator = 0.5 * (abs(angle["skyrmion_q_plus"])
                                 + abs(angle["skyrmion_q_minus"]))
            ratio = abs(angle["skyrmionium_q_zero"]) / denominator
            ratios.append(ratio)
            width_results[str(width)] = {
                "charge_hall_angles": angle,
                "relative_hall_suppression_ratio": ratio,
                "q_reversal_absolute_error": abs(
                    angle["skyrmion_q_plus"] + angle["skyrmion_q_minus"]
                ),
                "uniform_absolute_hall_angle": abs(angle["uniform"]),
                "all_valid": all(result["valid_hall_point"] for result in group.values()),
                "max_current_conservation_error": max(
                    result["current_conservation_error"] for result in group.values()
                ),
                "max_probe_current_error": max(
                    result["probe_current_error"] for result in group.values()
                ),
                "max_scattering_unitarity_error": max(
                    result["scattering_unitarity_error"] for result in group.values()
                ),
            }

        position = sorted([
            row for row in rows
            if row["case"]["scan"] == "position_coupling"
            and row["case"]["probe_coupling"] == coupling
        ], key=lambda row: row["case"]["probe_start"])
        if len(position) != 15:
            raise RuntimeError(f"Incomplete position scan for t_c/t={coupling}")
        hall = np.asarray([row["result"]["Rxy_h_over_e2"] for row in position])
        q_window = np.asarray([
            row["result"]["windowed_topological_charge"] for row in position
        ])
        scale = float(np.dot(q_window, hall) / np.dot(q_window, q_window))
        fitted = scale * q_window
        nrmse = float(np.sqrt(np.mean((hall - fitted) ** 2))
                      / (np.max(hall) - np.min(hall)))
        assessment["couplings"][str(coupling)] = {
            "width_scan": width_results,
            "all_three_widths_below_0.1": bool(max(ratios) < 0.1),
            "position_scan": {
                "Rxy_compressed_sign_sequence": compressed_signs(hall),
                "Q_window_compressed_sign_sequence": compressed_signs(q_window),
                "single_scale_factor": scale,
                "NRMSE": nrmse,
                "all_valid": all(row["result"]["valid_hall_point"] for row in position),
            },
        }
        coupling_ratios[coupling] = np.asarray(ratios)

    assessment["acceptance"] = {
        "strong_relative_suppression_at_all_tested_couplings_and_widths": bool(
            all(item["all_three_widths_below_0.1"]
                for item in assessment["couplings"].values())
        ),
        "outer_inner_outer_sequence_at_all_tested_couplings": bool(
            all(item["position_scan"]["Rxy_compressed_sign_sequence"] == "+,-,+"
                for item in assessment["couplings"].values())
        ),
        "interpretation": (
            "Changing t_c changes the probe self-energy and hence the coherent contact "
            "boundary condition, not merely a passive sampling weight."
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 8.0})
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.65), constrained_layout=True)
    for coupling in couplings:
        axes[0].semilogy(
            widths, coupling_ratios[coupling], "o-", ms=3.4,
            color=COLORS[coupling], label=rf"$t_c/t={coupling:g}$"
        )
        position = sorted([
            row for row in rows
            if row["case"]["scan"] == "position_coupling"
            and row["case"]["probe_coupling"] == coupling
        ], key=lambda row: row["case"]["probe_start"])
        centre = np.asarray([row["case"]["probe_start"] + 0.5 for row in position])
        hall = np.asarray([row["result"]["Rxy_h_over_e2"] for row in position])
        axes[1].plot(centre, hall, "o-", ms=2.8, color=COLORS[coupling],
                     label=rf"$t_c/t={coupling:g}$")
    axes[0].axhline(0.1, color="0.35", ls="--", lw=0.8)
    axes[0].set(xlabel=r"probe width $w_p/a$",
                ylabel=r"relative suppression $C_{\rm rel}$",
                title="(a) contact-coupling sensitivity", xticks=widths)
    axes[1].axhline(0.0, color="0.35", lw=0.7)
    axes[1].set(xlabel=r"probe-window centre $x/a$",
                ylabel=r"$R_{xy}$ ($h/e^2$)",
                title=r"(b) local scan, $w_p=2a$")
    for ax in axes:
        ax.grid(alpha=0.22, which="both")
        ax.legend(fontsize=6.5)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"probe_coupling_sensitivity_v4.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
