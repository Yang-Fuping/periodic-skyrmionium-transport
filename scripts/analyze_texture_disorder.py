"""Analyze cellwise texture-disorder length scaling."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from paper_plot_style import LEGEND_SIZE, configure_paper_style

from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import fit_exponential_length, paired_prefix_transmission


def main() -> None:
    configure_paper_style()
    folder = ROOT / "results" / "texture_disorder"
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    parameters = metadata["parameters"]
    nx = np.asarray(parameters["Nx"], dtype=int)
    lengths = parameters["A"] * nx
    clean_texture = make_array_texture(
        "skyrmionium_q_zero",
        parameters["A"],
        int(nx[-1]),
        parameters["Ny"],
        parameters["R"],
    )
    clean_result = paired_prefix_transmission(
        clean_texture,
        tuple(lengths.tolist()),
        np.asarray([parameters["energy"]]),
        parameters["J"],
        1.0,
        eta=parameters["eta"],
    )
    clean = np.asarray([clean_result[int(length)][0] for length in lengths])

    summaries = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    colors = {"radius": "tab:purple", "position": "tab:orange"}
    labels = {"radius": r"radius disorder, $\sigma_R=0.25a$",
              "position": r"position disorder, $\sigma_{r_0}=0.25a$"}
    axes[0].semilogy(
        nx, clean, color="black", marker="D", ms=3.0, lw=1.0,
        linestyle="-", label="perfect array", zorder=6,
    )
    line_styles = {
        "radius": {
            "marker": "o", "markersize": 7.0, "linestyle": "--",
            "markerfacecolor": "none", "markeredgewidth": 1.15,
            "linewidth": 1.0, "zorder": 4,
        },
        "position": {
            "marker": "s", "markersize": 5.2, "linestyle": "-.",
            "markerfacecolor": "none", "markeredgewidth": 1.05,
            "linewidth": 1.0, "zorder": 5,
        },
    }
    for disorder_kind in ("radius", "position"):
        values = np.asarray([
            row["transmission"] for row in rows
            if row["disorder_kind"] == disorder_kind
        ], dtype=float)
        log_values = np.log(np.maximum(values, np.finfo(float).tiny))
        # exp[median(ln T)] equals median(T) for positive T.  Use the latter
        # name explicitly; "typical" conventionally denotes exp[mean(ln T)].
        median_transmission = np.median(values, axis=0)
        q25 = np.exp(np.quantile(log_values, 0.25, axis=0))
        q75 = np.exp(np.quantile(log_values, 0.75, axis=0))
        fit = fit_exponential_length(lengths, median_transmission)
        ratios = values[:, -1] / clean[-1]
        summaries[disorder_kind] = {
            "sample_count": int(values.shape[0]),
            "median_transmission": median_transmission.tolist(),
            "q25_transmission": q25.tolist(),
            "q75_transmission": q75.tolist(),
            "median_fit": fit,
            "Nx8_ratio_to_clean": {
                "median": float(np.median(ratios)),
                "q25": float(np.quantile(ratios, 0.25)),
                "q75": float(np.quantile(ratios, 0.75)),
                "maximum": float(np.max(ratios)),
            },
        }
        axes[0].semilogy(
            nx, median_transmission, color=colors[disorder_kind],
            label=labels[disorder_kind], **line_styles[disorder_kind]
        )
        axes[0].fill_between(nx, q25, q75, color=colors[disorder_kind], alpha=0.18)
        axes[1].boxplot(
            np.log10(np.maximum(values[:, -1], np.finfo(float).tiny)),
            positions=[1 if disorder_kind == "radius" else 2],
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": colors[disorder_kind], "alpha": 0.6},
            medianprops={"color": "black"},
        )
    axes[1].axhline(np.log10(clean[-1]), color="black", ls="--", lw=1.1,
                    label="perfect array")
    axes[0].set(xlabel=r"array length $N_x$", ylabel=r"median transmission $T_{\rm med}$",
                title="(a) texture-disorder length scaling")
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(fontsize=LEGEND_SIZE, handlelength=2.1, framealpha=0.92)
    axes[1].set(xticks=[1, 2], xticklabels=["radius", "position"],
                ylabel=r"$\log_{10}T(N_x=8)$",
                title="(b) 100-realization distributions")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(fontsize=LEGEND_SIZE)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"texture_disorder_robustness.{suffix}", dpi=300)
    plt.close(fig)

    assessment = {
        "parameters": parameters,
        "clean_transmission": clean.tolist(),
        "summaries": summaries,
        "all_ensembles_complete": all(
            row["sample_count"] == parameters["samples"]
            for row in summaries.values()
        ),
        "median_scaling_r2_above_0p98": all(
            row["median_fit"]["r2"] > 0.98 for row in summaries.values()
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
