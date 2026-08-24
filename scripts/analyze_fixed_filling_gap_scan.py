"""Analyze the fixed-R/A minigap scale separation."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ROOT
from paper_plot_style import ANNOTATION_SIZE, LEGEND_SIZE, configure_paper_style


def main() -> None:
    configure_paper_style()
    folder = ROOT / "results" / "fixed_filling_gap_scan" / "ratio_0.44444444_J5_nk11"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    A = np.asarray([row["A"] for row in report["cells"]], dtype=float)
    gap = np.asarray([row["refined"]["indirect_gap"] for row in report["cells"]])
    inverse_square = 1.0 / A**2
    coefficient = float(np.dot(inverse_square, gap) / np.dot(inverse_square, inverse_square))
    prediction = coefficient * inverse_square
    ss_res = float(np.sum((gap - prediction) ** 2))
    ss_tot = float(np.sum((gap - gap.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    scaled = A**2 * gap

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    dense_A = np.linspace(A.min(), A.max(), 300)
    plot_inverse_square = 1000.0 * inverse_square
    axes[0].plot(1000.0 / dense_A**2, coefficient / dense_A**2,
                 "k--", label=rf"fit: $\Delta/t={coefficient:.3f}/A^2$")
    axes[0].plot(plot_inverse_square, gap, "o", color="tab:blue", ms=7)
    for a, x, y in zip(A, plot_inverse_square, gap):
        if int(a) == int(A.min()):
            offset, ha, va = (-7, -14), "right", "top"
        else:
            offset, ha, va = (6, 9), "left", "bottom"
        axes[0].annotate(
            rf"$A={int(a)}a$", (x, y), xytext=offset,
            textcoords="offset points", fontsize=ANNOTATION_SIZE, ha=ha, va=va,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.76, "pad": 0.35},
        )
    axes[0].margins(x=0.06, y=0.08)
    axes[0].set(xlabel=r"inverse period $10^3(a/A)^2$",
                ylabel=r"full-zone gap $\Delta/t$",
                title="(a) fixed $R/A=4/9$")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[1].plot(A, scaled, "s-", color="tab:purple")
    axes[1].axhline(scaled.mean(), color="black", ls="--", lw=1.0,
                    label=rf"mean $={scaled.mean():.3f}$")
    axes[1].set(xlabel=r"period $A/a$", ylabel=r"scaled gap $A^2\Delta/t$",
                title="(b) kinetic-scale collapse", xticks=A.astype(int))
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=LEGEND_SIZE)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"fixed_filling_gap_scaling.{suffix}", dpi=300)
    plt.close(fig)

    assessment = {
        "A": A.astype(int).tolist(),
        "R": [row["R"] for row in report["cells"]],
        "gap_t": gap.tolist(),
        "A2_gap_t": scaled.tolist(),
        "through_origin_coefficient": coefficient,
        "through_origin_r2": r2,
        "scaled_relative_range": float((scaled.max() - scaled.min()) / scaled.mean()),
        "all_refined_gaps_positive": bool(np.all(gap > 0)),
        "all_optimizers_succeeded": all(
            all(item["success"] for key, item in row["refined"].items()
                if isinstance(item, dict))
            for row in report["cells"]
        ),
    }
    (folder / "assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
