"""Audit FHS Berry-flux cancellation and branch-cut admissibility."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import RESULTS
from paper_plot_style import ANNOTATION_SIZE, configure_paper_style


def main() -> None:
    source = (RESULTS / "chern" /
              "skyrmionium_q_zero_A18_R8_J5_n325" / "nk31.npz")
    output = RESULTS / "peer_review_chern_validation"
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as archive:
        flux = np.asarray(archive["berry_flux"], dtype=float)
        ks = np.asarray(archive["k_values"], dtype=float)
        chern = float(np.asarray(archive["chern"]))

    positive = float(flux[flux > 0].sum())
    negative = float(flux[flux < 0].sum())
    max_abs = float(np.max(np.abs(flux)))
    report = {
        "source": str(source.relative_to(RESULTS)),
        "nk": int(flux.shape[0]),
        "chern": chern,
        "total_flux": float(flux.sum()),
        "positive_flux": positive,
        "negative_flux": negative,
        "positive_negative_mismatch": abs(positive + negative),
        "max_absolute_plaquette_flux": max_abs,
        "max_absolute_flux_over_pi": max_abs / np.pi,
        "distance_to_principal_branch_cut": float(np.pi - max_abs),
        "fhs_branch_admissibility_passed": bool(max_abs < np.pi),
        "interpretation": (
            "The total occupied-subspace Chern number vanishes by numerical "
            "cancellation of nonzero positive and negative plaquette flux. The "
            "largest flux is far below the FHS principal-branch boundary. This is "
            "a numerical admissibility check, not a proof of symmetry-enforced zero Chern number."
        ),
    }
    (output / "assessment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    q = ks * 18.0
    extent = [q[0], q[-1], q[0], q[-1]]
    limit = np.max(np.abs(flux))
    configure_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), constrained_layout=True)
    im = axes[0].imshow(
        flux.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_xlabel(r"$k_xA$")
    axes[0].set_ylabel(r"$k_yA$")
    axes[0].set_title("(a) occupied-subspace FHS flux")
    fig.colorbar(im, ax=axes[0], label=r"plaquette flux $F_{xy}$")

    axes[1].bar(
        ["positive", "negative", "total"],
        [positive, negative, float(flux.sum())],
        color=["tab:red", "tab:blue", "0.35"],
    )
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("summed FHS flux")
    axes[1].set_title("(b) positive–negative flux cancellation")
    axes[1].text(
        0.98, 0.98,
        rf"$C={chern:.1e}$" + "\n" +
        rf"$\max|F|/\pi={max_abs / np.pi:.1e}$",
        transform=axes[1].transAxes, ha="right", va="top",
        fontsize=ANNOTATION_SIZE,
    )

    for suffix in ("png", "pdf"):
        fig.savefig(output / f"chern_flux_cancellation.{suffix}", dpi=300)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
