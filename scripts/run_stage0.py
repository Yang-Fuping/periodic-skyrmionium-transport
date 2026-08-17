from pathlib import Path
import json

import numpy as np

from _bootstrap import ROOT

from skyrmion_transport.textures import (
    TextureKind, lattice_topological_charge, make_texture, max_norm_error,
)
from skyrmion_transport.transport import clean_lead_modes, two_terminal_transmission


def main():
    out = ROOT / "results" / "stage0"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for kind in TextureKind:
        m = make_texture(kind, 60, 30, 8)
        q, density = lattice_topological_charge(m)
        np.savez_compressed(out / f"{kind.value}.npz", m=m, q_density=density, Q=q)
        report[kind.value] = {"Q": q, "max_norm_error": max_norm_error(m)}
    energies = np.linspace(-3, 3, 121)
    m0 = make_texture("skyrmionium_q_zero", 60, 30, 8)
    T = two_terminal_transmission(m0, energies, 1.5, 1.0)
    N = clean_lead_modes(energies, 30, 1.5, 1.0)
    np.savez_compressed(out / "single_skyrmionium_transport.npz", energy=energies, T=T, N=N)
    i0 = int(np.argmin(abs(energies)))
    report["single_baseline"] = {"T0": float(T[i0]), "N0": int(N[i0])}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
