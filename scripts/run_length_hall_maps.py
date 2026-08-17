"""Local spectral maps for the length-driven Hall-mechanism crossover."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.hall import contact_spectral_maps
from skyrmion_transport.multiterminal import (
    four_terminal_observables, standard_four_contacts, transmission_matrix_sparse,
)
from skyrmion_transport.textures import make_array_texture


def main():
    out = ROOT / "results" / "length_hall_main_v1" / "spectral_maps"
    out.mkdir(exist_ok=True)
    for energy, label in ((1.0997714941836594, "center"), (1.1224, "upper")):
        for nx in (1, 2, 4, 8):
            path = out / f"q0_{label}_Nx{nx}.npz"
            if path.exists():
                continue
            texture = make_array_texture("skyrmionium_q_zero", 18, nx, 2, 8.0)
            offset = ((nx - 1) // 2) * 18
            contacts = standard_four_contacts(
                texture.shape[0], texture.shape[1], 2, probe_J=0.0,
                probe_start=offset + 8, longitudinal_J=5.0,
            )
            transmission, diagnostics = transmission_matrix_sparse(
                texture, energy, 5.0, 1.0, contacts, eta=5e-10
            )
            lb = four_terminal_observables(transmission)
            maps = contact_spectral_maps(diagnostics, texture.shape[:2], source_lead=0)
            np.savez_compressed(
                path, texture=texture, transmission=transmission,
                voltages=lb["voltages"], **maps,
            )
            print(json.dumps({"saved": str(path)}), flush=True)


if __name__ == "__main__":
    main()
