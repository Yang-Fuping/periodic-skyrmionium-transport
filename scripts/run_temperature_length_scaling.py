"""High-resolution spectra for finite-temperature length scaling."""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import ROOT
from skyrmion_transport.textures import make_array_texture
from skyrmion_transport.transport import clean_lead_modes, two_terminal_transmission


A = 18
R = 8.0
J = 5.0
NY = 2
NX_VALUES = (1, 2, 4)
KINDS = ("uniform", "skyrmionium_q_zero", "skyrmion_q_plus", "skyrmion_q_minus")
ENERGY = np.arange(1.02, 1.1800001, 0.00025)
REFINEMENT_ENERGY = np.arange(1.020125, 1.1800001, 0.00025)


def main():
    out = ROOT / "results" / "temperature_length_scaling_v1"
    out.mkdir(parents=True, exist_ok=True)
    for nx in NX_VALUES:
        for kind in KINDS:
            target = out / f"spectrum_Nx{nx}_{kind}.npz"
            if target.exists():
                with np.load(target) as saved:
                    if np.array_equal(saved["energy"], ENERGY):
                        print(json.dumps({"Nx": nx, "kind": kind, "status": "reused"}), flush=True)
                        continue
            texture = make_array_texture(kind, A, nx, NY, R)
            transmission = two_terminal_transmission(
                texture, ENERGY, J, 1.0, eta=5e-10,
            )
            channels = clean_lead_modes(ENERGY, texture.shape[1], J, 1.0)
            np.savez_compressed(
                target, energy=ENERGY, transmission=transmission,
                lead_channels=channels, eta=5e-10, Nx=nx, Ny=NY,
            )
            print(json.dumps({
                "Nx": nx, "kind": kind, "status": "stored", "points": len(ENERGY),
                "min_T": float(transmission.min()), "max_T": float(transmission.max()),
            }), flush=True)

    # The finite-temperature Q=0 curve is the main observable and develops
    # narrow band-edge resonances for longer arrays.  Fill only its remaining
    # midpoint grid for Nx=4 and 8, yielding dE=0.000125t without duplicating
    # the already converged comparison textures.
    kind = "skyrmionium_q_zero"
    for nx in (4, 8):
        target = out / f"spectrum_refinement_Nx{nx}_{kind}.npz"
        if target.exists():
            with np.load(target) as saved:
                if np.array_equal(saved["energy"], REFINEMENT_ENERGY):
                    print(json.dumps({"Nx": nx, "kind": kind, "status": "refinement-reused"}), flush=True)
                    continue
        texture = make_array_texture(kind, A, nx, NY, R)
        transmission = two_terminal_transmission(
            texture, REFINEMENT_ENERGY, J, 1.0, eta=5e-10,
        )
        channels = clean_lead_modes(REFINEMENT_ENERGY, texture.shape[1], J, 1.0)
        np.savez_compressed(
            target, energy=REFINEMENT_ENERGY, transmission=transmission,
            lead_channels=channels, eta=5e-10, Nx=nx, Ny=NY,
        )
        print(json.dumps({
            "Nx": nx, "kind": kind, "status": "refinement-stored",
            "points": len(REFINEMENT_ENERGY), "min_T": float(transmission.min()),
            "max_T": float(transmission.max()),
        }), flush=True)


if __name__ == "__main__":
    main()
