# Kwant independent cross-validation

## Status

The Kwant backend is retained as a supported, independently constructed
validation implementation.  It is not used to generate the production scans:
the primary implementation remains the transparent NumPy/SciPy NEGF and RGF
code.  Kwant rebuilds the finite Hamiltonian and every semi-infinite lead from
the frozen magnetization and contact geometry, then obtains transmissions from
its scattering matrix.

The full comparison was rerun successfully on 24 August 2026 with Python 3.11.15, Kwant 1.5.0,
NumPy 2.4.6 and SciPy 1.17.1.

## Recreate the environment

From `transport_project`:

```powershell
conda env create -f environment-kwant.yml
conda run -n kwant-validate python -m pytest -q -p no:cacheprovider
```

On Windows, use `conda run` or activate the environment before launching
Python.  This ensures that the numerical DLLs in the Conda environment are on
the runtime search path.

## Reproduce the comparison tables

```powershell
conda run -n kwant-validate python scripts/run_kwant_validation.py
conda run -n kwant-validate python scripts/run_kwant_validation.py --include-array-hall
```

The commands write machine-readable records to
`results/kwant_validation/key_cases.json` and
`results/kwant_validation/full_with_array_hall.json`.

## Covered cases

- Analytic uniform-strip channels and two-terminal transmission.
- One $Q=+1$ skyrmion and one $Q=0$ skyrmionium at $E/t=0$.
- A two-terminal $A=18a$, $R=8a$, $J/t=5$, $N_x=1$, $N_y=2$
  skyrmionium array at $E/t=1.065$, $1.099771494$ and $1.15$.
- Four-terminal uniform, $Q=+1$, $Q=-1$ and $Q=0$ devices.
- The representative padded four-terminal skyrmionium array used for the
  minigap-centre Hall analysis.

## Numerical outcome

- All analytic and Kwant propagating-channel counts agree exactly.
- The largest relative Euclidean error between a Kwant result and its
  NumPy/SciPy counterpart is $7.61\times10^{-6}$.
- The largest Kwant scattering-matrix unitarity error is
  $1.15\times10^{-12}$.
- For the representative array, the Hall angles are $0.0092539760$ (Kwant)
  and $0.0092540127$ (NumPy/SciPy), an absolute difference of
  $3.67\times10^{-8}$.
- The four-terminal $Q=+1$ and $Q=-1$ Hall angles are equal and opposite to
  numerical precision, while the uniform Hall angle is zero.
- The complete test suite reports `27 passed` in the Kwant environment: 19
  core numerical tests, three reproduction-entry tests, and five Kwant tests.

The small systematic differences from the production solver are consistent
with comparing the exactly open Kwant scattering problem against NEGF values
evaluated at the declared finite broadening $\eta=10^{-7}t$.
