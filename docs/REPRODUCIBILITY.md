# Reproducibility map

## Core claims and entry points

| Evidence | Main entry point | Principal output |
|---|---|---|
| Texture normalization and $Q=0,\pm1$ | `scripts/run_stage0.py` | `results/stage0/` |
| Full-zone gap scan | `scripts/run_gap_scan.py` | `results/gap_scan/` |
| Occupied-subspace Chern number | `scripts/run_chern_convergence.py` | `results/chern/` |
| Full-zone band-edge surfaces | `scripts/analyze_full_bz_gap.py` | `results/full_bz_gap/` |
| Finite-array length scaling | `scripts/run_length_scaling.py` | `results/length_scaling/` |
| Thermal length crossover | `scripts/run_temperature_length_scaling.py` | `results/temperature_length_scaling_v1/` |
| Local Hall compensation | `scripts/run_hall_mechanism.py` | `results/hall_mechanism_v1/` |
| Boundary dependence | `scripts/run_boundary_coherence.py` | `results/boundary_coherence_v1/` |
| Paired $Q=0$ versus $Q=\pm1$ disorder | `scripts/run_qpm_disorder_temperature_final.py` | `results/disorder_topology_comparison_v1/` |
| Independent Kwant cross-validation | `scripts/run_kwant_validation.py --include-array-hall` | `results/kwant_validation/` |
| Final manuscript figures 1--4 and S1--S10 | `scripts/generate_paper_figures.py` | `generated_figures/` |
| Final figure/data assertions | `scripts/verify_paper_artifacts.py` | terminal JSON report |

## Frozen numerical checks

- Single skyrmionium: $T(0)=33.3240877449$, $N(0)=34$.
- Baseline parameters: $A=18a$, $R=8a$, $J/t=5$.
- Full-zone gap: $E/t\in[1.0771431126,1.1223998757]$.
- Occupied subspace: $n_{\rm occ}=325$, Chern number numerically zero.
- Gap-centre decay length: $\xi=6.32a$ with $R^2=0.99997$.
- Largest selected-case Kwant/NEGF relative difference: $7.61\times10^{-6}$.
- Largest Kwant scattering-matrix unitarity error: $1.15\times10^{-12}$.

The manuscript's production arrays and their checksums are archived separately
as Zenodo version 1.1.0 at
[doi:10.5281/zenodo.22092300](https://doi.org/10.5281/zenodo.22092300), not in
Git history.

## Frozen environments

- Final figure/core audit: CPython 3.12.13, NumPy 2.5.2, SciPy 1.18.0,
  Matplotlib 3.11.1, SciPy-OpenBLAS 0.3.34 ILP64 on Windows x86-64. See
  `requirements-production-lock.txt`.
- Independent Kwant rerun: CPython 3.11.15, Kwant 1.5.0, NumPy 2.4.6, and
  SciPy 1.17.1. See `environment-kwant-lock.yml`.
- Finite-$\eta$ Fisher--Lee absorption is reported separately from exact-open
  Kwant scattering unitarity; numerical broadening is not interpreted as
  dephasing.

## Disorder provenance

The paired scalar-disorder base seed is `20260814`. All $Q=0$ and $Q=+1$
spectra are independently solved. For $Q=-1$, 17 cases at $W_d/t=0.25$ and 10
at $W_d/t=0.50$ are independent checks; the remainder use the validated exact
two-terminal $Q\rightarrow-Q$ reciprocity. Cellwise texture disorder uses seed
`20260821`. The quantitative contrast is restricted to $k_BT/t=0.01$ because
the strong-disorder $k_BT/t=0.005$ integral is grid sensitive.
