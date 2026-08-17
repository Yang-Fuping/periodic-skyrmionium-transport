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
| Final manuscript figures | `scripts/generate_paper_figures.py` | `generated_figures/` |

## Frozen numerical checks

- Single skyrmionium: $T(0)=33.3240877449$, $N(0)=34$.
- Baseline parameters: $A=18a$, $R=8a$, $J/t=5$.
- Full-zone gap: $E/t\in[1.0771431126,1.1223998757]$.
- Occupied subspace: $n_{\rm occ}=325$, Chern number numerically zero.
- Gap-centre decay length: $\xi=6.32a$ with $R^2=0.99997$.

The manuscript's production arrays and their checksums belong in the separate
Zenodo Dataset, not in Git history.
