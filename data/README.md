# Paper dataset

The production arrays used in the manuscript are intentionally archived as a
separate Zenodo Dataset. This keeps the code repository small and gives the
frozen numerical evidence an immutable, independently citable identifier.

## Status

Dataset version `1.0.0` is openly available at
[doi:10.5281/zenodo.22082741](https://doi.org/10.5281/zenodo.22082741).

## Expected layout

Download and extract the dataset so that this repository contains:

```text
data/results/stage0/
data/results/paper_main_figure_v1/
data/results/gap_scan/
data/results/chern/
data/results/full_bz_gap/
data/results/length_scaling/
data/results/temperature_length_scaling_v1/
data/results/hall_mechanism_v1/
data/results/disorder_temperature_joint_v1/
data/results/disorder_topology_comparison_v1/
data/results/jpcm_revision_validation/
```

Alternatively, set `SKYRMIONIUM_RESULTS` to the extracted `results` directory.
The directory `data/results/` is ignored by Git.
