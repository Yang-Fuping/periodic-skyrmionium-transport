# Paper dataset

The production arrays used in the manuscript are intentionally archived as a
separate Zenodo Dataset. This keeps the code repository small and gives the
frozen numerical evidence an immutable, independently citable identifier.

## Status

Zenodo version `1.0.0`
([doi:10.5281/zenodo.22082741](https://doi.org/10.5281/zenodo.22082741))
is a historical archive and does not contain every final peer-review result.
The complete `1.1.0` archive has SHA-256
`F7AA9D88E3B0198D01994DC9DBB505AF997187F8BABA9C6C4999D06412C93666`.
It is published at
[doi:10.5281/zenodo.22092300](https://doi.org/10.5281/zenodo.22092300).
The archived ZIP can be downloaded directly from the
[Zenodo file endpoint](https://zenodo.org/api/records/22092300/files/periodic-skyrmionium-transport-data-v1.1.0.zip/content).
Do not claim full final-figure reproduction from `1.0.0`.

The panel-level mapping from Figures 1--4 and S1--S10 to archived files and
fields is versioned with the immutable source release in
[docs/FIGURE_DATA_INDEX.md](../docs/FIGURE_DATA_INDEX.md), with a
[machine-readable JSON source](../docs/figure_data_index.json).

Download, verify, extract, and reproduce the figures with:

```powershell
python scripts/fetch_zenodo_dataset.py
python scripts/generate_paper_figures.py
python scripts/verify_paper_artifacts.py
```

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
