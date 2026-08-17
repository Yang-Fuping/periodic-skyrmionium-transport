# Periodic Skyrmionium Transport

**English** | [简体中文](https://github.com/Yang-Fuping/periodic-skyrmionium-transport/blob/main/README.zh-CN.md)

Reproducible NumPy/SciPy calculations for the manuscript
*Zero-Chern Minigap and Locally Compensated Hall Response in Periodic
Skyrmionium Arrays*.

> **Development release.** Version `0.1.0` contains the tested source code and
> reproducibility workflow. The immutable production dataset and its DOI will
> be linked before the archival `v1.0.0` release.

## Scope

The code studies non-interacting spinful electrons on a square lattice with
nearest-neighbour hopping and local exchange coupling to a frozen classical
magnetic texture,

$$
H=-t\sum_{\langle i,j\rangle}c_i^\dagger c_j
-J\sum_i c_i^\dagger(\mathbf m_i\cdot\boldsymbol\sigma)c_i
+\sum_i U_i c_i^\dagger c_i.
$$

It provides:

- normalized uniform, $Q=0$ skyrmionium, and $Q=\pm1$ skyrmion textures;
- lattice solid-angle topological charge and local charge density;
- Bloch-supercell bands, full-zone gaps, DOS, and FHS Chern numbers;
- two-terminal recursive Green-function and full-matrix reference solvers;
- four-terminal NEGF and Landauer--Büttiker voltage probes;
- spin-resolved scattering observables;
- Anderson-disorder statistics and finite-temperature Fermi convolution;
- scripts used for the manuscript's numerical checks and figures.

No Kwant result is claimed. The validation chain uses analytic lead channels,
uniform folded bands, full matrix inversion, recursive Green functions,
sparse/dense NEGF agreement, symmetries, and convergence tests.

## Repository layout

```text
skyrmion_transport/   Core numerical library
scripts/              Production, analysis, and plotting scripts
tests/                Fifteen numerical regression tests
legacy/               Frozen single-skyrmionium baseline and reference image
data/                 Instructions for the separately archived Zenodo dataset
docs/                  Reproducibility map and release notes
```

Generated arrays are written below `results/` and generated paper figures below
`generated_figures/`; both locations are ignored by Git.

## Installation

Python 3.10--3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate`.

## Validation

The test suite uses only small systems and normally finishes within seconds:

```powershell
python -m unittest discover -s tests -v
```

The expected result is `Ran 15 tests ... OK`. The tests cover texture
normalization and charge, analytic folded bands, a zero uniform Chern number,
recursive/full-inverse transmission agreement, lead channel counts,
Landauer--Büttiker gauge invariance and current conservation, $Q\to-Q$ Hall
symmetry, sparse/dense agreement, disorder identity at $W_d=0$, and thermal
convolution.

## Minimal examples

Generate the frozen single-texture benchmark:

```powershell
python scripts/run_stage0.py
```

The expected zero-energy values are approximately
`T(0) = 33.3240877` and `N(0) = 34`.

Compute the baseline occupied-subspace Chern convergence:

```powershell
python scripts/run_chern_convergence.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 `
  --n-occ 325 --nk 11 21 31
```

Compute the baseline length scaling:

```powershell
python scripts/run_length_scaling.py `
  --kind skyrmionium_q_zero --A 18 --R 8 --J 5 --Ny 2 `
  --Nx 1 2 4 8 --energy 1.065 1.0997714941836594 1.15
```

The production calculations can be computationally expensive. In particular,
fine Chern meshes, four-terminal scans, and 100-realization disorder ensembles
should not be started as quick smoke tests.

## Paper dataset and figures

The frozen production arrays are archived separately because research data
should have an immutable dataset DOI rather than being duplicated through Git
history. Follow [data/README.md](data/README.md), then arrange the extracted data
as:

```text
data/results/stage0/
data/results/gap_scan/
data/results/chern/
...
```

The data root and output directory can be overridden:

```powershell
$env:SKYRMIONIUM_RESULTS = "D:\path\to\results"
$env:SKYRMIONIUM_FIGURES = "D:\path\to\figures"
python scripts/generate_paper_figures.py
```

Without those variables, the defaults are `data/results/` and
`generated_figures/`.

## Interpretation limits

- A high-symmetry-path opening is not called a minigap unless the full magnetic
  Brillouin-zone indirect separation remains positive.
- Numerical broadening $\eta$ is not temperature or dephasing.
- The frozen textures do not establish thermodynamic magnetic stability.
- The baseline model excludes spin--orbit coupling, interactions, phonons, and
  self-consistent magnetic dynamics.
- The finite-device Hall result is not extrapolated to a converged two-dimensional
  bulk residual.

## Data, citation, and license

- Dataset DOI: **to be added before `v1.0.0`**.
- Software DOI: generated by the GitHub--Zenodo archive of `v1.0.0`.
- Source repository: [Yang-Fuping/periodic-skyrmionium-transport](https://github.com/Yang-Fuping/periodic-skyrmionium-transport).
- Citation metadata: [CITATION.cff](CITATION.cff).
- License: [MIT](LICENSE).
