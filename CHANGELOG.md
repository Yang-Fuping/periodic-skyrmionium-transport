# Changelog

## 0.2.1 - 2026-08-25

- Added machine-readable and generated Markdown figure-to-data indexes covering
  every panel of Figures 1--4 and Supplementary Figures S1--S10.
- Added a regression check for all 43 panel mappings, bringing the complete
  suite to 27 tests.
- Defines the immutable `v0.2.1` source version referenced by the manuscript.

## 0.2.0 - 2026-08-25

- Replaced the provisional plotting entry point with exact regeneration of
  Figures 1--4 and Supplementary Figures S1--S10 from the complete dataset.
- Added portable results/figure-root overrides, artifact verification, and two
  reproduction-entry tests, bringing the full suite to 26 tests.
- Added exact production and Kwant environment locks and a verified Zenodo
  download/extraction helper.
- Updated the physical scope, numerical provenance, and bilingual documentation
  for the final peer-review archive.
- Linked the complete Zenodo dataset version 1.1.0 at
  `doi:10.5281/zenodo.22092300`.

## 0.1.0 - 2026-08-17

- Prepared the first public-repository candidate.
- Added normalized $Q=0,\pm1$ magnetic textures and lattice solid-angle charge.
- Added Bloch-supercell gaps and FHS occupied-subspace Chern calculations.
- Added dense and sparse four-terminal NEGF implementations.
- Added two-terminal recursive Green functions, disorder statistics, and
  finite-temperature convolution.
- Added the original 15-test numerical validation suite and paper-figure generator.
- Added four regression checks for the complex-band solver and revised
  texture/profile controls, bringing the core suite to 19 tests.
- Added an optional Conda/Kwant 1.5 backend, five independent transport
  regression cases, and a retained full validation workflow.
