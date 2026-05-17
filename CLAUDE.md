# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`samana` is a Python package for dark matter inference using quadruply-imaged quasars (quad lenses). It stores reduced imaging data and lens models for ~28 quad lens systems, and provides tools to forward-model these systems with dark matter substructure using approximate Bayesian computation (ABC).

Key external dependencies: `lenstronomy` (lens modeling), `pyHalo` (dark matter realization generation), `trikde` (KDE-based inference), `cobyqa` (optimization). All live as sibling directories under `samana_dual_sources/`.

## Installation

```bash
pip install -e .
```

## Running tests

```bash
python setup.py test
# or
pytest tests/
# single test:
pytest tests/test_samana.py::TestSamana::test_000_something
```

## Running a production forward model

Production scripts live in `scripts/`. Each takes a job index as the first argument (for HPC job arrays):

```bash
python scripts/production_script_0405.py 0
```

Output goes to `$SCRATCH/chains/<job_name>/job_<job_index>/`. The simulation can be restarted by re-running the same command — it resumes from existing output files.

## Architecture

### Two parallel module trees

Every lens system has a paired **Data class** (`samana/Data/<sysname>.py`) and **Model class** (`samana/Model/<sysname>_model.py`). All data classes inherit from `ImagingDataBase` (or `QuadNoImageDataBase` for systems without HST/JWST imaging). All model classes inherit from `EPLModelBase` in `samana/Model/model_base.py`.

- **Data class** holds: pixel data, PSF model, coordinate system, image positions, flux ratios, astrometric uncertainties, likelihood masks.
- **Model class** defines: macromodel lens profile (EPL + external shear ± multipoles ± satellites), light models, parameter priors, constraints for `lenstronomy`.

### Forward model loop (`samana/forward_model.py`)

`forward_model()` is the top-level ABC sampler. For each iteration it calls `forward_model_single_iteration()`, which:

1. Samples priors for DM realization parameters, source size, and fixed macromodel params.
2. Generates a pyHalo dark matter realization (CDM, WDM, SIDM, etc. via `preset_model_name`).
3. Aligns halo line-of-sight positions using ray-path interpolation.
4. Sets up the decoupled multiplane approximation (DMPA) via `model_class.setup_kwargs_model()`.
5. Solves the lens equation (using `lenstronomy`'s `Optimizer` or `cobyqa` minimizer).
6. Computes image magnifications via finite-source ray tracing.
7. Evaluates a flux-ratio summary statistic and accepts/rejects via ABC tolerance.
8. Optionally reconstructs imaging data with `lenstronomy`'s `FittingSequence` (PSO).

Accepted samples are written to text files (`parameters.txt`, `mags.txt`, `macromodel_samples.txt`).

There are two variants for specialized physics: `forward_model_sidm.py` (self-interacting DM) and `forward_model_pbh.py` (primordial black holes).

### Prior specification

Priors passed as dicts with syntax:
```python
{'param_name': ['UNIFORM', low, high]}
{'param_name': ['GAUSSIAN', mean, std]}
{'param_name': ['FIXED', value]}
```
Three separate prior dicts are passed to `forward_model`: `kwargs_sample_realization` (DM halo params), `kwargs_sample_source` (source size), `kwargs_sample_fixed_macromodel` (macromodel params like `gamma`, `a3_a`, `a4_a`).

### Output and inference

`samana/output_storage.py`: aggregates per-job text output into HDF5 files.  
`samana/inference_util.py`: post-processing utilities using `trikde` for KDE-based posteriors.  
`samana/analysis_util.py`: higher-level analysis helpers.  
`samana/plotting_util.py`: plotting utilities.

### Run scripts (`/home/anierenberg/repositories/test_dual_sources/`)

Production run scripts live outside this repo. Each per-lens script (e.g. `j0405.py`) imports from three shared config files in that directory:

- **`flux_ratio_measurements.py`** — `measurements` dict keyed by lens ID. Each entry contains `measured_fluxes`, `flux_ratio_covariance_matrix` (N×N for N flux ratios), `keep_flux_ratio_index`, and `uncertainty_on_ratios`. For two-source runs, the covariance matrix should be 6×6 (block diagonal: source 1's 3×3 in the top-left, source 2's 3×3 in the bottom-right).
- **`background_source_priors.py`** — `source_priors` dict keyed by lens ID. Each entry contains `source_size_pc` (prior spec), optionally `source_size_pc_2` for two-source runs, `shapelets_order`, and `mask_images_reconstruction`.
- **`realization_priors_wdm.py`** / **`realisation_priors_sidm_fixed.py`** — DM realization prior dicts.

The run script sets `data_class.flux_ratio_covariance_matrix` and `data_class.magnifications` directly before calling `forward_model`.

### pyHalo (`/home/anierenberg/repositories/samana_dual_sources/pyHalo`)

Entry point: `preset_model_from_name(name)` returns a callable; invoke as `fn(z_lens, z_source, **kwargs)` to get a `Realization`. Preset names: `CDM`, `WDM`, `WDMGeneral`, `SIDM_core_collapse`, `SIDM_parametric`, `ULDM`, `CDM_plus_BH`, etc.

Key classes: `Realization` (halo population container), `LensCosmo` (`Halos/lens_cosmo.py`, lensing cosmology), `Geometry` (`Cosmology/geometry.py`, rendering volume), `HaloPopulation` (`Rendering/halo_population.py`, orchestrates SUBHALOS + LINE_OF_SIGHT + TWO_HALO).

Key parameters: `sigma_sub` (subhalo abundance), `log_mlow`/`log_mhigh`/`shmf_log_slope` (mass function), `log_mc` (WDM half-mode mass), `concentration_model_subhalos/fieldhalos`, `truncation_model_subhalos/fieldhalos`, `cone_opening_angle_arcsec`, `log_m_host`.

Extension points: new mass function → subclass `_PowerLawBase` in `Rendering/MassFunctions/`, register in `mass_function_models.py`; new concentration → subclass in `Halos/concentration.py`, register in `concentration_models.py`; new preset → add to `PresetModels/`, register in `preset_models.py`.

### Adding a new lens system

1. Create `samana/Data/<sysname>.py` inheriting from `ImagingDataBase` (or `QuadNoImageDataBase`). Implement `coordinate_properties`, `kwargs_psf`, `kwargs_numerics`, and `likelihood_masks`.
2. Create `samana/Model/<sysname>_model.py` inheriting from `EPLModelBase`. Implement `setup_lens_model`, `setup_source_light_model`, `setup_lens_light_model`, `kwargs_constraints`, `kwargs_likelihood`.
3. Write a production script in `scripts/` following the existing pattern.
