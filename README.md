# DEM Processing

HydroPol2D-style DEM conditioning toolbox for FABDEM, D4 routing, and
Lin et al. 2020 hydraulic-geometry calibration.

## Current Layout

```text
DEM_fabdem.tif
pyproject.toml
src/dem_processing/
  condition_dem.py
  run_conditioning_1000m.py
  prepare_lin2020_bankfull_geometry.py
  calibrate_spatial_hydraulic_geometry.py
  preflight.py
  config.py
  qa.py
  paths.py
configs/
  india_1000m_spatial.json
  india_1000m_powerlaw_first_pass.json
tests/
Data/Lin2020_bankfull_width/
Outputs/
  dem/
  d4/
  diagnostics/
  reports/
```

Large data files are intentionally not tracked by Git. The source DEM, downloaded
Lin shapefile, generated rasters, plots, and output reports are ignored by
`.gitignore` because GitHub rejects normal Git files larger than 100 MB and
these products can be regenerated from the pipeline.

The old loose root-level Python scripts were removed. Run the toolbox as a
package with `PYTHONPATH=src`, or install it locally with `python3 -m pip
install -e .`.

## Quick Rerun

Use this when the Lin et al. data and spatial coefficient rasters already exist.

```bash
cd /Users/mngomes/Documents/GitHub/DEM_Processing

PYTHONPATH=src python3 -m dem_processing.preflight \
  --config configs/india_1000m_spatial.json \
  --require-lin \
  --require-spatial-coefficients

PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m --dry-run
PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m
```

The current runner uses:

```text
river-geometry-source = spatial_coefficients_or_power_law
river carving threshold = 100 km2
coefficient region threshold = 5000 km2
DEM/model resolution = 1000 m
slope artifact percentile = 95
```

## Complete Clean Pipeline

Use this full sequence when rebuilding from scratch, changing the DEM/grid
resolution, or regenerating the Lin-calibrated hydraulic geometry products.

### 1. Create the FABDEM-D4 drainage world

The Lin calibration needs FABDEM-D4 flow direction and accumulation first. This
first pass uses the base HydroPol2D power law only to create the D4 products
needed for calibration.

```bash
cd /Users/mngomes/Documents/GitHub/DEM_Processing

PYTHONPATH=src python3 -m dem_processing.condition_dem \
  --config configs/india_1000m_powerlaw_first_pass.json
```

This creates the D4 rasters used by the calibration step:

```text
Outputs/d4/D4_flow_direction.tif
Outputs/d4/D4_Wshed_Properties_fac_area_km2.tif
```

### 2. Prepare Lin et al. 2020 width/depth data

```bash
PYTHONPATH=src python3 -m dem_processing.prepare_lin2020_bankfull_geometry --download
```

This downloads the Lin et al. reach data if needed, clips it to the DEM domain,
computes Manning Q2 depth from `width_m`, `Q2`, and `Slp`, and writes the
processed products under:

```text
Data/Lin2020_bankfull_width/processed/
```

### 3. Calibrate spatial hydraulic geometry

```bash
PYTHONPATH=src python3 -m dem_processing.calibrate_spatial_hydraulic_geometry \
  --selected-threshold-km2 5000 \
  --fit-area-source d4
```

This delineates FABDEM-D4 calibration subcatchments, matches valid Lin samples,
fits spatial power-law coefficients, and writes:

```text
Data/Lin2020_bankfull_width/calibration/D4_beta_1_width_5000km2.tif
Data/Lin2020_bankfull_width/calibration/D4_beta_2_width_5000km2.tif
Data/Lin2020_bankfull_width/calibration/D4_alfa_1_depth_5000km2.tif
Data/Lin2020_bankfull_width/calibration/D4_alfa_2_depth_5000km2.tif
```

The selected calibration threshold is `5000 km2` because it had the best
validation performance among `500`, `1000`, `2500`, and `5000 km2`. The current
default fit uses matched FABDEM-D4 drainage area (`--fit-area-source d4`) so the
calibration predictor matches the area used later by HydroPol2D.

### 4. Final DEM conditioning with spatial coefficients

```bash
PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m \
  --config configs/india_1000m_spatial.json \
  --dry-run

PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m \
  --config configs/india_1000m_spatial.json
```

In this mode, the river mask and drainage area come from FABDEM-D4. Lin et al.
2020 is used to calibrate spatial width/depth coefficients, not to replace the
FABDEM-derived D4 river network. Missing coefficient cells fall back to the base
HydroPol2D power law to keep the carved network continuous.

## Output Folders

```text
Outputs/dem/
  DEM_resampled_1000m.tif
  DEM_raw_cleaned.tif
  DEM_artifact_smoothed.tif
  DEM_hydrologically_conditioned_pre_bathymetry.tif
  DEM_reduced_creeks_D4.tif
  DEM_hydraulic_conditioned.tif
  DEM_modification_final_minus_cleaned.tif

Outputs/d4/
  D4_flow_direction.tif
  D4_flow_accum_cells.tif
  D4_Wshed_Properties_fac_area_km2.tif
  D4_idx_facc.tif
  D4_Wshed_Properties_River_Width_m.tif
  D4_Wshed_Properties_River_Depth_m.tif
  D4_H_abg_m.tif
  D4_spatial_coefficients_used.tif
  D4_power_law_fallback_used.tif

Outputs/diagnostics/
  quicklook_dem_conditioning.png
  diagnostic_pipeline_stages.png
  diagnostic_smoothing_artifacts.png
  diagnostic_d4_river_extraction.png
  diagnostic_d4_threshold_sweep.png
  diagnostic_d4_mask_over_accumulation.png
  diagnostic_d4_geometry_source_map.png
  diagnostic_hydraulic_geometry_histograms.png
  diagnostic_final_modifications.png

Outputs/reports/
  conditioning_config.json
  run_manifest.json
  qa_scorecard.csv
  qa_scorecard.json
  qa_scorecard.md
  modification_summary.csv
  D4_HydroPol2D_creek_reduction_summary.csv
  D4_river_connectivity_summary.csv
  DEM_conditioning_README.md
```

Key files to inspect after a final run:

```text
Outputs/dem/DEM_hydraulic_conditioned.tif
Outputs/dem/DEM_modification_final_minus_cleaned.tif
Outputs/d4/D4_idx_facc.tif
Outputs/d4/D4_H_abg_m.tif
Outputs/diagnostics/diagnostic_d4_river_extraction.png
Outputs/diagnostics/diagnostic_d4_geometry_source_map.png
Outputs/diagnostics/diagnostic_final_modifications.png
Outputs/reports/D4_HydroPol2D_creek_reduction_summary.csv
Outputs/reports/qa_scorecard.csv
Outputs/reports/run_manifest.json
Data/Lin2020_bankfull_width/calibration/diagnostic_spatial_calibration_d4_area_5000km2.png
```

## QA, Manifest, and Tests

Every final conditioning run now writes:

```text
Outputs/reports/run_manifest.json
Outputs/reports/qa_scorecard.csv
Outputs/reports/qa_scorecard.json
Outputs/reports/qa_scorecard.md
```

The scorecard currently flags large DEM lowering and river-mask fragmentation
as warnings, while keeping coefficient fallback counts visible.

Run the lightweight tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Lin Calibration Diagnostics

The Lin et al. preparation and calibration stages also create:

```text
Data/Lin2020_bankfull_width/processed/diagnostic_lin2020_width_depth.png
Data/Lin2020_bankfull_width/processed/lin2020_width_depth_summary.csv
Data/Lin2020_bankfull_width/processed/lin2020_overlap_with_current_d4_summary.csv
Data/Lin2020_bankfull_width/calibration/spatial_calibration_threshold_optimization_summary.csv
Data/Lin2020_bankfull_width/calibration/diagnostic_spatial_calibration_thresholds.png
Data/Lin2020_bankfull_width/calibration/diagnostic_spatial_calibration_5000km2.png
Data/Lin2020_bankfull_width/calibration/diagnostic_spatial_calibration_d4_area_5000km2.png
```

## Optional Installed Commands

After:

```bash
python3 -m pip install -e .
```

you can use:

```bash
dem-condition-1000m --dry-run
dem-condition-1000m
dem-preflight --config configs/india_1000m_spatial.json --require-lin --require-spatial-coefficients
dem-prepare-lin2020 --download
dem-calibrate-hydraulics --selected-threshold-km2 5000 --fit-area-source d4
```
