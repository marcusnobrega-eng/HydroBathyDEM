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
  paths.py
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
  --dem /Users/mngomes/Documents/GitHub/DEM_Processing/DEM_fabdem.tif \
  --out-dir /Users/mngomes/Documents/GitHub/DEM_Processing/Outputs \
  --resample-dem \
  --target-resolution-m 1000 \
  --resampling-method bilinear \
  --auto-rivers-d4 \
  --min-area 100 \
  --river-geometry-source power_law \
  --carve-mode wide \
  --channel-cell-width-m 1000 \
  --river-width-cap-m 10000 \
  --river-depth-cap-m 30 \
  --max-H-abg-m 50 \
  --max-nodata-fill-pixels 10 \
  --slope-percentile 95 \
  --smooth-filter-cells 5 \
  --protect-stream-buffer-m 2000 \
  --breach-dist-cells 10000 \
  --breach-flat-increment 0.01 \
  --fill-max-depth-m 0.25
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
PYTHONPATH=src python3 -m dem_processing.calibrate_spatial_hydraulic_geometry --selected-threshold-km2 5000
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
validation performance among `500`, `1000`, `2500`, and `5000 km2`.

### 4. Final DEM conditioning with spatial coefficients

```bash
PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m --dry-run
PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m
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

Outputs/diagnostics/
  quicklook_dem_conditioning.png
  diagnostic_pipeline_stages.png
  diagnostic_smoothing_artifacts.png
  diagnostic_d4_river_extraction.png
  diagnostic_d4_threshold_sweep.png
  diagnostic_d4_mask_over_accumulation.png
  diagnostic_hydraulic_geometry_histograms.png
  diagnostic_final_modifications.png

Outputs/reports/
  conditioning_config.json
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
Outputs/diagnostics/diagnostic_final_modifications.png
Outputs/reports/D4_HydroPol2D_creek_reduction_summary.csv
Data/Lin2020_bankfull_width/calibration/diagnostic_spatial_calibration_d4_area_5000km2.png
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
dem-prepare-lin2020 --download
dem-calibrate-hydraulics --selected-threshold-km2 5000
```
