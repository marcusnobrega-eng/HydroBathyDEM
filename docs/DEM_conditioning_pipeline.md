# DEM Conditioning Pipeline Notes

This document summarizes the active pipeline after the code cleanup. The old
root-level `v4`, `v5`, and `blockfix` script names are no longer used; the
toolbox now runs from `src/dem_processing`.

## Active Python Modules

```text
src/dem_processing/condition_dem.py
src/dem_processing/run_conditioning_1000m.py
src/dem_processing/prepare_lin2020_bankfull_geometry.py
src/dem_processing/calibrate_spatial_hydraulic_geometry.py
src/dem_processing/paths.py
```

## Modeling Choice

The final river network is derived from FABDEM-D4. Lin et al. 2020 is used as
calibration evidence for spatial width/depth coefficients, not as a replacement
river network. This avoids burning a MERIT-derived river geometry directly into
a FABDEM-derived D4 drainage network.

The final bathymetry step computes:

```text
River_Width = beta_1(x,y) * A_D4 ^ beta_2(x,y)
River_Depth = alfa_1(x,y) * A_D4 ^ alfa_2(x,y)
H_abg = ((River_Width / Resolution) * River_Depth^(5/3))^(3/5)
DEM_reduced = DEM - H_abg
```

where `A_D4` is the FABDEM-D4 upstream area in km2.

## Current Parameters

```text
DEM/model resolution = 1000 m
river carving threshold = 100 km2
coefficient calibration threshold = 5000 km2
river geometry mode = spatial_coefficients_or_power_law
slope artifact percentile = 95
carve mode = wide
river depth cap = 30 m
H_abg safety cap = 50 m
```

## Fallback Hierarchy

The intended width/depth hierarchy is:

```text
local Lin-calibrated subcatchment fit
global Lin-calibrated fit for sparse/noisy calibration zones
base HydroPol2D power law only for true coefficient nodata holes
```

The latest completed run used spatial coefficients for almost all D4 river
cells, with only a small number of coefficient nodata cells falling back to the
base power law.

## Output Organization

Generated model outputs are grouped by theme:

```text
Outputs/dem/          DEM stages and final conditioned DEM
Outputs/d4/           D4 routing, accumulation, river mask, width/depth/H_abg
Outputs/diagnostics/  plots and diagnostic rasters
Outputs/reports/      CSV/JSON summaries and generated run README
```

The main final DEM is:

```text
Outputs/dem/DEM_hydraulic_conditioned.tif
```

The main audit layers are:

```text
Outputs/dem/DEM_modification_final_minus_cleaned.tif
Outputs/d4/D4_H_abg_m.tif
Outputs/d4/D4_idx_facc.tif
Outputs/diagnostics/diagnostic_d4_river_extraction.png
Outputs/diagnostics/diagnostic_final_modifications.png
Outputs/reports/D4_HydroPol2D_creek_reduction_summary.csv
```

## Clean Rebuild

From the repository root:

```bash
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

PYTHONPATH=src python3 -m dem_processing.prepare_lin2020_bankfull_geometry --download
PYTHONPATH=src python3 -m dem_processing.calibrate_spatial_hydraulic_geometry --selected-threshold-km2 5000
PYTHONPATH=src python3 -m dem_processing.run_conditioning_1000m
```
