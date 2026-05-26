# DEM Conditioning Pipeline Notes

This document summarizes the active pipeline after the code cleanup. The old
root-level `v4`, `v5`, and `blockfix` script names are no longer used. The
toolbox now installs as a Python package with console commands.

## Active Python Modules

```text
src/dem_processing/condition_dem.py
src/dem_processing/run_conditioning_1000m.py
src/dem_processing/prepare_lin2020_bankfull_geometry.py
src/dem_processing/calibrate_spatial_hydraulic_geometry.py
src/dem_processing/preflight.py
src/dem_processing/config.py
src/dem_processing/qa.py
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
calibration fit area source = matched FABDEM-D4 area
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
Outputs/d4/D4_power_law_fallback_used.tif
Outputs/diagnostics/diagnostic_d4_river_extraction.png
Outputs/diagnostics/diagnostic_d4_geometry_source_map.png
Outputs/diagnostics/diagnostic_final_modifications.png
Outputs/reports/D4_HydroPol2D_creek_reduction_summary.csv
Outputs/reports/qa_scorecard.csv
Outputs/reports/run_manifest.json
```

## Clean Rebuild

From the repository root:

```bash
python3 -m pip install -e .

riverdem-preflight \
  --config configs/india_1000m_spatial.json \
  --require-lin \
  --require-spatial-coefficients

riverdem-condition --config configs/india_1000m_powerlaw_first_pass.json
riverdem-prepare-lin2020 --download
riverdem-calibrate-hydraulics --selected-threshold-km2 5000 --fit-area-source d4
riverdem-condition-1000m --config configs/india_1000m_spatial.json
```
