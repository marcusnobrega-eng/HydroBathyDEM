#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
condition_dem.py

HydroBathyDEM pipeline for flood/hydrodynamic DEM conditioning.

Default DEM path
----------------
DEM_fabdem.tif in the active project root.

Default output folder
---------------------
If --out-dir is not provided, all outputs are written to:
Outputs next to the input DEM.

Purpose
-------
This script prepares a DEM for hydrologic and hydraulic flood modeling by combining:

1) raw DEM cleaning,
2) selective artifact smoothing,
3) optional automatic D4 river detection,
4) optional HydroPol2D-style creek DEM reduction using:

       Wshed_Properties.fac_area = D4 flow accumulation area [km2]
       idx_facc = fac_area >= GIS_data.min_area
       River_Width = beta_1 * fac_area^beta_2
       River_Depth = alfa_1 * fac_area^alfa_2
       H_abg = ((River_Width / Resolution) * River_Depth^(5/3))^(3/5)
       DEM = DEM - H_abg

5) optional vector-based river/crossing burning,
6) depression breaching/final hydrologic conditioning,
7) diagnostics and modification audit layers.

Important modeling philosophy
-----------------------------
The script does NOT assume that a perfectly draining DEM is always the best flood-model DEM.
It avoids blind global sink filling and global smoothing because these operations can destroy:
real floodplain storage, levees, road crowns, embankments, river banks, and channel-floodplain controls.

The final output must be inspected before use in HydroPol2D, HEC-RAS, LISFLOOD-FP, or any other
2D flood model. The most important inspection layer is:

    DEM_modification_final_minus_cleaned.tif

Negative values show lowering; positive values show raising.

Core outputs
------------
Outputs are saved in the Outputs folder by default:

    DEM_resampled_<resolution>m.tif                  [if --resample-dem]
    DEM_raw_cleaned.tif
    DEM_artifact_smoothed.tif
    DEM_reduced_creeks_D4.tif                         [if --auto-rivers-d4]
    DEM_hydraulic_conditioned.tif                     [final DEM]
    DEM_modification_final_minus_cleaned.tif          [audit layer]
    slope_before_deg.tif
    slope_after_deg.tif
    diagnostic_d8_flow_accum_log.tif
    diagnostic_dinf_flow_accum_log.tif
    diagnostic_no_flow_cells.tif
    diagnostic_sinks.tif
    modification_summary.csv
    conditioning_config.json
    quicklook_dem_conditioning.png
    DEM_conditioning_README.md

Automatic D4 HydroPol2D channel outputs
---------------------------------------
When --auto-rivers-d4 is used:

    D4_flow_accum_cells.tif
    DEM_D4_monotonic_routing_surface.tif
    D4_Wshed_Properties_fac_area_km2.tif
    D4_idx_facc.tif
    D4_external_geometry_available.tif    [if --river-geometry-source is external*]
    D4_external_geometry_used.tif         [if --river-geometry-source is external*]
    D4_Wshed_Properties_River_Width_m.tif
    D4_Wshed_Properties_River_Depth_m.tif
    D4_H_abg_m.tif
    D4_HydroPol2D_creek_reduction_summary.csv

Installation
------------
Recommended:

    conda create -n dem_processing python=3.11 -y
    conda activate dem_processing
    conda install -c conda-forge rasterio geopandas shapely pyproj scipy matplotlib pandas numpy -y
    python3 -m pip install -e .

Minimal run
-----------
Uses the default DEM path and saves to Outputs:

    hydrobathydem-condition

HydroPol2D D4 automatic creek reduction run
-------------------------------------------
Example only; replace coefficients with your calibrated/selected values:

    hydrobathydem-condition \
        --auto-rivers-d4 \
        --min-area 5 \
        --beta-1 5.0 \
        --beta-2 0.50 \
        --alfa-1 0.30 \
        --alfa-2 0.30 \
        --carve-mode wide \
        --slope-percentile 99.9 \
        --breach-dist-cells 100

Print full method documentation
-------------------------------

    hydrobathydem-condition --documentation
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.fill import fillnodata
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject
from scipy import ndimage

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dem_processing_matplotlib")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import whitebox
except Exception:
    whitebox = None

try:
    from .config import explicit_cli_flags, load_config_file
    from .paths import PROJECT_ROOT, ensure_output_layout, themed_output_path
    from .qa import build_qa_scorecard, write_run_manifest
except ImportError:  # Allows direct execution from the source directory.
    from config import explicit_cli_flags, load_config_file
    from paths import PROJECT_ROOT, ensure_output_layout, themed_output_path
    from qa import build_qa_scorecard, write_run_manifest


# =============================================================================
# Defaults and full documentation string
# =============================================================================

DEFAULT_DEM_PATH = str(PROJECT_ROOT / "DEM_fabdem.tif")

FULL_DOCUMENTATION = r"""
# HydroPol2D DEM Conditioning Tool — Full Documentation

## 1. Goal

This tool conditions a Digital Elevation Model (DEM) so it is more useful for
hydrologic and hydrodynamic flood modeling. It is designed around the needs of
HydroPol2D-style workflows, where the DEM controls floodplain gradients,
channel conveyance, wetting/drying, local ponding, and numerical stability.

The workflow is intentionally conservative. It does not blindly fill all sinks
or globally smooth the terrain. Instead, it creates a reproducible sequence of
intermediate products and audit layers so you can inspect exactly what was
changed.

Default input DEM:

    DEM_fabdem.tif

Default output directory:

    Outputs


## 2. Main processing stages

### Stage 0 — Optional DEM resampling to model resolution

Creates:

    DEM_resampled_<resolution>m.tif

If activated, this is the first operation. The input DEM is resampled to the
target model/grid resolution before cleaning, slope-artifact smoothing, D4 flow
accumulation, HydroPol2D creek reduction, and depression breaching.

Important parameters:

    --resample-dem
    --target-resolution-m 1000
    --resampling-method average

Recommended for HydroPol2D regional applications when the original DEM is finer
than the model grid. For coarsening a DEM, `average` is usually a stable first
choice; `bilinear` is also reasonable if you want a smoother continuous terrain.
For categorical rasters, never use this DEM resampling option.


### Stage 1 — Raw DEM cleaning

Creates:

    DEM_raw_cleaned.tif

This step reads the DEM, checks the coordinate reference system, converts the
raster to Float32, and optionally fills small NoData holes.

Activated by default.

Important parameters:

    --no-fill-nodata
    --max-nodata-fill-pixels 10

Use `--no-fill-nodata` if NoData regions are large or physically meaningful.


### Stage 2 — Selective extreme-slope artifact smoothing

Creates:

    DEM_artifact_smoothed.tif
    slope_before_deg.tif
    mask_extreme_slope_artifacts.tif

The script computes a slope raster and identifies the steepest cells using a
percentile threshold, for example the 99.8th or 99.9th percentile. It then
uses WhiteboxTools feature-preserving smoothing, but only replaces values in
cells identified as extreme-slope artifacts.

Activated by default.

Important parameters:

    --no-smooth-artifacts
    --slope-percentile 99.8
    --slope-threshold-deg 70
    --smooth-filter-cells 7

Recommended first test:

    --slope-percentile 99.9 --smooth-filter-cells 5

Interpretation:

- Higher slope percentile = fewer cells modified.
- Lower slope percentile = more aggressive smoothing.
- Larger smoothing filter = smoother replacement surface.

Warning:

Real levees, channel banks, road embankments, and cliffs can also appear as
extreme slopes. Always inspect `mask_extreme_slope_artifacts.tif` and
`DEM_modification_final_minus_cleaned.tif`.


### Stage 3 — Hydrologic conditioning before stream extraction

Creates:

    DEM_01_single_cell_pits_breached.tif
    DEM_02_breached.tif
    DEM_hydrologically_conditioned_pre_bathymetry.tif

This stage conditions the DEM first so the D4 stream network is extracted from
the best available routing surface.


### Stage 4 — Automatic D4 river detection and HydroPol2D creek reduction

Creates, when activated:

    DEM_D4_monotonic_routing_surface.tif
    D4_flow_direction.tif
    D4_flow_accum_cells.tif
    D4_Wshed_Properties_fac_area_km2.tif
    D4_idx_facc.tif
    D4_external_geometry_available.tif    [if direct external geometry is used]
    D4_external_geometry_used.tif         [if direct external geometry is used]
    D4_spatial_coefficients_used.tif      [if spatial coefficient rasters are used]
    D4_Wshed_Properties_River_Width_m.tif
    D4_Wshed_Properties_River_Depth_m.tif
    D4_H_abg_m.tif
    DEM_reduced_creeks_D4.tif
    D4_HydroPol2D_creek_reduction_summary.csv

Activated with:

    --auto-rivers-d4

This stage reproduces your HydroPol2D creek-lowering logic using Python.
Only D4 routing is used: north, south, east, or west. Diagonal D8 routing is
not used.

The script computes:

    Wshed_Properties.fac_area = D4 flow accumulation area [km2]

Then river cells are identified as:

    idx_facc = Wshed_Properties.fac_area >= GIS_data.min_area

By default, the river width raster is estimated as:

    Wshed_Properties.River_Width = beta_1 * fac_area^beta_2

and the river depth raster is estimated as:

    Wshed_Properties.River_Depth = alfa_1 * fac_area^alfa_2

External geometry can also be used:

    --river-geometry-source external
    --river-geometry-source external_or_power_law
    --external-river-width-raster /path/to/width.tif
    --external-river-depth-raster /path/to/depth.tif

Use `external_or_power_law` to use external width/depth where they overlap the
D4 river mask, while keeping the HydroPol2D power-law values for river cells
that do not overlap the external rasters.

For coarse domains where DEM-only routing creates implausible cross-ridge links,
the D4 network itself can be supplied as directed reaches:

    --external-river-network /path/to/HydroRIVERS_as.shp

The default fields match HydroRIVERS v1.0: `HYRIV_ID`, `NEXT_DOWN`, and
`UPLAND_SKM`. Reaches are snapped to cardinal grid cells, so the external data
sets network topology while the model DEM still supplies channel elevations.

Spatially calibrated coefficient rasters can also be used:

    --river-geometry-source spatial_coefficients_or_power_law
    --spatial-beta-1-raster /path/to/D4_beta_1_width_5000km2.tif
    --spatial-beta-2-raster /path/to/D4_beta_2_width_5000km2.tif
    --spatial-alfa-1-raster /path/to/D4_alfa_1_depth_5000km2.tif
    --spatial-alfa-2-raster /path/to/D4_alfa_2_depth_5000km2.tif

The spatial calibration script delineates FABDEM-D4 subcatchments, assigns Lin
samples to those zones, screens likely DEM/network mismatches, fits robust local
power laws, and writes coefficient maps:

    hydrobathydem-calibrate-hydraulics --selected-threshold-km2 5000

The current runner uses the 5000 km2 coefficient maps because that tested
threshold had the best validation performance among 500, 1000, 2500, and
5000 km2.

The included Lin et al. 2020 preparation script creates external rasters from:

    https://zenodo.org/records/3552776

Run:

    hydrobathydem-prepare-lin2020 --download

This writes:

    Data/Lin2020_bankfull_width/processed/Lin2020_width_m_1000m.tif
    Data/Lin2020_bankfull_width/processed/Lin2020_depth_Q2_m_1000m.tif

Depth is solved from Lin `width_m`, `Q2`, and `Slp` using the rectangular
Manning equation:

    Q2 = (1/n) * A * R^(2/3) * S^(1/2)
    A = width_m * depth
    R = A / (width_m + 2 * depth)

The DEM lowering depth is:

    H_abg = ((River_Width / Resolution) * River_Depth^(5/3))^(3/5)

Then:

    DEM_reduced = DEM - H_abg

Important parameters:

    --auto-rivers-d4
    --river-geometry-source spatial_coefficients_or_power_law
    --spatial-beta-1-raster Data/Lin2020_bankfull_width/calibration/D4_beta_1_width_5000km2.tif
    --spatial-beta-2-raster Data/Lin2020_bankfull_width/calibration/D4_beta_2_width_5000km2.tif
    --spatial-alfa-1-raster Data/Lin2020_bankfull_width/calibration/D4_alfa_1_depth_5000km2.tif
    --spatial-alfa-2-raster Data/Lin2020_bankfull_width/calibration/D4_alfa_2_depth_5000km2.tif
    --min-area 5
    --beta-1 5.0
    --beta-2 0.50
    --alfa-1 0.30
    --alfa-2 0.30
    --carve-mode wide
    --channel-cell-width-m 30
    --river-width-cap-m 200
    --river-depth-cap-m 15
    --max-H-abg-m 100

Nomenclature follows HydroPol2D:

    --min-area = GIS_data.min_area [km2]
    --beta-1   = GIS_data.beta_1
    --beta-2   = GIS_data.beta_2
    --alfa-1   = GIS_data.alfa_1
    --alfa-2   = GIS_data.alfa_2
    --max-H-abg-m = safety cap on H_abg [m]

Recommended first test:

    --auto-rivers-d4 --min-area 5 --beta-1 5.0 --beta-2 0.50 --alfa-1 0.30 --alfa-2 0.30

Interpretation of `H_abg`:

It is not simply the estimated river depth. It is the equivalent one-cell DEM
lowering depth that preserves approximate in-bank Manning conveyance when a
subgrid river is represented inside one model cell.


### Stage 4 — Optional vector stream burning

Creates, when activated:

    mask_stream_burn.tif
    DEM_channel_crossing_enforced.tif

Activated with:

    --streams /path/to/rivers.shp --stream-burn-depth-m 1.0

This lowers DEM elevations within a buffer around a vector stream network.

Important parameters:

    --streams rivers.shp
    --stream-buffer-m 20
    --stream-burn-depth-m 1.0

Use this only if you trust the vector river alignment. If the vector line is
misaligned with the DEM valley bottom, stream burning can create artificial
canals across hillslopes.


### Stage 5 — Optional crossing / culvert burning

Creates, when activated:

    mask_crossing_burn.tif
    DEM_channel_crossing_enforced.tif

Activated with:

    --crossings /path/to/crossings.shp --crossing-burn-depth-m 2.0

This locally lowers road-stream crossings, bridge decks, or culvert openings.
It is useful when a lidar DEM represents a bridge or road fill as a solid dam.

Important parameters:

    --crossings crossings.shp
    --crossing-buffer-m 10
    --crossing-burn-depth-m 2.0

Warning:

Do not burn real levees, dams, or embankments that should block flow.


### Stage 6 — Final hydraulic DEM assembly

Creates:

    DEM_hydraulic_conditioned.tif

Activated by default.

Important parameters:

    --breach-dist-cells 100
    --breach-max-cost VALUE
    --breach-flat-increment 0.01
    --no-least-cost-breaching
    --breach-fill-remaining
    --fill-residual-depressions
    --fill-max-depth-m 0.25

Recommended first test:

    --breach-dist-cells 100

Conservative residual filling, only after inspection:

    --fill-residual-depressions --fill-max-depth-m 0.25

Warning:

Filling depressions can destroy real floodplain storage. Breaching is generally
safer than filling for flood modeling.


### Stage 7 — Diagnostics and audit layers

Creates:

    DEM_modification_final_minus_cleaned.tif
    diagnostic_d8_flow_accum_log.tif
    diagnostic_dinf_flow_accum_log.tif
    diagnostic_no_flow_cells.tif
    diagnostic_sinks.tif
    diagnostic_depth_in_sink.tif
    slope_after_deg.tif
    modification_summary.csv
    quicklook_dem_conditioning.png
    diagnostic_pipeline_stages.png
    diagnostic_smoothing_artifacts.png
    diagnostic_d4_river_extraction.png
    diagnostic_hydraulic_geometry_histograms.png
    diagnostic_final_modifications.png

The most important audit raster is:

    DEM_modification_final_minus_cleaned.tif

Interpretation:

    negative values = DEM was lowered
    positive values = DEM was raised
    zero values     = unchanged terrain

Inspect this raster before using `DEM_hydraulic_conditioned.tif` in HydroPol2D.


## 3. Recommended test cases

### Test 1 — Very conservative cleaning + breaching only

Purpose: establish a baseline with minimal changes.

    hydrobathydem-condition \
        --out-dir Outputs/Test_01_conservative \
        --no-smooth-artifacts \
        --breach-dist-cells 50

Inspect:

    DEM_hydraulic_conditioned.tif
    DEM_modification_final_minus_cleaned.tif
    diagnostic_d8_flow_accum_log.tif


### Test 2 — Conservative artifact smoothing

Purpose: remove only the most suspicious slope spikes.

    hydrobathydem-condition \
        --out-dir Outputs/Test_02_smoothing \
        --slope-percentile 99.9 \
        --smooth-filter-cells 5 \
        --breach-dist-cells 50

Inspect:

    mask_extreme_slope_artifacts.tif
    slope_before_deg.tif
    slope_after_deg.tif
    DEM_modification_final_minus_cleaned.tif


### Test 3 — Automatic HydroPol2D D4 creek reduction

Purpose: generate a synthetic D4 drainage network and carve creek cells using
HydroPol2D hydraulic geometry.

    hydrobathydem-condition \
        --out-dir Outputs/Test_03_D4_creeks \
        --auto-rivers-d4 \
        --min-area 5 \
        --beta-1 5.0 \
        --beta-2 0.50 \
        --alfa-1 0.30 \
        --alfa-2 0.30 \
        --carve-mode wide \
        --slope-percentile 99.9 \
        --smooth-filter-cells 5 \
        --breach-dist-cells 100

Inspect:

    D4_Wshed_Properties_fac_area_km2.tif
    D4_idx_facc.tif
    D4_Wshed_Properties_River_Width_m.tif
    D4_Wshed_Properties_River_Depth_m.tif
    D4_H_abg_m.tif
    DEM_reduced_creeks_D4.tif
    DEM_modification_final_minus_cleaned.tif


### Test 4 — Sensitivity to min_area

Purpose: understand how river initiation threshold affects carving.

Small river network:

    --min-area 20

Dense river network:

    --min-area 1

Compare:

    D4_idx_facc.tif
    D4_H_abg_m.tif
    modification_summary.csv


### Test 5 — Sensitivity to hydraulic geometry coefficients

Purpose: evaluate uncertainty in width/depth assumptions.

Narrow/shallow:

    --beta-1 3.0 --beta-2 0.45 --alfa-1 0.20 --alfa-2 0.25

Wider/deeper:

    --beta-1 7.0 --beta-2 0.55 --alfa-1 0.40 --alfa-2 0.35

Inspect:

    D4_Wshed_Properties_River_Width_m.tif
    D4_Wshed_Properties_River_Depth_m.tif
    D4_H_abg_m.tif


### Test 6 — With known bridge/culvert crossings

Purpose: avoid artificial road or bridge dams.

    hydrobathydem-condition \
        --out-dir Outputs/Test_06_crossings \
        --auto-rivers-d4 \
        --min-area 5 \
        --beta-1 5.0 --beta-2 0.50 \
        --alfa-1 0.30 --alfa-2 0.30 \
        --crossings /path/to/crossings.shp \
        --crossing-buffer-m 10 \
        --crossing-burn-depth-m 2.0

Inspect:

    mask_crossing_burn.tif
    DEM_modification_final_minus_cleaned.tif


## 4. Recommended acceptance checks before HydroPol2D

1. Open `DEM_modification_final_minus_cleaned.tif`.
   Confirm changes are concentrated in channels, pits, or known artifacts.

2. Open `D4_idx_facc.tif`.
   Confirm the automatic river network follows valleys and does not cross ridges.

3. Open `D4_H_abg_m.tif`.
   Confirm DEM lowering depths are physically plausible.

4. Open `diagnostic_d8_flow_accum_log.tif`.
   Confirm drainage follows expected valleys.

5. Compare `slope_before_deg.tif` and `slope_after_deg.tif`.
   Confirm artifact smoothing did not erase banks, levees, or important embankments.

6. Inspect `modification_summary.csv`.
   If the percent of modified cells is unexpectedly large, reduce aggressiveness.


## 5. Practical parameter guidance

### min_area

Controls where synthetic rivers begin.

- Smaller value: denser river network, more DEM carving.
- Larger value: only larger channels are carved.

Start with:

    --min-area 5

Then test 1, 2, 10, and 20 km2.


### beta_1 and beta_2

Control river width:

    River_Width = beta_1 * fac_area^beta_2

Increasing `beta_1` or `beta_2` increases width and usually increases H_abg.


### alfa_1 and alfa_2

Control river depth:

    River_Depth = alfa_1 * fac_area^alfa_2

Increasing these parameters has a strong effect on H_abg because depth appears
with a 5/3 power inside the wide-channel conveyance formula.


### H_abg safety

If `D4_H_abg_m.tif` contains unrealistically large values, reduce coefficients
or set caps:

    --river-width-cap-m 100
    --river-depth-cap-m 10
    --max-H-abg-m 30


### slope-percentile

Controls selective smoothing aggressiveness.

- 99.95: very conservative.
- 99.9: conservative.
- 99.8: moderate.
- 99.5: aggressive.

For flood models, start conservative.


### breach-dist-cells

Controls the search distance for least-cost breaching.

- 25–50: conservative, local breaching.
- 100: moderate.
- 200+: aggressive; inspect carefully.


## 6. Final product for HydroPol2D

Use this as the conditioned DEM:

    DEM_hydraulic_conditioned.tif

But keep these alongside it for reproducibility:

    conditioning_config.json
    DEM_modification_final_minus_cleaned.tif
    modification_summary.csv
    D4_HydroPol2D_creek_reduction_summary.csv

"""

CURRENT_DOCUMENTATION = f"""# DEM Conditioning Run Documentation

This run was produced by HydroBathyDEM's packaged source layout in
`src/dem_processing`.

## Active Modules

```text
src/dem_processing/condition_dem.py
src/dem_processing/run_conditioning_1000m.py
src/dem_processing/prepare_lin2020_bankfull_geometry.py
src/dem_processing/calibrate_spatial_hydraulic_geometry.py
src/dem_processing/preflight.py
src/dem_processing/config.py
src/dem_processing/qa.py
```

## Output Layout

```text
Outputs/dem/          DEM stages and final conditioned DEM
Outputs/d4/           D4 routing, accumulation, river mask, width/depth/H_abg
Outputs/diagnostics/  diagnostic plots and diagnostic rasters
Outputs/reports/      run configuration, summaries, and this README
```

## Main Products

```text
Outputs/dem/DEM_hydraulic_conditioned.tif
Outputs/dem/DEM_modification_final_minus_cleaned.tif
Outputs/d4/D4_idx_facc.tif
Outputs/d4/D4_H_abg_m.tif
Outputs/d4/D4_Wshed_Properties_fac_area_km2.tif
Outputs/d4/D4_power_law_fallback_used.tif
Outputs/diagnostics/diagnostic_d4_river_extraction.png
Outputs/diagnostics/diagnostic_d4_geometry_source_map.png
Outputs/diagnostics/diagnostic_final_modifications.png
Outputs/reports/D4_HydroPol2D_creek_reduction_summary.csv
Outputs/reports/modification_summary.csv
Outputs/reports/qa_scorecard.csv
Outputs/reports/run_manifest.json
```

## Current Pipeline Philosophy

The model river network and drainage area come from FABDEM-D4. Lin et al. 2020
is used to calibrate spatial hydraulic-geometry coefficients, not to replace
the FABDEM-derived D4 network.

Width/depth hierarchy:

```text
local Lin-calibrated subcatchment fit
global Lin-calibrated fit for sparse/noisy calibration zones
base HydroPol2D power law only for coefficient nodata holes
```

## Rebuild Commands

```bash
hydrobathydem-condition --help
hydrobathydem-preflight --config configs/india_1000m_spatial.json --require-lin --require-spatial-coefficients
hydrobathydem-prepare-lin2020 --download
hydrobathydem-calibrate-hydraulics --selected-threshold-km2 5000 --fit-area-source d4
hydrobathydem-condition-1000m --config configs/india_1000m_spatial.json --dry-run
hydrobathydem-condition-1000m --config configs/india_1000m_spatial.json
```
"""


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DEMConditioningConfig:
    dem: str = DEFAULT_DEM_PATH
    out_dir: Optional[str] = None
    original_dem: Optional[str] = None
    config_path: Optional[str] = None

    # Optional first step: resample input DEM to model/grid resolution
    resample_dem: bool = False
    target_resolution_m: Optional[float] = None
    resampling_method: str = "average"

    # NoData cleaning
    fill_nodata: bool = True
    max_nodata_fill_pixels: int = 10

    # Selective smoothing of unrealistic slope artifacts
    smooth_artifacts: bool = True
    smooth_filter_cells: int = 5
    slope_percentile: float = 99.9
    slope_threshold_deg: Optional[float] = None
    protect_stream_buffer_m: float = 2000.0
    allow_stream_smoothing: bool = False

    # Optional stream/channel enforcement
    streams: Optional[str] = None
    stream_buffer_m: float = 15.0
    stream_burn_depth_m: float = 0.0

    # Optional culvert / crossing enforcement
    # This can be a point or line vector marking places where roads/bridges block flow.
    crossings: Optional[str] = None
    crossing_buffer_m: float = 10.0
    crossing_burn_depth_m: float = 0.0

    # Automatic synthetic river detection and hydraulic carving from D4 flow accumulation
    auto_rivers_d4: bool = False
    river_geometry_source: str = "power_law"
    external_river_width_raster: Optional[str] = None
    external_river_depth_raster: Optional[str] = None
    external_geometry_min_width_m: float = 1.0
    external_geometry_min_depth_m: float = 0.01
    external_river_network: Optional[str] = None
    external_network_id_field: str = "HYRIV_ID"
    external_network_next_down_field: str = "NEXT_DOWN"
    external_network_area_field: str = "UPLAND_SKM"
    external_network_snap_radius_cells: int = 0
    external_network_profile_from_source_dem: bool = False
    spatial_beta_1_raster: Optional[str] = None
    spatial_beta_2_raster: Optional[str] = None
    spatial_alfa_1_raster: Optional[str] = None
    spatial_alfa_2_raster: Optional[str] = None
    # HydroPol2D nomenclature for synthetic river geometry
    # idx_facc = Wshed_Properties.fac_area >= GIS_data.min_area
    # River_Width = GIS_data.beta_1 * fac_area ^ GIS_data.beta_2
    # River_Depth = GIS_data.alfa_1 * fac_area ^ GIS_data.alfa_2
    # where fac_area is upstream contributing area in km2.
    min_area: float = 100.0
    beta_1: float = 2.2695
    beta_2: float = 0.4942
    alfa_1: float = 0.1097
    alfa_2: float = 0.3856

    # Channel carving method:
    #   "wide"         -> same formula as your MATLAB code:
    #                     h_eq = ((B / dx) * H^(5/3))^(3/5)
    #   "manning_exact" -> solves A R^(2/3) equivalence for rectangular channels
    carve_mode: str = "wide"
    channel_cell_width_m: Optional[float] = None
    river_width_cap_m: Optional[float] = None
    river_depth_cap_m: Optional[float] = None
    max_H_abg_m: float = 50.0
    d4_routing_max_raise_m: Optional[float] = None
    condition_d4_channel_bed: bool = True
    channel_bed_min_slope: float = 1e-5

    # Whitebox breaching
    use_least_cost_breaching: bool = True
    breach_dist_cells: int = 10000
    breach_max_cost: Optional[float] = None
    breach_min_dist: bool = True
    breach_flat_increment: Optional[float] = 0.01
    breach_fill_remaining: bool = False

    # Fallback / conventional breach parameters
    breach_max_depth_m: Optional[float] = None
    breach_max_length_cells: Optional[int] = None

    # Residual filling: only use for very small pits, if needed
    fill_residual_depressions: bool = False
    fill_max_depth_m: Optional[float] = 0.25

    # Diagnostics
    make_plots: bool = True
    compression: str = "deflate"

    # Safety
    allow_geographic_crs: bool = False


# =============================================================================
# General utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_path(out_dir: Path, filename: str) -> Path:
    return themed_output_path(out_dir, filename)


def check_dependencies() -> None:
    missing = []
    if whitebox is None:
        missing.append("whitebox")
    if gpd is None:
        missing.append("geopandas")
    if missing:
        raise ImportError(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            + "pip install whitebox rasterio geopandas shapely pyproj scipy matplotlib pandas numpy"
        )


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def raster_profile(path: Path) -> dict:
    with rasterio.open(path) as src:
        return src.profile.copy()


def read_raster(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
    return arr, profile


def valid_mask(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def make_safe_gtiff_profile(
    profile: dict,
    dtype: str = "float32",
    nodata: float = -9999.0,
    compress: str = "deflate",
) -> dict:
    """
    Build a clean GeoTIFF profile for output rasters.

    Rasterio source profiles can carry block-size metadata from the input DEM.
    After resampling or shape changes, inherited block sizes may be invalid and
    can trigger RasterBlockError. Every writer should pass through this helper.
    """
    prof = profile.copy()

    for key in [
        "blockxsize",
        "blockysize",
        "tiled",
        "interleave",
        "photometric",
        "compress",
        "predictor",
        "zlevel",
        "BIGTIFF",
    ]:
        prof.pop(key, None)

    prof.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return prof


def write_raster(
    path: Path,
    arr: np.ndarray,
    profile: dict,
    nodata: float = -9999.0,
    dtype: str = "float32",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = make_safe_gtiff_profile(profile, dtype=dtype, nodata=nodata)
    out = arr.astype(dtype, copy=True)
    if np.issubdtype(out.dtype, np.floating):
        out[~np.isfinite(out)] = nodata
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def normalize_existing_raster(path: Path, dtype: str = "float32") -> None:
    """Rewrite an externally generated raster with the safe GeoTIFF profile."""
    path = Path(path)
    if not path.exists():
        return

    arr, profile = read_raster(path)
    nodata = profile.get("nodata")
    if nodata is None:
        nodata = -9999.0

    tmp_path = path.with_name(f"{path.stem}.tmp_safe{path.suffix}")
    write_raster(tmp_path, arr, profile, nodata=nodata, dtype=dtype)
    tmp_path.replace(path)
    print(f"[NORMALIZED] {path}")


def get_cellsize(profile: dict) -> Tuple[float, float]:
    transform: Affine = profile["transform"]
    dx = abs(transform.a)
    dy = abs(transform.e)
    return dx, dy


def assert_projected_meters(profile: dict, allow_geographic: bool = False) -> None:
    crs = profile.get("crs")
    if crs is None:
        raise ValueError(
            "DEM has no CRS. Assign the correct projected CRS before hydrologic conditioning."
        )
    if crs.is_geographic and not allow_geographic:
        raise ValueError(
            "DEM CRS is geographic degrees. Reproject to a projected CRS in meters first.\n"
            "For flood modeling, all slope, buffer, burn depth, and distance operations should be in meters.\n"
            "Use --allow-geographic-crs only for diagnostics, not for production conditioning."
        )


def get_resampling_enum(method: str) -> Resampling:
    """Convert a user string to a rasterio Resampling enum."""
    method = str(method).lower().strip()
    valid = {name: getattr(Resampling, name) for name in dir(Resampling) if not name.startswith("_")}
    if method not in valid:
        allowed = ", ".join(sorted(valid.keys()))
        raise ValueError(f"Unsupported resampling method: {method}. Allowed methods include: {allowed}")
    return valid[method]


def safe_resolution_label(resolution_m: float) -> str:
    """Return a filename-safe resolution label such as 1000m or 12p5m."""
    if abs(resolution_m - round(resolution_m)) < 1e-9:
        return f"{int(round(resolution_m))}m"
    return (f"{resolution_m:g}m").replace(".", "p")


def resample_dem_if_requested(cfg: DEMConditioningConfig, out_dir: Path) -> Path:
    """
    Optionally resample the input DEM to the target model resolution before any
    DEM conditioning. This ensures that D4 flow accumulation, channel carving,
    slope smoothing, and breaching are all applied to the same raster resolution
    used by the flood model.
    """
    input_path = Path(cfg.dem).expanduser().resolve()

    if not cfg.resample_dem:
        print_header("Step 0: DEM resampling")
        print("[SKIP] DEM resampling disabled. Using original DEM resolution.")
        return input_path

    if cfg.target_resolution_m is None or cfg.target_resolution_m <= 0:
        raise ValueError("--resample-dem requires --target-resolution-m > 0")

    print_header("Step 0: DEM resampling to model resolution")
    print(f"[INFO] Original DEM      : {input_path}")
    print(f"[INFO] Target resolution : {cfg.target_resolution_m} m")
    print(f"[INFO] Resampling method : {cfg.resampling_method}")

    resampling = get_resampling_enum(cfg.resampling_method)
    res = float(cfg.target_resolution_m)
    label = safe_resolution_label(res)
    out_path = output_path(out_dir, f"DEM_resampled_{label}.tif")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        assert_projected_meters(profile, allow_geographic=cfg.allow_geographic_crs)

        dx, dy = abs(src.transform.a), abs(src.transform.e)
        print(f"[INFO] Original resolution: dx = {dx:.6g} m, dy = {dy:.6g} m")

        if abs(dx - res) < 1e-6 and abs(dy - res) < 1e-6:
            print("[INFO] DEM is already at the requested resolution. Copying to output folder for provenance.")

        left, bottom, right, top = src.bounds
        dst_width = int(np.ceil((right - left) / res))
        dst_height = int(np.ceil((top - bottom) / res))
        dst_transform = from_origin(left, top, res, res)

        src_nodata = src.nodata
        dst_nodata = src_nodata if src_nodata is not None else -9999.0
        dst_arr = np.full((dst_height, dst_width), dst_nodata, dtype="float32")

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

        out_profile = make_safe_gtiff_profile(
            profile,
            dtype="float32",
            nodata=dst_nodata,
            compress=cfg.compression,
        )
        out_profile.update(
            height=dst_height,
            width=dst_width,
            transform=dst_transform,
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(dst_arr.astype("float32"), 1)

    print(f"[SAVED] {out_path}")
    return out_path


def compute_slope_deg_numpy(dem: np.ndarray, profile: dict, nodata: Optional[float]) -> np.ndarray:
    """
    Compute slope in degrees using central differences.
    This is for masking/diagnostics, not as a replacement for full terrain analysis.
    """
    dx, dy = get_cellsize(profile)
    data = dem.astype("float64", copy=True)
    mask = valid_mask(data, nodata)
    data[~mask] = np.nan

    # Fill temporary NaNs locally for gradient stability.
    # Output is masked back afterward.
    tmp = data.copy()
    if np.any(~np.isfinite(tmp)):
        mean_val = np.nanmean(tmp)
        tmp[~np.isfinite(tmp)] = mean_val

    dz_dy, dz_dx = np.gradient(tmp, dy, dx)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    slope[~mask] = np.nan
    return slope.astype("float32")


def rasterize_vector_mask(
    vector_path: str,
    profile: dict,
    buffer_m: float,
    burn_value: int = 1,
) -> np.ndarray:
    """
    Rasterize vector geometries to match the DEM.
    The vector is reprojected to the DEM CRS.
    """
    if gpd is None:
        raise ImportError("geopandas is required for vector rasterization.")

    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"Vector file is empty: {vector_path}")

    dem_crs = profile["crs"]
    if gdf.crs is None:
        raise ValueError(f"Vector file has no CRS: {vector_path}")

    gdf = gdf.to_crs(dem_crs)

    if buffer_m and buffer_m > 0:
        # Buffer in projected units. This assumes DEM CRS is in meters.
        geoms = gdf.geometry.buffer(buffer_m)
    else:
        geoms = gdf.geometry

    shapes = [(geom, burn_value) for geom in geoms if geom is not None and not geom.is_empty]

    mask = rasterize(
        shapes=shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return mask.astype(bool)


def selective_replace(base: np.ndarray, replacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    valid_repl = np.isfinite(replacement)
    out[mask & valid_repl] = replacement[mask & valid_repl]
    return out


def summarize_delta(
    cleaned_path: Path,
    final_path: Path,
    out_csv: Path,
) -> pd.DataFrame:
    clean, prof = read_raster(cleaned_path)
    final, _ = read_raster(final_path)
    nodata = prof.get("nodata", None)

    vm = valid_mask(clean, nodata) & np.isfinite(final)
    delta = np.full(clean.shape, np.nan, dtype="float32")
    delta[vm] = final[vm] - clean[vm]

    changed = vm & (np.abs(delta) > 1e-6)
    total_valid = int(vm.sum())
    n_changed = int(changed.sum())

    stats = {
        "total_valid_cells": total_valid,
        "changed_cells_abs_delta_gt_1e-6": n_changed,
        "changed_percent": 100.0 * n_changed / total_valid if total_valid > 0 else np.nan,
        "delta_min_m": float(np.nanmin(delta)) if total_valid > 0 else np.nan,
        "delta_max_m": float(np.nanmax(delta)) if total_valid > 0 else np.nan,
        "delta_mean_m": float(np.nanmean(delta)) if total_valid > 0 else np.nan,
        "delta_median_m": float(np.nanmedian(delta)) if total_valid > 0 else np.nan,
        "max_lowering_m": float(abs(np.nanmin(delta))) if total_valid > 0 else np.nan,
        "max_raising_m": float(np.nanmax(delta)) if total_valid > 0 else np.nan,
    }

    df = pd.DataFrame([stats])
    df.to_csv(out_csv, index=False)
    return df



# =============================================================================
# D4 automatic river detection and hydraulic channel carving
# =============================================================================

def compute_d4_flow_accumulation(
    dem: np.ndarray,
    profile: dict,
    nodata: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute D4, not D8, flow accumulation.

    D4 means each cell can drain only to one of the four cardinal neighbours:
    north, south, west, or east. Diagonal routing is intentionally disabled.

    The receiver is selected as the cardinal neighbour with the steepest
    positive downslope gradient. Cells with no lower cardinal neighbour become
    outlets/sinks.

    Returns
    -------
    acc_cells : 2D float64 array
        Number of upstream contributing cells, including the cell itself.
    receiver : 2D int64 array
        Flattened index of the receiver cell, or -1 if no receiver.
    """
    rows, cols = dem.shape
    dx, dy = get_cellsize(profile)

    z = dem.astype("float64", copy=True)
    valid = valid_mask(z, nodata)
    z[~valid] = np.nan

    lin = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    receiver = -np.ones((rows, cols), dtype=np.int64)
    best_slope = np.zeros((rows, cols), dtype="float64")

    def update_receiver(center_slice, neigh_slice, dist):
        zc = z[center_slice]
        zn = z[neigh_slice]
        slope = (zc - zn) / dist
        valid_pair = np.isfinite(zc) & np.isfinite(zn) & (slope > 0)

        best_view = best_slope[center_slice]
        rec_view = receiver[center_slice]
        neigh_lin = lin[neigh_slice]

        mask = valid_pair & (slope > best_view)
        best_view[mask] = slope[mask]
        rec_view[mask] = neigh_lin[mask]

    # North receiver: current row 1..end drains to row 0..end-1
    update_receiver((slice(1, None), slice(None)), (slice(0, -1), slice(None)), dy)

    # South receiver
    update_receiver((slice(0, -1), slice(None)), (slice(1, None), slice(None)), dy)

    # West receiver
    update_receiver((slice(None), slice(1, None)), (slice(None), slice(0, -1)), dx)

    # East receiver
    update_receiver((slice(None), slice(0, -1)), (slice(None), slice(1, None)), dx)

    flat_valid = valid.ravel()
    flat_z = z.ravel()
    flat_receiver = receiver.ravel()

    acc = np.zeros(rows * cols, dtype="float64")
    acc[flat_valid] = 1.0

    valid_idx = np.flatnonzero(flat_valid)
    # Process from high to low elevation so upstream cells contribute to receivers.
    order = valid_idx[np.argsort(flat_z[valid_idx])[::-1]]

    for idx in order:
        rec = flat_receiver[idx]
        if rec >= 0:
            acc[rec] += acc[idx]

    return acc.reshape(rows, cols), receiver


def receiver_to_d4_direction(receiver: np.ndarray) -> np.ndarray:
    """
    Convert flattened receiver indices to simple D4 direction codes.

    Codes:
        0 = no receiver / local outlet
        1 = north
        2 = east
        3 = south
        4 = west
    """
    rows, cols = receiver.shape
    lin = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    diff = receiver - lin

    direction = np.zeros(receiver.shape, dtype="float32")
    direction[diff == -cols] = 1.0
    direction[diff == 1] = 2.0
    direction[diff == cols] = 3.0
    direction[diff == -1] = 4.0
    return direction


def _d4_line_cells(coords: np.ndarray, transform: Affine, shape: tuple[int, int]) -> list[int]:
    """Snap a line to an ordered, cardinal-neighbour sequence of grid cells."""
    rows, cols = shape
    cells: list[int] = []
    for x0, y0, x1, y1 in zip(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1]):
        r0, c0 = rasterio.transform.rowcol(transform, x0, y0)
        r1, c1 = rasterio.transform.rowcol(transform, x1, y1)
        r, c = int(r0), int(c0)
        target_r, target_c = int(r1), int(c1)
        if 0 <= r < rows and 0 <= c < cols and (not cells or cells[-1] != r * cols + c):
            cells.append(r * cols + c)
        while (r, c) != (target_r, target_c):
            if abs(target_c - c) >= abs(target_r - r) and c != target_c:
                c += 1 if target_c > c else -1
            elif r != target_r:
                r += 1 if target_r > r else -1
            if 0 <= r < rows and 0 <= c < cols and (not cells or cells[-1] != r * cols + c):
                cells.append(r * cols + c)
    return cells


def _d4_cells_between_cells(start: int, end: int, cols: int) -> list[int]:
    """Return the inclusive cardinal path between two already-snapped cells."""
    r, c = divmod(start, cols)
    target_r, target_c = divmod(end, cols)
    cells = [start]
    while (r, c) != (target_r, target_c):
        if abs(target_c - c) >= abs(target_r - r) and c != target_c:
            c += 1 if target_c > c else -1
        elif r != target_r:
            r += 1 if target_r > r else -1
        cells.append(r * cols + c)
    return cells


def _snap_d4_cells_to_local_lowest(
    cells: list[int],
    elevation: np.ndarray,
    nodata: float,
    radius: int,
) -> list[int]:
    """Snap a coarse centreline cell to its lowest valid local DEM cell."""
    if radius <= 0:
        return cells
    rows, cols = elevation.shape
    snapped: list[int] = []
    for cell in cells:
        row, col = divmod(cell, cols)
        r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
        c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
        window = elevation[r0:r1, c0:c1]
        valid = np.isfinite(window) & (window != nodata)
        if np.any(valid):
            candidate = np.where(valid, window, np.inf)
            wr, wc = np.unravel_index(np.argmin(candidate), candidate.shape)
            cell = (r0 + wr) * cols + c0 + wc
        if not snapped or snapped[-1] != cell:
            snapped.append(cell)
    return snapped


def _connect_d4_cells(cells: list[int], cols: int) -> list[int]:
    """Insert cardinal cells between snapped centreline locations."""
    connected: list[int] = []
    for start, end in zip(cells[:-1], cells[1:]):
        segment = _d4_cells_between_cells(start, end, cols)
        connected.extend(segment if not connected else segment[1:])
    return connected or cells


def _directed_d4_order(valid: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Return channel cells ordered from headwaters towards the outlets."""
    flat_valid = valid.ravel()
    flat_receiver = receiver.ravel()
    source = np.flatnonzero(flat_valid & (flat_receiver >= 0) & (flat_receiver < flat_valid.size))
    source = source[flat_valid[flat_receiver[source]]]
    indegree = np.zeros(flat_valid.size, dtype=np.int32)
    np.add.at(indegree, flat_receiver[source], 1)
    pending = list(np.flatnonzero(flat_valid & (indegree == 0)))
    ordered: list[int] = []
    cycle_breaks = 0
    while len(ordered) < int(flat_valid.sum()):
        while pending:
            src = pending.pop()
            ordered.append(src)
            dst = flat_receiver[src]
            if dst >= 0 and flat_valid[dst]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    pending.append(int(dst))
        if len(ordered) == int(flat_valid.sum()):
            break
        # Vector reaches are acyclic, but snapping several reaches to a raster
        # can create a tiny artificial loop.  Seed one member so the profile
        # remains usable; the adjustment raster records the consequence.
        remaining = np.flatnonzero(flat_valid & (indegree > 0))
        if not remaining.size:
            break
        pending.append(int(remaining[0]))
        indegree[remaining[0]] = 0
        cycle_breaks += 1
    if cycle_breaks:
        warnings.warn(f"Broke {cycle_breaks} raster-induced external-network cycle(s) while building the channel profile.")
    return np.asarray(ordered, dtype=np.int64)


def enforce_downstream_channel_surface(
    dem: np.ndarray,
    river_mask: np.ndarray,
    receiver: np.ndarray,
    profile: dict,
    min_slope: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lower river-cell terrain only enough to make its directed profile descend."""
    if not (dem.shape == river_mask.shape == receiver.shape):
        raise ValueError("Channel-profile DEM, mask, and receiver must have the same shape.")
    nodata = profile.get("nodata", -9999.0)
    valid = river_mask & valid_mask(dem, nodata)
    surface = dem.astype("float64", copy=True)
    flat_surface = surface.ravel()
    flat_receiver = receiver.ravel()
    dx, dy = get_cellsize(profile)
    drop = max(float(min_slope), 0.0) * min(dx, dy)
    for src in _directed_d4_order(valid, receiver):
        dst = flat_receiver[src]
        if dst >= 0 and valid.ravel()[dst]:
            flat_surface[dst] = min(flat_surface[dst], flat_surface[src] - drop)
    adjustment = np.zeros_like(surface)
    adjustment[valid] = dem[valid] - surface[valid]
    return surface.astype("float32"), adjustment.astype("float32")


def aggregate_channel_surface_minimum(
    source_surface: np.ndarray,
    source_profile: dict,
    target_profile: dict,
) -> np.ndarray:
    """Aggregate a river-only source surface to model cells using its minimum."""
    source_nodata = -9999.0
    source = np.where(np.isfinite(source_surface), source_surface, source_nodata).astype("float32")
    target = np.full((target_profile["height"], target_profile["width"]), source_nodata, dtype="float32")
    reproject(
        source=source,
        destination=target,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        src_nodata=source_nodata,
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        dst_nodata=source_nodata,
        resampling=Resampling.min,
    )
    target[target == source_nodata] = np.nan
    return target


def build_external_d4_river_network(
    vector_path: str,
    profile: dict,
    min_area_km2: float,
    id_field: str,
    next_down_field: str,
    area_field: str,
    elevation: Optional[np.ndarray] = None,
    nodata: float = -9999.0,
    snap_radius_cells: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a D4 network from directed reaches such as HydroRIVERS.

    The input lines must be directed upstream-to-downstream and contain a
    reach ID plus the ID of the next downstream reach.  At confluences, the
    largest upstream-area reach owns the single HydroPol D4 receiver; smaller
    tributaries terminate in that shared coarse cell and still contribute to
    its geometry/location.
    """
    if gpd is None:
        raise ImportError("geopandas is required for an external directed river network.")
    gdf = gpd.read_file(vector_path)
    required = [id_field, next_down_field, area_field]
    missing = [field for field in required if field not in gdf.columns]
    if missing:
        raise ValueError(f"External river network is missing required fields: {', '.join(missing)}")
    if gdf.crs is None:
        raise ValueError(f"External river network has no CRS: {vector_path}")
    gdf = gdf.to_crs(profile["crs"])
    bounds = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    gdf = gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].copy()
    gdf[area_field] = pd.to_numeric(gdf[area_field], errors="coerce")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & (gdf[area_field] >= min_area_km2)]
    if gdf.empty:
        raise ValueError("No external river reaches remain after domain clip and upstream-area filtering.")

    rows, cols = profile["height"], profile["width"]
    coordinates: dict[object, np.ndarray] = {}
    areas: dict[object, float] = {}
    next_down: dict[object, object] = {}
    for row in gdf.itertuples(index=False):
        record = row._asdict()
        geom = record["geometry"]
        if geom.geom_type == "MultiLineString":
            geom = max(geom.geoms, key=lambda part: part.length)
        if geom.geom_type != "LineString":
            continue
        rid = record[id_field]
        coordinates[rid] = np.asarray(geom.coords)
        areas[rid] = float(record[area_field])
        next_down[rid] = record[next_down_field]
    if not coordinates:
        raise ValueError("No external river reaches intersect the model grid.")

    upstream: dict[object, list[object]] = {}
    for rid, downstream_id in next_down.items():
        if downstream_id in coordinates:
            upstream.setdefault(downstream_id, []).append(rid)

    def endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(min(np.linalg.norm(a - b[0]), np.linalg.norm(a - b[-1])))

    paths: dict[object, list[int]] = {}
    for rid, coords in coordinates.items():
        downstream = coordinates.get(next_down[rid])
        if downstream is not None:
            if endpoint_distance(coords[0], downstream) < endpoint_distance(coords[-1], downstream):
                coords = coords[::-1]
        elif rid in upstream:
            upstream_coords = [coordinates[up] for up in upstream[rid]]
            start_distance = min(endpoint_distance(coords[0], up) for up in upstream_coords)
            end_distance = min(endpoint_distance(coords[-1], up) for up in upstream_coords)
            if end_distance < start_distance:
                coords = coords[::-1]
        cells = _d4_line_cells(coords, profile["transform"], (rows, cols))
        if elevation is not None:
            cells = _snap_d4_cells_to_local_lowest(cells, elevation, nodata, snap_radius_cells)
            cells = _connect_d4_cells(cells, cols)
        if cells:
            paths[rid] = cells
    if not paths:
        raise ValueError("No external river reaches remain after rasterization.")

    receiver = np.full(rows * cols, -1, dtype=np.int64)
    owner_area = np.full(rows * cols, -np.inf, dtype="float64")
    area = np.full(rows * cols, np.nan, dtype="float64")
    mask = np.zeros(rows * cols, dtype=bool)
    candidates: list[tuple[int, int, float]] = []
    for rid, cells in paths.items():
        weight = areas[rid]
        for cell in cells:
            mask[cell] = True
            if weight > owner_area[cell]:
                owner_area[cell] = weight
                area[cell] = weight
        candidates.extend((src, dst, weight) for src, dst in zip(cells[:-1], cells[1:]) if src != dst)
        downstream = paths.get(next_down.get(rid))
        if downstream and cells[-1] != downstream[0]:
            bridge = _d4_cells_between_cells(cells[-1], downstream[0], cols)
            if len(bridge) > 9:
                warnings.warn(
                    f"Skipping a nonlocal external-network bridge of {len(bridge) - 1} D4 cells for reach {rid}."
                )
                continue
            for cell in bridge:
                mask[cell] = True
                if weight > owner_area[cell]:
                    owner_area[cell] = weight
                    area[cell] = weight
            candidates.extend((src, dst, weight) for src, dst in zip(bridge[:-1], bridge[1:]))
    for src, dst, weight in candidates:
        if src != dst and weight >= owner_area[src]:
            receiver[src] = dst
    return mask.reshape(rows, cols), receiver.reshape(rows, cols), area.reshape(rows, cols)


def compute_d4_channel_bed_sills(
    dem: np.ndarray,
    river_depth: np.ndarray,
    river_mask: np.ndarray,
    receiver: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Neal channel-bed elevation and uphill D4 receiver links in metres.

    HydroPol2D's Neal solver routes channels over ``DEM - RiverDepth``.  The
    river network, however, is derived before hydraulic geometry is applied.
    This check identifies links that drain downhill on the routing DEM but
    become uphill after the assigned channel depth is applied.  It deliberately
    reports the issue without changing any geometry.
    """
    if not (dem.shape == river_depth.shape == river_mask.shape == receiver.shape):
        raise ValueError("DEM, river depth, river mask, and D4 receiver must have the same shape.")

    valid = river_mask & np.isfinite(dem) & np.isfinite(river_depth) & (river_depth > 0)
    bed = np.full(dem.shape, np.nan, dtype="float64")
    bed[valid] = dem[valid] - river_depth[valid]

    flat_valid = valid.ravel()
    flat_bed = bed.ravel()
    flat_receiver = receiver.ravel()
    source = np.flatnonzero(flat_valid & (flat_receiver >= 0) & (flat_receiver < flat_valid.size))
    source = source[flat_valid[flat_receiver[source]]]

    sill = np.full(flat_bed.size, np.nan, dtype="float64")
    rise = flat_bed[flat_receiver[source]] - flat_bed[source]
    sill[source] = np.maximum(rise, 0.0)
    return bed, sill.reshape(dem.shape)


def write_d4_channel_bed_qa(
    dem: np.ndarray,
    river_depth: np.ndarray,
    river_mask: np.ndarray,
    receiver: np.ndarray,
    profile: dict,
    out_dir: Path,
) -> dict:
    """Write a static Neal channel-bed QA raster and a compact summary."""
    bed, sill = compute_d4_channel_bed_sills(dem, river_depth, river_mask, receiver)
    write_raster(output_path(out_dir, "D4_Neal_channel_bed_elevation_m.tif"), bed.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_Neal_channel_receiver_sill_m.tif"), sill.astype("float32"), profile, nodata=-9999.0)

    values = sill[np.isfinite(sill)]
    positive = values[values > 0.01]
    summary = {
        "channel_bed_receiver_links": int(values.size),
        "channel_bed_receiver_sill_cells_gt_0p01m": int(positive.size),
        "channel_bed_receiver_sill_max_m": float(np.nanmax(positive)) if positive.size else 0.0,
        "channel_bed_receiver_sill_p99_m": float(np.nanpercentile(positive, 99)) if positive.size else 0.0,
    }
    pd.DataFrame([summary]).to_csv(output_path(out_dir, "D4_Neal_channel_bed_QA_summary.csv"), index=False)
    return summary


def condition_d4_channel_bed(
    dem: np.ndarray,
    river_depth: np.ndarray,
    river_mask: np.ndarray,
    receiver: np.ndarray,
    routing_surface: np.ndarray,
    profile: dict,
    min_slope: float,
    depth_cap_m: Optional[float],
    topology_order: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deepen only downstream channel cells needed for a D4-descending bed.

    The receiver graph is ordered from high to low routing elevation.  At each
    river link, the downstream bed is lowered only enough to preserve the
    requested minimum gradient.  A depth cap remains authoritative; residual
    inconsistencies are then visible in the post-condition QA rather than
    being silently hidden.
    """
    if not (dem.shape == river_depth.shape == river_mask.shape == receiver.shape == routing_surface.shape):
        raise ValueError("D4 channel-bed conditioning inputs must have the same shape.")

    dx, dy = get_cellsize(profile)
    drop = max(float(min_slope), 0.0) * min(dx, dy)
    conditioned = river_depth.astype("float64", copy=True)
    valid = river_mask & np.isfinite(dem) & np.isfinite(conditioned) & (conditioned > 0)
    flat_valid = valid.ravel()
    flat_dem = dem.ravel()
    flat_depth = conditioned.ravel()
    flat_receiver = receiver.ravel()
    flat_route = routing_surface.ravel()
    source = np.flatnonzero(flat_valid & (flat_receiver >= 0) & (flat_receiver < flat_valid.size))
    source = source[flat_valid[flat_receiver[source]]]
    if topology_order:
        indegree = np.zeros(flat_valid.size, dtype=np.int32)
        np.add.at(indegree, flat_receiver[source], 1)
        pending = list(np.flatnonzero(flat_valid & (indegree == 0)))
        ordered: list[int] = []
        while pending:
            src = pending.pop()
            ordered.append(src)
            dst = flat_receiver[src]
            if dst >= 0 and flat_valid[dst]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    pending.append(int(dst))
        source = np.asarray([idx for idx in ordered if flat_receiver[idx] >= 0 and flat_valid[flat_receiver[idx]]], dtype=np.int64)
    else:
        source = source[np.argsort(flat_route[source])[::-1]]
    cap = float(depth_cap_m) if depth_cap_m is not None else np.inf

    for src in source:
        dst = flat_receiver[src]
        max_downstream_bed = flat_dem[src] - flat_depth[src] - drop
        required_depth = flat_dem[dst] - max_downstream_bed
        if required_depth > flat_depth[dst]:
            flat_depth[dst] = min(required_depth, cap)

    adjustment = conditioned - river_depth
    return conditioned.astype("float32"), adjustment.astype("float32")


def build_d4_monotonic_routing_surface(
    dem_in: Path,
    out_dir: Path,
    flat_increment: Optional[float],
    max_raise_m: Optional[float],
) -> Path:
    """
    Create a routing-only D4 surface with a bounded artificial rise to D4 outlets.

    Whitebox conditioning is not guaranteed to leave a Float32 DEM with enough
    cardinal-neighbour relief for strict D4 routing. This priority-flood pass is
    used only for stream-network extraction. ``max_raise_m`` prevents distant
    mountain basins from being connected through an implausibly deep fill.
    """
    print_header("D4 routing surface conditioning")

    dem, profile = read_raster(dem_in)
    nodata = profile.get("nodata", -9999.0)
    valid = valid_mask(dem, nodata)

    eps = max(float(flat_increment or 0.0), 0.01)
    rows, cols = dem.shape
    filled = np.full(dem.shape, np.nan, dtype="float64")
    visited = np.zeros(dem.shape, dtype=bool)

    boundary = valid.copy()
    boundary[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & (
            ~valid[:-2, 1:-1]
            | ~valid[2:, 1:-1]
            | ~valid[1:-1, :-2]
            | ~valid[1:-1, 2:]
        )
    )
    boundary[0, :] &= valid[0, :]
    boundary[-1, :] &= valid[-1, :]
    boundary[:, 0] &= valid[:, 0]
    boundary[:, -1] &= valid[:, -1]

    heap: list[tuple[float, int]] = []
    boundary_idx = np.flatnonzero(boundary.ravel())
    flat_dem = dem.astype("float64", copy=False).ravel()
    flat_filled = filled.ravel()
    flat_visited = visited.ravel()
    flat_valid = valid.ravel()

    for idx in boundary_idx:
        z = float(flat_dem[idx])
        flat_filled[idx] = z
        flat_visited[idx] = True
        heapq.heappush(heap, (z, int(idx)))

    print(f"[INFO] D4 routing outlets/boundary cells = {len(boundary_idx):,}")
    print(f"[INFO] D4 routing flat increment = {eps:g} m")
    if max_raise_m is not None:
        print(f"[INFO] D4 routing maximum artificial raise = {float(max_raise_m):g} m")

    neighbour_offsets = (-cols, cols, -1, 1)
    processed = 0
    while heap:
        elev, idx = heapq.heappop(heap)
        r = idx // cols
        c = idx - r * cols
        processed += 1

        for offset in neighbour_offsets:
            nidx = idx + offset
            if offset == -cols and r == 0:
                continue
            if offset == cols and r == rows - 1:
                continue
            if offset == -1 and c == 0:
                continue
            if offset == 1 and c == cols - 1:
                continue
            if flat_visited[nidx] or not flat_valid[nidx]:
                continue

            new_elev = max(float(flat_dem[nidx]), elev + eps)
            if max_raise_m is not None:
                new_elev = min(new_elev, float(flat_dem[nidx]) + max(float(max_raise_m), 0.0))
            flat_filled[nidx] = new_elev
            flat_visited[nidx] = True
            heapq.heappush(heap, (new_elev, int(nidx)))

    missing = valid & ~visited
    if np.any(missing):
        warnings.warn(f"D4 routing surface left {int(missing.sum())} valid cells unvisited.")
        filled[missing] = dem[missing]

    out_path = output_path(out_dir, "DEM_D4_monotonic_routing_surface.tif")
    write_raster(out_path, filled.astype("float32"), profile, nodata=nodata)
    print(f"[INFO] D4 routing cells processed = {processed:,}")
    print(f"[SAVED] {out_path}")
    return out_path


def rectangular_conveyance_term(width: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Manning conveyance term without n and S:
        K* = A R^(2/3)

    Q = (1/n) A R^(2/3) S^(1/2)

    If the same n and S are assumed for the true channel and equivalent
    carved cell, matching A R^(2/3) matches in-bank flow capacity.
    """
    width = np.asarray(width, dtype="float64")
    depth = np.asarray(depth, dtype="float64")

    A = width * depth
    P = width + 2.0 * depth
    R = np.where(P > 0, A / P, 0.0)
    return A * np.power(np.maximum(R, 0.0), 2.0 / 3.0)


def compute_equivalent_H_abg(
    B: np.ndarray,
    H: np.ndarray,
    grid_width: float,
    mode: str = "wide",
    max_depth: float = 100.0,
) -> np.ndarray:
    """
    Compute the depth to lower a one-cell-wide synthetic channel so that it
    has approximately the same in-bank conveyance as the target subgrid river.

    Parameters
    ----------
    B : river width [m]
    H : river depth [m]
    grid_width : effective cell/channel width [m], usually DEM resolution
    mode : "wide" or "manning_exact"
    max_depth : safety cap for equivalent carved depth [m]

    Notes
    -----
    The "wide" formula is the same as the MATLAB logic:

        h_eq = ((B / dx) * H^(5/3))^(3/5)

    It comes from the wide rectangular Manning approximation:
        Q ∝ B H^(5/3)

    The exact rectangular method solves:
        (grid_width * h) * R_grid(h)^(2/3)
        =
        (B * H) * R_channel(B,H)^(2/3)
    """
    B = np.asarray(B, dtype="float64")
    H = np.asarray(H, dtype="float64")

    out = np.zeros_like(B, dtype="float64")
    valid = np.isfinite(B) & np.isfinite(H) & (B > 0) & (H > 0)

    if not np.any(valid):
        return out.astype("float32")

    if mode == "wide":
        out[valid] = np.power((B[valid] / grid_width) * np.power(H[valid], 5.0 / 3.0), 3.0 / 5.0)

    elif mode == "manning_exact":
        target = rectangular_conveyance_term(B[valid], H[valid])

        lo = np.zeros_like(target, dtype="float64")
        hi = np.full_like(target, max_depth, dtype="float64")

        # If the cap is too small for some cells, those cells will hit the cap.
        # This is intentional: the user should inspect the cap-hit mask/output.
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            k_mid = rectangular_conveyance_term(grid_width, mid)
            too_low = k_mid < target
            lo[too_low] = mid[too_low]
            hi[~too_low] = mid[~too_low]

        out[valid] = 0.5 * (lo + hi)

    else:
        raise ValueError("carve_mode must be either 'wide' or 'manning_exact'.")

    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, max_depth)
    return out.astype("float32")


def read_external_geometry_raster(path: str, target_profile: dict, label: str) -> np.ndarray:
    """Read an external width/depth raster and align it to the active DEM grid if needed."""
    raster_path = Path(path).expanduser()
    if not raster_path.exists():
        raise FileNotFoundError(f"{label} raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        src_arr = src.read(1).astype("float32")
        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr = np.where(src_arr == src_nodata, np.nan, src_arr)
        src_arr = np.where(np.isfinite(src_arr), src_arr, np.nan)

        same_grid = (
            src.height == target_profile["height"]
            and src.width == target_profile["width"]
            and src.transform == target_profile["transform"]
            and src.crs == target_profile["crs"]
        )
        if same_grid:
            return src_arr

        aligned = np.full((target_profile["height"], target_profile["width"]), np.nan, dtype="float32")
        reproject(
            source=src_arr,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    warnings.warn(f"{label} raster was reprojected/resampled to match the DEM grid: {raster_path}")
    return aligned


def wbt_preliminary_breach_for_d4_routing(wbt, cfg: DEMConditioningConfig, dem_in: Path, out_dir: Path) -> Path:
    """
    Create a preliminary breached DEM only for deriving a connected D4 drainage
    network. This DEM is not automatically the final hydraulic model terrain.
    """
    print_header("Preliminary breaching for automatic D4 river detection")

    pitless = output_path(out_dir, "DEM_preD4_single_cell_pits_breached.tif")
    routing_dem = output_path(out_dir, "DEM_prebreached_for_D4_routing.tif")

    print("[WBT] Breaching single-cell pits for D4 routing DEM...")
    wbt.breach_single_cell_pits(str(dem_in), str(pitless))
    normalize_existing_raster(pitless)

    if cfg.use_least_cost_breaching:
        print("[WBT] Least-cost breaching for D4 routing DEM...")
        wbt.breach_depressions_least_cost(
            str(pitless),
            str(routing_dem),
            cfg.breach_dist_cells,
            max_cost=cfg.breach_max_cost,
            min_dist=cfg.breach_min_dist,
            flat_increment=cfg.breach_flat_increment,
            fill=cfg.breach_fill_remaining,
        )
        normalize_existing_raster(routing_dem)
    else:
        print("[WBT] Conventional breaching for D4 routing DEM...")
        wbt.breach_depressions(
            str(pitless),
            str(routing_dem),
            max_depth=cfg.breach_max_depth_m,
            max_length=cfg.breach_max_length_cells,
            flat_increment=cfg.breach_flat_increment,
            fill_pits=False,
        )
        normalize_existing_raster(routing_dem)

    print(f"[SAVED] {routing_dem}")
    return routing_dem


def automatic_d4_hydraulic_channel_carving(
    cfg: DEMConditioningConfig,
    dem_to_carve: Path,
    routing_dem: Path,
    out_dir: Path,
) -> Path:
    """
    Automatically detect a synthetic D4 river network from flow accumulation,
    estimate hydraulic geometry from upstream area, and lower channel cells
    by an equivalent depth.

    This follows the logic in the user's MATLAB code:
        B = beta_1 * A^beta_2
        H = alfa_1 * A^alfa_2
        h_eq = ((B / Resolution) * H^(5/3))^(3/5)

    where A is upstream area in km2.
    """
    print_header("Automatic D4 river detection and hydraulic channel carving")

    dem_route, profile = read_raster(routing_dem)
    nodata = profile.get("nodata", -9999.0)

    dx, dy = get_cellsize(profile)
    cell_area_m2 = dx * dy
    default_grid_width = float(np.sqrt(cell_area_m2))
    grid_width = cfg.channel_cell_width_m if cfg.channel_cell_width_m else default_grid_width

    print(f"[INFO] D4 flow only: cardinal neighbours N/S/E/W; no diagonal flow.")
    print(f"[INFO] cell size dx={dx:.3f} m, dy={dy:.3f} m")
    print(f"[INFO] effective channel cell width = {grid_width:.3f} m")

    external_network_used = bool(cfg.external_river_network)
    if external_network_used:
        river_mask, receiver, fac_area = build_external_d4_river_network(
            cfg.external_river_network,
            profile,
            cfg.min_area,
            cfg.external_network_id_field,
            cfg.external_network_next_down_field,
            cfg.external_network_area_field,
            dem_route,
            nodata,
            cfg.external_network_snap_radius_cells,
        )
        acc_cells = fac_area * 1_000_000.0 / cell_area_m2
        print(f"[INFO] directed external river network = {cfg.external_river_network}")
    else:
        acc_cells, receiver = compute_d4_flow_accumulation(dem_route, profile, nodata)
        fac_area = acc_cells * cell_area_m2 / 1_000_000.0
        river_mask = np.isfinite(fac_area) & (fac_area >= cfg.min_area)
    n_river = int(river_mask.sum())
    n_valid = int(valid_mask(dem_route, nodata).sum())

    print(f"[INFO] min upstream area threshold = {cfg.min_area:.6g} km2")
    print(f"[INFO] auto river cells = {n_river} ({100*n_river/max(n_valid,1):.4f}% of valid DEM)")
    print(f"[INFO] D4 network derived from conditioned DEM: {routing_dem}")

    B_power = np.zeros_like(fac_area, dtype="float32")
    H_power = np.zeros_like(fac_area, dtype="float32")

    pos = river_mask & np.isfinite(fac_area) & (fac_area > 0)
    B_power[pos] = cfg.beta_1 * np.power(fac_area[pos], cfg.beta_2)
    H_power[pos] = cfg.alfa_1 * np.power(fac_area[pos], cfg.alfa_2)

    B = B_power.copy()
    H = H_power.copy()
    external_available = np.zeros_like(river_mask, dtype=bool)
    external_used = np.zeros_like(river_mask, dtype=bool)
    spatial_coefficients_used = np.zeros_like(river_mask, dtype=bool)
    fallback_used = np.zeros_like(river_mask, dtype=bool)

    valid_geometry_sources = {
        "power_law",
        "external",
        "external_or_power_law",
        "spatial_coefficients",
        "spatial_coefficients_or_power_law",
    }
    if cfg.river_geometry_source not in valid_geometry_sources:
        raise ValueError("river_geometry_source must be one of: " + ", ".join(sorted(valid_geometry_sources)) + ".")

    if cfg.river_geometry_source in {"external", "external_or_power_law"}:
        if not cfg.external_river_width_raster or not cfg.external_river_depth_raster:
            raise ValueError(
                "External river geometry requested. Provide both "
                "--external-river-width-raster and --external-river-depth-raster."
            )
        ext_B = read_external_geometry_raster(cfg.external_river_width_raster, profile, "External river width")
        ext_H = read_external_geometry_raster(cfg.external_river_depth_raster, profile, "External river depth")
        external_available = (
            np.isfinite(ext_B)
            & np.isfinite(ext_H)
            & (ext_B >= cfg.external_geometry_min_width_m)
            & (ext_H >= cfg.external_geometry_min_depth_m)
        )
        external_used = river_mask & external_available

        if cfg.river_geometry_source == "external":
            B = np.zeros_like(B_power, dtype="float32")
            H = np.zeros_like(H_power, dtype="float32")
        else:
            fallback_used = river_mask & ~external_available & (B_power > 0) & (H_power > 0)

        B[external_used] = ext_B[external_used]
        H[external_used] = ext_H[external_used]

        print(f"[INFO] river geometry source = {cfg.river_geometry_source}")
        print(f"[INFO] external geometry available cells = {int((river_mask & external_available).sum()):,}")
        print(f"[INFO] external geometry used cells = {int(external_used.sum()):,}")
        if cfg.river_geometry_source == "external_or_power_law":
            print(f"[INFO] power-law fallback river cells = {int(fallback_used.sum()):,}")
    elif cfg.river_geometry_source in {"spatial_coefficients", "spatial_coefficients_or_power_law"}:
        required = [
            cfg.spatial_beta_1_raster,
            cfg.spatial_beta_2_raster,
            cfg.spatial_alfa_1_raster,
            cfg.spatial_alfa_2_raster,
        ]
        if not all(required):
            raise ValueError(
                "Spatial coefficient geometry requested. Provide "
                "--spatial-beta-1-raster, --spatial-beta-2-raster, "
                "--spatial-alfa-1-raster, and --spatial-alfa-2-raster."
            )
        beta_1_map = read_external_geometry_raster(cfg.spatial_beta_1_raster, profile, "Spatial beta_1")
        beta_2_map = read_external_geometry_raster(cfg.spatial_beta_2_raster, profile, "Spatial beta_2")
        alfa_1_map = read_external_geometry_raster(cfg.spatial_alfa_1_raster, profile, "Spatial alfa_1")
        alfa_2_map = read_external_geometry_raster(cfg.spatial_alfa_2_raster, profile, "Spatial alfa_2")
        coeff_valid = (
            np.isfinite(beta_1_map)
            & np.isfinite(beta_2_map)
            & np.isfinite(alfa_1_map)
            & np.isfinite(alfa_2_map)
            & (beta_1_map > 0)
            & (alfa_1_map > 0)
        )
        spatial_coefficients_used = river_mask & coeff_valid

        if cfg.river_geometry_source == "spatial_coefficients":
            B = np.zeros_like(B_power, dtype="float32")
            H = np.zeros_like(H_power, dtype="float32")
        else:
            fallback_used = river_mask & ~coeff_valid & (B_power > 0) & (H_power > 0)

        B[spatial_coefficients_used] = beta_1_map[spatial_coefficients_used] * np.power(
            fac_area[spatial_coefficients_used],
            beta_2_map[spatial_coefficients_used],
        )
        H[spatial_coefficients_used] = alfa_1_map[spatial_coefficients_used] * np.power(
            fac_area[spatial_coefficients_used],
            alfa_2_map[spatial_coefficients_used],
        )

        print(f"[INFO] river geometry source = {cfg.river_geometry_source}")
        print(f"[INFO] spatial coefficient cells used = {int(spatial_coefficients_used.sum()):,}")
        if cfg.river_geometry_source == "spatial_coefficients_or_power_law":
            print(f"[INFO] power-law fallback river cells = {int(fallback_used.sum()):,}")
    else:
        fallback_used = river_mask & (B_power > 0) & (H_power > 0)

    if cfg.river_width_cap_m is not None:
        B = np.minimum(B, cfg.river_width_cap_m)
    if cfg.river_depth_cap_m is not None:
        H = np.minimum(H, cfg.river_depth_cap_m)

    B[~river_mask] = 0.0
    H[~river_mask] = 0.0
    B[~np.isfinite(B)] = 0.0
    H[~np.isfinite(H)] = 0.0

    dem_base, base_profile = read_raster(dem_to_carve)
    base_nodata = base_profile.get("nodata", -9999.0)
    valid_base = valid_mask(dem_base, base_nodata)
    channel_surface = dem_base.copy()
    corridor_lowering = np.zeros_like(dem_base, dtype="float32")
    if external_network_used and cfg.external_network_profile_from_source_dem:
        source_dem, source_profile = read_raster(Path(cfg.dem))
        source_nodata = source_profile.get("nodata", -9999.0)
        source_mask, source_receiver, _ = build_external_d4_river_network(
            cfg.external_river_network,
            source_profile,
            cfg.min_area,
            cfg.external_network_id_field,
            cfg.external_network_next_down_field,
            cfg.external_network_area_field,
        )
        source_profile_surface, source_lowering = enforce_downstream_channel_surface(
            source_dem,
            source_mask,
            source_receiver,
            source_profile,
            cfg.channel_bed_min_slope,
        )
        source_profile_surface[~source_mask] = np.nan
        source_profile_surface[~valid_mask(source_dem, source_nodata)] = np.nan
        coarse_profile = aggregate_channel_surface_minimum(source_profile_surface, source_profile, base_profile)
        profile_available = river_mask & valid_base & np.isfinite(coarse_profile)
        channel_surface[profile_available] = np.minimum(channel_surface[profile_available], coarse_profile[profile_available])
        channel_surface, corridor_lowering = enforce_downstream_channel_surface(
            channel_surface,
            river_mask & valid_base,
            receiver,
            base_profile,
            cfg.channel_bed_min_slope,
        )
        write_raster(
            output_path(out_dir, "D4_external_river_profile_source_surface_m.tif"),
            source_profile_surface,
            source_profile,
            nodata=-9999.0,
        )
        write_raster(
            output_path(out_dir, "D4_external_river_profile_source_lowering_m.tif"),
            source_lowering,
            source_profile,
            nodata=-9999.0,
        )
        write_raster(
            output_path(out_dir, "D4_external_river_corridor_surface_m.tif"),
            channel_surface,
            base_profile,
            nodata=-9999.0,
        )
        write_raster(
            output_path(out_dir, "D4_external_river_corridor_lowering_m.tif"),
            corridor_lowering,
            base_profile,
            nodata=-9999.0,
        )
        values = corridor_lowering[(river_mask & valid_base) & (corridor_lowering > 0)]
        pd.DataFrame([{
            "profile_lowered_cells": int(values.size),
            "profile_lowering_max_m": float(np.nanmax(values)) if values.size else 0.0,
            "profile_lowering_p99_m": float(np.nanpercentile(values, 99)) if values.size else 0.0,
        }]).to_csv(output_path(out_dir, "D4_external_river_profile_QA_summary.csv"), index=False)
    channel_depth_adjustment = np.zeros_like(H, dtype="float32")
    if cfg.condition_d4_channel_bed:
        H, channel_depth_adjustment = condition_d4_channel_bed(
            channel_surface,
            H,
            river_mask & valid_base,
            receiver,
            dem_route,
            base_profile,
            cfg.channel_bed_min_slope,
            cfg.river_depth_cap_m,
            topology_order=external_network_used,
        )
        H[~river_mask] = 0.0
        write_raster(
            output_path(out_dir, "D4_Neal_channel_depth_adjustment_m.tif"),
            channel_depth_adjustment,
            base_profile,
            nodata=-9999.0,
        )

    H_abg = compute_equivalent_H_abg(
        B,
        H,
        grid_width=grid_width,
        mode=cfg.carve_mode,
        max_depth=cfg.max_H_abg_m,
    )
    H_abg[~river_mask] = 0.0

    # Safety checks
    max_H_abg = float(np.nanmax(H_abg)) if np.any(np.isfinite(H_abg)) else 0.0
    if max_H_abg >= cfg.max_H_abg_m:
        warnings.warn(
            f"Some cells reached the maximum carve depth cap ({cfg.max_H_abg_m} m). "
            "Inspect D4_H_abg_m.tif and consider changing hydraulic geometry parameters."
        )
    if max_H_abg > 100.0:
        raise ValueError(
            "Computed H_abg DEM reduction depth exceeds 100 m. "
            "This is probably caused by unrealistic width/depth coefficients or too small a grid resolution."
        )

    channel_bed_qa = write_d4_channel_bed_qa(
        channel_surface,
        H,
        river_mask & valid_base,
        receiver,
        base_profile,
        out_dir,
    )

    carved = channel_surface.copy()
    apply_mask = river_mask & valid_base & (H_abg > 0)
    carved[apply_mask] = carved[apply_mask] - H_abg[apply_mask]

    out_dem = output_path(out_dir, "DEM_reduced_creeks_D4.tif")

    write_raster(output_path(out_dir, "D4_flow_direction.tif"), receiver_to_d4_direction(receiver), profile, nodata=0.0)
    write_raster(output_path(out_dir, "D4_flow_accum_cells.tif"), acc_cells.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_Wshed_Properties_fac_area_km2.tif"), fac_area.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_idx_facc.tif"), river_mask.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_Wshed_Properties_River_Width_m.tif"), B.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_Wshed_Properties_River_Depth_m.tif"), H.astype("float32"), profile, nodata=-9999.0)
    write_raster(output_path(out_dir, "D4_H_abg_m.tif"), H_abg.astype("float32"), profile, nodata=-9999.0)
    if external_network_used:
        write_raster(output_path(out_dir, "D4_external_network_used.tif"), river_mask.astype("float32"), profile, nodata=-9999.0)
    if cfg.river_geometry_source in {"external", "external_or_power_law"}:
        write_raster(
            output_path(out_dir, "D4_external_geometry_available.tif"),
            external_available.astype("float32"),
            profile,
            nodata=-9999.0,
        )
    if cfg.river_geometry_source in {"spatial_coefficients", "spatial_coefficients_or_power_law"}:
        write_raster(
            output_path(out_dir, "D4_spatial_coefficients_used.tif"),
            spatial_coefficients_used.astype("float32"),
            profile,
            nodata=-9999.0,
        )
    write_raster(
        output_path(out_dir, "D4_power_law_fallback_used.tif"),
        fallback_used.astype("float32"),
        profile,
        nodata=-9999.0,
    )
    if cfg.river_geometry_source in {"external", "external_or_power_law"}:
        write_raster(
            output_path(out_dir, "D4_external_geometry_used.tif"),
            external_used.astype("float32"),
            profile,
            nodata=-9999.0,
        )
    write_raster(out_dem, carved.astype("float32"), base_profile, nodata=base_nodata)

    geometry_mask = river_mask & (B > 0) & (H > 0) & (H_abg > 0)
    adjusted = channel_depth_adjustment[geometry_mask & (channel_depth_adjustment > 0)]
    channel_bed_conditioning_summary = {
        "channel_bed_depth_adjusted_cells": int(adjusted.size),
        "channel_bed_depth_adjustment_max_m": float(np.nanmax(adjusted)) if adjusted.size else 0.0,
        "channel_bed_depth_adjustment_p99_m": float(np.nanpercentile(adjusted, 99)) if adjusted.size else 0.0,
    }

    rows = []
    if np.any(geometry_mask):
        rows.append({
            "GIS_data.min_area_km2": cfg.min_area,
            "river_cells": n_river,
            "geometry_source": cfg.river_geometry_source,
            "geometry_cells_with_width_depth": int(geometry_mask.sum()),
            "external_geometry_available_cells": int((river_mask & external_available).sum()),
            "external_geometry_used_cells": int(external_used.sum()),
            "spatial_coefficients_used_cells": int(spatial_coefficients_used.sum()),
            "power_law_geometry_cells": int(fallback_used.sum()),
            "power_law_fallback_cells": int(
                fallback_used.sum()
                if cfg.river_geometry_source in {"external_or_power_law", "spatial_coefficients_or_power_law"}
                else 0
            ),
            "beta_1": cfg.beta_1,
            "beta_2": cfg.beta_2,
            "alfa_1": cfg.alfa_1,
            "alfa_2": cfg.alfa_2,
            "carve_mode": cfg.carve_mode,
            "effective_channel_cell_width_m": grid_width,
            "River_Width_min_m": float(np.nanmin(B[geometry_mask])),
            "River_Width_mean_m": float(np.nanmean(B[geometry_mask])),
            "River_Width_max_m": float(np.nanmax(B[geometry_mask])),
            "River_Depth_min_m": float(np.nanmin(H[geometry_mask])),
            "River_Depth_mean_m": float(np.nanmean(H[geometry_mask])),
            "River_Depth_max_m": float(np.nanmax(H[geometry_mask])),
            "H_abg_min_m": float(np.nanmin(H_abg[geometry_mask])),
            "H_abg_mean_m": float(np.nanmean(H_abg[geometry_mask])),
            "H_abg_max_m": float(np.nanmax(H_abg[geometry_mask])),
            **channel_bed_qa,
            **channel_bed_conditioning_summary,
        })
    else:
        rows.append({
            "GIS_data.min_area_km2": cfg.min_area,
            "river_cells": n_river,
            "geometry_source": cfg.river_geometry_source,
            "geometry_cells_with_width_depth": 0,
            "external_geometry_available_cells": int((river_mask & external_available).sum()),
            "external_geometry_used_cells": int(external_used.sum()),
            "spatial_coefficients_used_cells": int(spatial_coefficients_used.sum()),
            "power_law_geometry_cells": int(fallback_used.sum()),
            "power_law_fallback_cells": int(
                fallback_used.sum()
                if cfg.river_geometry_source in {"external_or_power_law", "spatial_coefficients_or_power_law"}
                else 0
            ),
            "beta_1": cfg.beta_1,
            "beta_2": cfg.beta_2,
            "alfa_1": cfg.alfa_1,
            "alfa_2": cfg.alfa_2,
            "carve_mode": cfg.carve_mode,
            "effective_channel_cell_width_m": grid_width,
            **channel_bed_qa,
            **channel_bed_conditioning_summary,
        })

    summary = pd.DataFrame(rows)
    summary_path = output_path(out_dir, "D4_HydroPol2D_creek_reduction_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[SAVED] {out_dem}")
    print(f"[SAVED] {summary_path}")
    print(summary.to_string(index=False))

    return out_dem


# =============================================================================
# WhiteboxTools utilities
# =============================================================================

def init_wbt(out_dir: Path):
    if whitebox is None:
        raise ImportError("whitebox package is not installed.")

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(out_dir))
    wbt.verbose = True
    return wbt


def wbt_feature_preserving_smoothing(wbt, dem_in: Path, dem_out: Path, filter_cells: int) -> None:
    """
    Whitebox has used both function names in different examples/frontends.
    Try the common names.
    """
    if hasattr(wbt, "feature_preserving_smoothing"):
        wbt.feature_preserving_smoothing(str(dem_in), str(dem_out), filter=filter_cells)
    elif hasattr(wbt, "feature_preserving_denoise"):
        wbt.feature_preserving_denoise(str(dem_in), str(dem_out), filter=filter_cells)
    else:
        raise AttributeError(
            "Could not find feature_preserving_smoothing or feature_preserving_denoise in WhiteboxTools."
        )


def wbt_breach_pipeline(
    wbt,
    cfg: DEMConditioningConfig,
    dem_in: Path,
    out_dir: Path,
    final_name: str = "DEM_hydraulic_conditioned.tif",
) -> Path:
    """
    Breach/fill pipeline.
    We prefer breaching over filling. The least-cost breach is used with fill=False
    to avoid accidentally filling large real depressions or edge catchments.
    """
    print_header("WhiteboxTools hydrologic conditioning")

    pitless = output_path(out_dir, "DEM_01_single_cell_pits_breached.tif")
    breached = output_path(out_dir, "DEM_02_breached.tif")
    final = output_path(out_dir, final_name)

    print("[WBT] Breaching single-cell pits...")
    wbt.breach_single_cell_pits(str(dem_in), str(pitless))
    normalize_existing_raster(pitless)

    if cfg.use_least_cost_breaching:
        print("[WBT] Least-cost breaching depressions...")
        print(f"      dist cells = {cfg.breach_dist_cells}")
        print(f"      max cost   = {cfg.breach_max_cost}")
        print(f"      fill       = {cfg.breach_fill_remaining}")

        # Python signature commonly follows:
        # breach_depressions_least_cost(dem, output, dist, max_cost=None,
        #                              min_dist=True, flat_increment=None, fill=True)
        wbt.breach_depressions_least_cost(
            str(pitless),
            str(breached),
            cfg.breach_dist_cells,
            max_cost=cfg.breach_max_cost,
            min_dist=cfg.breach_min_dist,
            flat_increment=cfg.breach_flat_increment,
            fill=cfg.breach_fill_remaining,
        )
        normalize_existing_raster(breached)

        # A second conventional breach pass catches residual small pits
        # without aggressive global filling.
        print("[WBT] Secondary controlled breach pass...")
        wbt.breach_depressions(
            str(breached),
            str(final),
            max_depth=cfg.breach_max_depth_m,
            max_length=cfg.breach_max_length_cells,
            flat_increment=cfg.breach_flat_increment,
            fill_pits=False,
        )
        normalize_existing_raster(final)

    else:
        print("[WBT] Conventional breaching depressions...")
        wbt.breach_depressions(
            str(pitless),
            str(final),
            max_depth=cfg.breach_max_depth_m,
            max_length=cfg.breach_max_length_cells,
            flat_increment=cfg.breach_flat_increment,
            fill_pits=False,
        )
        normalize_existing_raster(final)

    if cfg.fill_residual_depressions:
        filled = output_path(out_dir, "DEM_hydraulic_conditioned_filled_residual.tif")
        print("[WBT] Optional residual filling. Use this carefully.")
        print(f"      max depth = {cfg.fill_max_depth_m} m")
        wbt.fill_depressions(
            str(final),
            str(filled),
            fix_flats=True,
            flat_increment=cfg.breach_flat_increment,
            max_depth=cfg.fill_max_depth_m,
        )
        normalize_existing_raster(filled)
        final = filled

    return final


def wbt_diagnostics(wbt, final_dem: Path, out_dir: Path) -> None:
    """
    Create standard hydrologic diagnostics.
    Some Whitebox function signatures have changed across frontends; if a
    diagnostic fails, the main DEM is still produced.
    """
    print_header("WhiteboxTools diagnostics")

    diagnostics = [
        ("slope", output_path(out_dir, "slope_after_deg.tif")),
        ("sink", output_path(out_dir, "diagnostic_sinks.tif")),
        ("depth_in_sink", output_path(out_dir, "diagnostic_depth_in_sink.tif")),
        ("find_no_flow_cells", output_path(out_dir, "diagnostic_no_flow_cells.tif")),
        ("d8_pointer", output_path(out_dir, "diagnostic_d8_pointer.tif")),
    ]

    for tool_name, out_path in diagnostics:
        try:
            print(f"[WBT] {tool_name}...")
            if tool_name == "slope":
                wbt.slope(str(final_dem), str(out_path), units="degrees")
            elif tool_name == "sink":
                wbt.sink(str(final_dem), str(out_path))
            elif tool_name == "depth_in_sink":
                wbt.depth_in_sink(str(final_dem), str(out_path))
            elif tool_name == "find_no_flow_cells":
                wbt.find_no_flow_cells(str(final_dem), str(out_path))
            elif tool_name == "d8_pointer":
                wbt.d8_pointer(str(final_dem), str(out_path))
            normalize_existing_raster(out_path)
        except Exception as exc:
            warnings.warn(f"Diagnostic {tool_name} failed: {exc}")

    try:
        print("[WBT] D8 flow accumulation...")
        wbt.d8_flow_accumulation(
            str(final_dem),
            str(output_path(out_dir, "diagnostic_d8_flow_accum_log.tif")),
            out_type="cells",
            log=True,
            clip=False,
        )
        normalize_existing_raster(output_path(out_dir, "diagnostic_d8_flow_accum_log.tif"))
    except Exception as exc:
        warnings.warn(f"D8 flow accumulation failed: {exc}")

    try:
        print("[WBT] D-infinity flow accumulation...")
        wbt.d_inf_flow_accumulation(
            str(final_dem),
            str(output_path(out_dir, "diagnostic_dinf_flow_accum_log.tif")),
            out_type="specific contributing area",
            log=True,
        )
        normalize_existing_raster(output_path(out_dir, "diagnostic_dinf_flow_accum_log.tif"))
    except Exception as exc:
        warnings.warn(f"D-infinity flow accumulation failed: {exc}")


# =============================================================================
# Main pipeline
# =============================================================================

def clean_raw_dem(cfg: DEMConditioningConfig, out_dir: Path) -> Path:
    print_header("Step 1: Raw DEM cleaning")

    raw_path = Path(cfg.dem)
    cleaned_path = output_path(out_dir, "DEM_raw_cleaned.tif")

    with rasterio.open(raw_path) as src:
        profile = src.profile.copy()
        assert_projected_meters(profile, allow_geographic=cfg.allow_geographic_crs)

        data = src.read(1).astype("float32")
        nodata_in = src.nodata
        nodata_out = -9999.0

        valid = valid_mask(data, nodata_in)

        if cfg.fill_nodata and np.any(~valid):
            print("[INFO] Filling small NoData holes using rasterio.fill.fillnodata.")
            print(f"       max_nodata_fill_pixels = {cfg.max_nodata_fill_pixels}")

            # fillnodata expects mask > 0 as valid data.
            filled = fillnodata(
                data,
                mask=valid.astype("uint8"),
                max_search_distance=cfg.max_nodata_fill_pixels,
                smoothing_iterations=0,
            )

            # Keep original valid cells exactly unchanged.
            data = np.where(valid, data, filled).astype("float32")

            # Remaining invalid cells stay NoData.
            data[~np.isfinite(data)] = nodata_out
        else:
            data[~valid] = nodata_out

        out_profile = make_safe_gtiff_profile(
            profile,
            dtype="float32",
            nodata=nodata_out,
            compress=cfg.compression,
        )

        with rasterio.open(cleaned_path, "w", **out_profile) as dst:
            dst.write(data.astype("float32"), 1)

    print(f"[SAVED] {cleaned_path}")
    return cleaned_path


def selective_artifact_smoothing(
    cfg: DEMConditioningConfig,
    wbt,
    dem_in: Path,
    out_dir: Path,
) -> Path:
    print_header("Step 2: Selective extreme-slope artifact smoothing")

    if not cfg.smooth_artifacts:
        print("[SKIP] Artifact smoothing disabled.")
        return dem_in

    arr, profile = read_raster(dem_in)
    nodata = profile.get("nodata", -9999.0)

    slope = compute_slope_deg_numpy(arr, profile, nodata)
    slope_before_path = output_path(out_dir, "slope_before_deg.tif")
    write_raster(slope_before_path, slope, profile, nodata=-9999.0)

    finite_slope = slope[np.isfinite(slope)]
    if finite_slope.size == 0:
        warnings.warn("Slope calculation returned no finite values; skipping smoothing.")
        return dem_in

    percentile_threshold = np.nanpercentile(finite_slope, cfg.slope_percentile)

    if cfg.slope_threshold_deg is not None:
        threshold = max(percentile_threshold, cfg.slope_threshold_deg)
    else:
        threshold = percentile_threshold

    print(f"[INFO] slope percentile threshold p{cfg.slope_percentile} = {percentile_threshold:.3f} deg")
    print(f"[INFO] final smoothing threshold = {threshold:.3f} deg")

    artifact_mask = np.isfinite(slope) & (slope >= threshold)

    # Protect stream corridors from generic smoothing unless explicitly allowed.
    if cfg.streams and (not cfg.allow_stream_smoothing):
        print("[INFO] Protecting stream corridor from artifact smoothing.")
        stream_protect_mask = rasterize_vector_mask(
            cfg.streams,
            profile,
            buffer_m=cfg.protect_stream_buffer_m,
            burn_value=1,
        )
        artifact_mask &= ~stream_protect_mask

    n_artifacts = int(artifact_mask.sum())
    n_valid = int(valid_mask(arr, nodata).sum())
    print(f"[INFO] artifact cells selected = {n_artifacts} ({100*n_artifacts/max(n_valid,1):.4f}% of valid DEM)")

    if n_artifacts == 0:
        print("[SKIP] No artifact cells selected.")
        return dem_in

    smoothed_full = output_path(out_dir, "DEM_tmp_feature_preserving_smoothed_full.tif")
    wbt_feature_preserving_smoothing(wbt, dem_in, smoothed_full, cfg.smooth_filter_cells)

    smoothed_arr, _ = read_raster(smoothed_full)
    selectively_smoothed = selective_replace(arr, smoothed_arr, artifact_mask)

    mask_path = output_path(out_dir, "mask_extreme_slope_artifacts.tif")
    write_raster(mask_path, artifact_mask.astype("float32"), profile, nodata=-9999.0)

    out_path = output_path(out_dir, "DEM_artifact_smoothed.tif")
    write_raster(out_path, selectively_smoothed, profile, nodata=nodata)

    print(f"[SAVED] {out_path}")
    print(f"[SAVED] {mask_path}")
    return out_path


def optional_channel_and_crossing_enforcement(
    cfg: DEMConditioningConfig,
    dem_in: Path,
    out_dir: Path,
) -> Path:
    print_header("Step 3: Optional channel/crossing enforcement")

    arr, profile = read_raster(dem_in)
    nodata = profile.get("nodata", -9999.0)
    modified = arr.copy()
    changed = np.zeros(arr.shape, dtype=bool)

    if cfg.streams and cfg.stream_burn_depth_m and cfg.stream_burn_depth_m > 0:
        print("[INFO] Applying simple stream corridor burn.")
        print(f"       buffer = {cfg.stream_buffer_m} m")
        print(f"       burn depth = {cfg.stream_burn_depth_m} m")
        stream_mask = rasterize_vector_mask(
            cfg.streams,
            profile,
            buffer_m=cfg.stream_buffer_m,
            burn_value=1,
        )
        valid = valid_mask(modified, nodata)
        burn_mask = stream_mask & valid
        modified[burn_mask] = modified[burn_mask] - float(cfg.stream_burn_depth_m)
        changed |= burn_mask

        write_raster(output_path(out_dir, "mask_stream_burn.tif"), burn_mask.astype("float32"), profile, nodata=-9999.0)

    else:
        print("[SKIP] No stream burn applied.")
        print("       Provide --streams and --stream-burn-depth-m > 0 to enable it.")

    if cfg.crossings and cfg.crossing_burn_depth_m and cfg.crossing_burn_depth_m > 0:
        print("[INFO] Applying simple crossing/culvert burn.")
        print(f"       buffer = {cfg.crossing_buffer_m} m")
        print(f"       burn depth = {cfg.crossing_burn_depth_m} m")
        cross_mask = rasterize_vector_mask(
            cfg.crossings,
            profile,
            buffer_m=cfg.crossing_buffer_m,
            burn_value=1,
        )
        valid = valid_mask(modified, nodata)
        burn_mask = cross_mask & valid
        modified[burn_mask] = modified[burn_mask] - float(cfg.crossing_burn_depth_m)
        changed |= burn_mask

        write_raster(output_path(out_dir, "mask_crossing_burn.tif"), burn_mask.astype("float32"), profile, nodata=-9999.0)

    else:
        print("[SKIP] No crossing/culvert burn applied.")
        print("       Provide --crossings and --crossing-burn-depth-m > 0 to enable it.")

    if not changed.any():
        print("[INFO] No channel/crossing modifications were made.")
        return dem_in

    out_path = output_path(out_dir, "DEM_channel_crossing_enforced.tif")
    write_raster(out_path, modified, profile, nodata=nodata)
    print(f"[SAVED] {out_path}")
    return out_path


def write_delta_dem(cleaned_path: Path, final_path: Path, out_dir: Path) -> Path:
    clean, profile = read_raster(cleaned_path)
    final, _ = read_raster(final_path)
    nodata = profile.get("nodata", -9999.0)

    vm = valid_mask(clean, nodata) & np.isfinite(final)
    delta = np.full(clean.shape, np.nan, dtype="float32")
    delta[vm] = final[vm] - clean[vm]

    out_path = output_path(out_dir, "DEM_modification_final_minus_cleaned.tif")
    write_raster(out_path, delta, profile, nodata=-9999.0)
    print(f"[SAVED] {out_path}")
    return out_path


def make_quicklook_plot(cleaned_path: Path, final_path: Path, delta_path: Path, out_dir: Path) -> None:
    if plt is None:
        warnings.warn("matplotlib not available; skipping quicklook plot.")
        return

    print_header("Quicklook plot")

    clean, profile = read_raster(cleaned_path)
    final, _ = read_raster(final_path)
    delta, _ = read_raster(delta_path)
    nodata = profile.get("nodata", -9999.0)

    for arr in (clean, final, delta):
        arr[arr == nodata] = np.nan

    # Downsample for plotting if large.
    max_size = 2000
    factor = max(1, int(np.ceil(max(clean.shape) / max_size)))

    def ds(a):
        if factor == 1:
            return a
        return a[::factor, ::factor]

    clean_ds = ds(clean)
    final_ds = ds(final)
    delta_ds = ds(delta)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    im0 = axes[0].imshow(clean_ds, cmap="terrain")
    axes[0].set_title("Cleaned DEM")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.75, label="m")

    im1 = axes[1].imshow(final_ds, cmap="terrain")
    axes[1].set_title("Conditioned DEM")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.75, label="m")

    # Robust delta scale
    finite = delta_ds[np.isfinite(delta_ds)]
    if finite.size:
        vmax = np.nanpercentile(np.abs(finite), 99)
        vmax = max(vmax, 0.01)
    else:
        vmax = 1.0

    im2 = axes[2].imshow(delta_ds, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Final - cleaned DEM")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], shrink=0.75, label="m")

    out_png = output_path(out_dir, "quicklook_dem_conditioning.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def _plot_ready_array(path: Path, max_size: int = 1600) -> Tuple[np.ndarray, dict]:
    arr, profile = read_raster(path)
    nodata = profile.get("nodata", -9999.0)
    arr = arr.astype("float32", copy=False)
    arr[arr == nodata] = np.nan

    factor = max(1, int(np.ceil(max(arr.shape) / max_size)))
    if factor > 1:
        arr = arr[::factor, ::factor]
    return arr, profile


def _robust_limits(arr: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> Tuple[Optional[float], Optional[float]]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    return float(np.nanpercentile(finite, lower)), float(np.nanpercentile(finite, upper))


def _imshow_with_colorbar(ax, arr: np.ndarray, title: str, cmap: str, label: str = "", robust: bool = True) -> None:
    vmin = vmax = None
    if robust:
        vmin, vmax = _robust_limits(arr)
        if vmin == vmax:
            vmin = vmax = None
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.78, label=label)


def make_pipeline_stage_diagnostics(out_dir: Path) -> None:
    """Create map panels showing the DEM at each major conditioning stage."""
    if plt is None:
        warnings.warn("matplotlib not available; skipping pipeline stage diagnostics.")
        return

    paths = [
        ("Resampled 1000 m", output_path(out_dir, "DEM_resampled_1000m.tif")),
        ("Cleaned", output_path(out_dir, "DEM_raw_cleaned.tif")),
        ("Selective smoothing", output_path(out_dir, "DEM_artifact_smoothed.tif")),
        ("Hydrologic conditioning", output_path(out_dir, "DEM_hydrologically_conditioned_pre_bathymetry.tif")),
        ("D4 bathymetry applied", output_path(out_dir, "DEM_reduced_creeks_D4.tif")),
        ("Final conditioned", output_path(out_dir, "DEM_hydraulic_conditioned.tif")),
    ]
    available = [(title, path) for title, path in paths if path.exists()]
    if not available:
        return

    print_header("Diagnostic plots: pipeline stages")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, (title, path) in zip(axes_flat, available):
        arr, _ = _plot_ready_array(path)
        _imshow_with_colorbar(ax, arr, title, "terrain", "m")

    for ax in axes_flat[len(available):]:
        ax.axis("off")

    out_png = output_path(out_dir, "diagnostic_pipeline_stages.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_smoothing_diagnostics(out_dir: Path) -> None:
    """Create maps showing the selective smoothing mask and elevation change."""
    if plt is None:
        warnings.warn("matplotlib not available; skipping smoothing diagnostics.")
        return

    required = [
        output_path(out_dir, "slope_before_deg.tif"),
        output_path(out_dir, "mask_extreme_slope_artifacts.tif"),
        output_path(out_dir, "DEM_raw_cleaned.tif"),
        output_path(out_dir, "DEM_artifact_smoothed.tif"),
    ]
    if not all(path.exists() for path in required):
        return

    print_header("Diagnostic plots: slope artifact smoothing")
    slope, _ = _plot_ready_array(output_path(out_dir, "slope_before_deg.tif"))
    mask, _ = _plot_ready_array(output_path(out_dir, "mask_extreme_slope_artifacts.tif"))
    clean, _ = _plot_ready_array(output_path(out_dir, "DEM_raw_cleaned.tif"))
    smooth, _ = _plot_ready_array(output_path(out_dir, "DEM_artifact_smoothed.tif"))
    smoothing_delta = smooth - clean
    changed = np.isfinite(smoothing_delta) & (np.abs(smoothing_delta) > 1e-6)
    smoothing_delta_changed = np.where(changed, smoothing_delta, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    _imshow_with_colorbar(axes[0], slope, "Slope before smoothing", "magma", "degrees")
    _imshow_with_colorbar(axes[1], mask, "Extreme slope artifact mask", "gray", "mask", robust=False)

    changed_values = smoothing_delta_changed[np.isfinite(smoothing_delta_changed)]
    if changed_values.size:
        vmax = max(float(np.nanpercentile(np.abs(changed_values), 99)), 0.01)
        changed_pct = 100.0 * changed_values.size / max(int(np.isfinite(smoothing_delta).sum()), 1)
        min_change = float(np.nanmin(changed_values))
        max_change = float(np.nanmax(changed_values))
    else:
        vmax = 1.0
        changed_pct = 0.0
        min_change = 0.0
        max_change = 0.0

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(color="white", alpha=0.0)
    im = axes[2].imshow(smoothing_delta_changed, cmap=cmap, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Smoothing change, changed cells only")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], shrink=0.78, label="m")
    axes[2].text(
        0.02,
        0.02,
        f"changed: {changed_values.size:,} cells ({changed_pct:.3f}%)\n"
        f"range: {min_change:.3f} to {max_change:.3f} m\n"
        f"color scale: +/- p99 abs = {vmax:.3f} m",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 4},
    )

    out_png = output_path(out_dir, "diagnostic_smoothing_artifacts.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_d4_river_diagnostics(out_dir: Path) -> None:
    """Create D4 flow accumulation, river mask, and hydraulic-geometry plots."""
    if plt is None:
        warnings.warn("matplotlib not available; skipping D4 diagnostics.")
        return

    fac_path = output_path(out_dir, "D4_Wshed_Properties_fac_area_km2.tif")
    mask_path = output_path(out_dir, "D4_idx_facc.tif")
    width_path = output_path(out_dir, "D4_Wshed_Properties_River_Width_m.tif")
    depth_path = output_path(out_dir, "D4_Wshed_Properties_River_Depth_m.tif")
    habg_path = output_path(out_dir, "D4_H_abg_m.tif")
    if not all(path.exists() for path in [fac_path, mask_path, width_path, depth_path, habg_path]):
        return

    print_header("Diagnostic plots: D4 river extraction")
    fac, _ = _plot_ready_array(fac_path)
    river_mask, _ = _plot_ready_array(mask_path)
    width, _ = _plot_ready_array(width_path)
    depth, _ = _plot_ready_array(depth_path)
    habg, _ = _plot_ready_array(habg_path)

    fac_log = np.full_like(fac, np.nan, dtype="float32")
    positive_fac = np.isfinite(fac) & (fac > 0)
    fac_log[positive_fac] = np.log10(fac[positive_fac])
    width_masked = np.where((river_mask > 0) & np.isfinite(width) & (width > 0), width, np.nan)
    habg_masked = np.where(river_mask > 0, habg, np.nan)

    geometry_source = "power_law"
    summary_path = output_path(out_dir, "D4_HydroPol2D_creek_reduction_summary.csv")
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if "geometry_source" in summary.columns and len(summary):
                geometry_source = str(summary.loc[0, "geometry_source"])
        except Exception:
            geometry_source = "unknown"

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    _imshow_with_colorbar(axes[0, 0], fac_log, "D4 upstream area", "viridis", "log10(km2)")
    _imshow_with_colorbar(axes[0, 1], river_mask, "idx_facc river mask", "gray", "mask", robust=False)
    _imshow_with_colorbar(axes[1, 0], width_masked, f"River_Width ({geometry_source})", "plasma", "m")
    _imshow_with_colorbar(axes[1, 1], habg_masked, "H_abg lowering", "inferno", "m")

    out_png = output_path(out_dir, "diagnostic_d4_river_extraction.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    finite_fac = fac[np.isfinite(fac)]
    valid_cell_count = max(int(finite_fac.size), 1)
    thresholds = [100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    cmap_mask = plt.get_cmap("gray").copy()
    for ax, threshold in zip(axes.ravel(), thresholds):
        threshold_mask = np.where(np.isfinite(fac) & (fac >= threshold), 1.0, np.nan)
        n_cells = int(np.nansum(threshold_mask == 1.0))
        pct = 100.0 * n_cells / valid_cell_count
        ax.imshow(threshold_mask, cmap=cmap_mask, vmin=0, vmax=1)
        ax.set_title(f"min_area >= {threshold:g} km2\n{n_cells:,} cells ({pct:.4f}%)")
        ax.axis("off")

    out_png = output_path(out_dir, "diagnostic_d4_threshold_sweep.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    min_area = 100.0
    mask_100 = np.isfinite(fac) & (fac >= min_area)
    mask_overlay = np.where(mask_100, 1.0, np.nan)

    fac_linear = np.where(np.isfinite(fac), fac, np.nan)
    linear_vmin, linear_vmax = _robust_limits(fac_linear, lower=0.0, upper=99.9)
    if linear_vmax is None or linear_vmax <= 0:
        linear_vmax = float(np.nanmax(fac_linear)) if np.any(np.isfinite(fac_linear)) else 1.0

    overlay_cmap = plt.get_cmap("autumn").copy()
    overlay_cmap.set_bad(color="white", alpha=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    im0 = axes[0].imshow(fac_log, cmap="viridis")
    axes[0].imshow(mask_overlay, cmap=overlay_cmap, vmin=0, vmax=1, alpha=0.85)
    axes[0].set_title("River mask over log10 D4 upstream area\nmin_area = 100 km2")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], shrink=0.78, label="log10(km2)")

    im1 = axes[1].imshow(fac_linear, cmap="viridis", vmin=0, vmax=linear_vmax)
    axes[1].imshow(mask_overlay, cmap=overlay_cmap, vmin=0, vmax=1, alpha=0.85)
    axes[1].set_title("River mask over linear D4 upstream area\nmin_area = 100 km2")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], shrink=0.78, label="km2")

    out_png = output_path(out_dir, "diagnostic_d4_mask_over_accumulation.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")

    fac_full, fac_full_profile = read_raster(fac_path)
    fac_full_nodata = fac_full_profile.get("nodata", -9999.0)
    fac_full_valid = valid_mask(fac_full, fac_full_nodata)
    mask_100_full = fac_full_valid & np.isfinite(fac_full) & (fac_full >= min_area)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labels, n_components = ndimage.label(mask_100_full, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size:
        sizes_sorted = np.sort(sizes)[::-1]
        largest = int(sizes_sorted[0])
        top10 = int(sizes_sorted[:10].sum())
    else:
        largest = 0
        top10 = 0

    component_summary = pd.DataFrame([{
        "min_area_km2": min_area,
        "river_cells": int(mask_100_full.sum()),
        "d4_connected_components": int(n_components),
        "largest_component_cells": largest,
        "largest_component_percent_of_river_cells": 100.0 * largest / max(int(mask_100_full.sum()), 1),
        "top10_components_cells": top10,
        "top10_components_percent_of_river_cells": 100.0 * top10 / max(int(mask_100_full.sum()), 1),
    }])
    summary_path = output_path(out_dir, "D4_river_connectivity_summary.csv")
    component_summary.to_csv(summary_path, index=False)
    print(f"[SAVED] {summary_path}")
    print(component_summary.to_string(index=False))

    fallback_path = output_path(out_dir, "D4_power_law_fallback_used.tif")
    spatial_path = output_path(out_dir, "D4_spatial_coefficients_used.tif")
    if fallback_path.exists():
        fallback, _ = _plot_ready_array(fallback_path)
        spatial = None
        if spatial_path.exists():
            spatial, _ = _plot_ready_array(spatial_path)
        fig, axes = plt.subplots(1, 2 if spatial is not None else 1, figsize=(14, 6), constrained_layout=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes[0].imshow(np.where(fallback > 0, 1.0, np.nan), cmap="Reds", vmin=0, vmax=1)
        axes[0].set_title("Power-law fallback cells")
        axes[0].axis("off")
        if spatial is not None and len(axes) > 1:
            axes[1].imshow(np.where(spatial > 0, 1.0, np.nan), cmap="Greens", vmin=0, vmax=1)
            axes[1].set_title("Spatial coefficient cells")
            axes[1].axis("off")
        out_png = output_path(out_dir, "diagnostic_d4_geometry_source_map.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"[SAVED] {out_png}")

    river = river_mask > 0
    fac_v = fac[river & np.isfinite(fac) & (fac > 0)]
    width_v = width[river & np.isfinite(width) & (width > 0)]
    depth_v = depth[river & np.isfinite(depth) & (depth > 0)]
    habg_v = habg[river & np.isfinite(habg) & (habg > 0)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    datasets = [
        (fac_v, "Upstream area in river cells", "km2", True),
        (width_v, "River_Width in river cells", "m", False),
        (depth_v, "River_Depth in river cells", "m", False),
        (habg_v, "H_abg in river cells", "m", False),
    ]
    for ax, (values, title, xlabel, logx) in zip(axes.ravel(), datasets):
        if values.size:
            ax.hist(values, bins=60, color="#3366aa", alpha=0.85)
            if logx:
                ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cell count")
        ax.grid(True, alpha=0.25)

    out_png = output_path(out_dir, "diagnostic_hydraulic_geometry_histograms.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_final_modification_diagnostics(delta_path: Path, out_dir: Path) -> None:
    """Create final modification map and histogram diagnostics."""
    if plt is None:
        warnings.warn("matplotlib not available; skipping final modification diagnostics.")
        return
    if not Path(delta_path).exists():
        return

    print_header("Diagnostic plots: final modifications")
    delta, _ = _plot_ready_array(delta_path)
    finite = delta[np.isfinite(delta)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    vmax = max(float(np.nanpercentile(np.abs(finite), 99)), 0.01) if finite.size else 1.0
    im = axes[0].imshow(delta, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Final minus cleaned DEM")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], shrink=0.8, label="m")

    if finite.size:
        clipped = finite[np.abs(finite) <= vmax]
        axes[1].hist(clipped, bins=80, color="#555555", alpha=0.85)
        axes[1].axvline(0, color="black", linewidth=1.0)
    axes[1].set_title("Modification histogram, clipped to p99 abs")
    axes[1].set_xlabel("final - cleaned DEM (m)")
    axes[1].set_ylabel("cell count")
    axes[1].grid(True, alpha=0.25)

    out_png = output_path(out_dir, "diagnostic_final_modifications.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


def make_diagnostic_plot_pack(delta_path: Path, out_dir: Path) -> None:
    make_pipeline_stage_diagnostics(out_dir)
    make_smoothing_diagnostics(out_dir)
    make_d4_river_diagnostics(out_dir)
    make_final_modification_diagnostics(delta_path, out_dir)


def resolve_output_dir(cfg: DEMConditioningConfig) -> Path:
    """
    Resolve output directory.

    If the user does not provide --out-dir, all outputs go into an Outputs
    folder next to the DEM, e.g. `Outputs`.
    """
    if cfg.out_dir:
        return Path(cfg.out_dir).expanduser().resolve()
    dem_path = Path(cfg.dem).expanduser().resolve()
    return dem_path.parent / "Outputs"


def write_documentation_file(out_dir: Path) -> Path:
    """Write the full Markdown documentation into the run output directory."""
    doc_path = output_path(out_dir, "DEM_conditioning_README.md")
    doc_path.write_text(CURRENT_DOCUMENTATION, encoding="utf-8")
    print(f"[SAVED] {doc_path}")
    return doc_path


def remove_obsolete_outputs(out_dir: Path) -> None:
    """Remove outputs from older pipeline order so diagnostics do not mix runs."""
    obsolete = [
        output_path(out_dir, "DEM_preD4_single_cell_pits_breached.tif"),
        output_path(out_dir, "DEM_prebreached_for_D4_routing.tif"),
    ]
    for path in obsolete:
        if path.exists():
            path.unlink()
            print(f"[REMOVED obsolete output] {path}")


def print_run_summary(cfg: DEMConditioningConfig, out_dir: Path) -> None:
    """Print a compact summary of the active functionality."""
    print_header("Active DEM conditioning configuration")
    if cfg.original_dem:
        print(f"Original input DEM     : {cfg.original_dem}")
        print(f"Working DEM            : {cfg.dem}")
    else:
        print(f"Input DEM              : {cfg.dem}")
    print(f"Output directory       : {out_dir}")
    print(f"Resample DEM first     : {cfg.resample_dem}")
    if cfg.resample_dem:
        print(f"  target_resolution_m  : {cfg.target_resolution_m}")
        print(f"  resampling_method    : {cfg.resampling_method}")
    print(f"Fill small NoData holes: {cfg.fill_nodata}")
    print(f"Smooth slope artifacts : {cfg.smooth_artifacts}")
    print(f"Auto rivers D4         : {cfg.auto_rivers_d4}")
    if cfg.auto_rivers_d4:
        print(f"  GIS_data.min_area    : {cfg.min_area} km2")
        print(f"  geometry_source      : {cfg.river_geometry_source}")
        if cfg.river_geometry_source in {"external", "external_or_power_law"}:
            print(f"  external_width       : {cfg.external_river_width_raster}")
            print(f"  external_depth       : {cfg.external_river_depth_raster}")
        if cfg.river_geometry_source.startswith("spatial_coefficients"):
            print(f"  spatial_beta_1      : {cfg.spatial_beta_1_raster}")
            print(f"  spatial_beta_2      : {cfg.spatial_beta_2_raster}")
            print(f"  spatial_alfa_1      : {cfg.spatial_alfa_1_raster}")
            print(f"  spatial_alfa_2      : {cfg.spatial_alfa_2_raster}")
        print(f"  GIS_data.beta_1      : {cfg.beta_1}")
        print(f"  GIS_data.beta_2      : {cfg.beta_2}")
        print(f"  GIS_data.alfa_1      : {cfg.alfa_1}")
        print(f"  GIS_data.alfa_2      : {cfg.alfa_2}")
        print(f"  carve_mode           : {cfg.carve_mode}")
        if cfg.external_river_network:
            print(f"  external_network     : {cfg.external_river_network}")
    print(f"Vector stream burn     : {bool(cfg.streams and cfg.stream_burn_depth_m > 0)}")
    print(f"Crossing/culvert burn  : {bool(cfg.crossings and cfg.crossing_burn_depth_m > 0)}")
    print(f"Least-cost breaching   : {cfg.use_least_cost_breaching}")
    print(f"Residual filling       : {cfg.fill_residual_depressions}")


def run_pipeline(cfg: DEMConditioningConfig) -> None:
    check_dependencies()

    out_dir = resolve_output_dir(cfg)
    ensure_dir(out_dir)
    ensure_output_layout(out_dir)
    cfg.out_dir = str(out_dir)
    remove_obsolete_outputs(out_dir)

    write_documentation_file(out_dir)

    # Step 0 must happen before all other DEM analysis so that slope smoothing,
    # D4 flow accumulation, HydroPol2D creek reduction, and depression breaching
    # are performed at the same resolution used by the flood model.
    cfg.original_dem = str(Path(cfg.dem).expanduser().resolve())
    working_dem = resample_dem_if_requested(cfg, out_dir)
    cfg.dem = str(working_dem)

    print_run_summary(cfg, out_dir)

    config_path = output_path(out_dir, "conditioning_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"[SAVED] {config_path}")

    wbt = init_wbt(out_dir)

    cleaned = clean_raw_dem(cfg, out_dir)
    smoothed = selective_artifact_smoothing(cfg, wbt, cleaned, out_dir)

    # Important HydroPol2D sequencing:
    # 1) condition the terrain so D4 routing has a connected surface,
    # 2) derive the D4 stream network from that conditioned surface,
    # 3) apply H_abg bathymetry/channel lowering after network extraction.
    hydrologic_conditioned = wbt_breach_pipeline(
        wbt,
        cfg,
        smoothed,
        out_dir,
        final_name="DEM_hydrologically_conditioned_pre_bathymetry.tif",
    )

    terrain_before_vector_enforcement = hydrologic_conditioned
    if cfg.auto_rivers_d4:
        if cfg.external_river_network:
            d4_routing_surface = hydrologic_conditioned
        else:
            d4_routing_surface = build_d4_monotonic_routing_surface(
                hydrologic_conditioned,
                out_dir,
                cfg.breach_flat_increment,
                cfg.d4_routing_max_raise_m,
            )
        terrain_before_vector_enforcement = automatic_d4_hydraulic_channel_carving(
            cfg,
            dem_to_carve=hydrologic_conditioned,
            routing_dem=d4_routing_surface,
            out_dir=out_dir,
        )

    enforced = optional_channel_and_crossing_enforcement(cfg, terrain_before_vector_enforcement, out_dir)
    final = Path(enforced)

    # Copy final bathymetry-applied terrain to the canonical output name.
    canonical_final = output_path(out_dir, "DEM_hydraulic_conditioned.tif")
    if Path(final) != canonical_final:
        shutil.copy2(final, canonical_final)
        normalize_existing_raster(canonical_final)
        final = canonical_final

    delta_path = write_delta_dem(cleaned, Path(final), out_dir)

    # Summary
    summary_csv = output_path(out_dir, "modification_summary.csv")
    df_summary = summarize_delta(cleaned, Path(final), summary_csv)
    print_header("Modification summary")
    print(df_summary.to_string(index=False))
    print(f"[SAVED] {summary_csv}")

    # Diagnostics
    wbt_diagnostics(wbt, Path(final), out_dir)

    # Quicklook
    if cfg.make_plots:
        make_quicklook_plot(cleaned, Path(final), delta_path, out_dir)
        make_diagnostic_plot_pack(delta_path, out_dir)

    qa_df = build_qa_scorecard(out_dir)
    print_header("QA scorecard")
    print(qa_df.to_string(index=False))
    manifest_path = write_run_manifest(out_dir, cfg, config_path=Path(cfg.config_path) if cfg.config_path else None)
    print(f"[SAVED] {manifest_path}")

    print_header("Done")
    print(f"Final conditioned DEM: {final}")
    print("Inspect the delta DEM, slope rasters, flow accumulation, no-flow cells, and sinks before using this in a flood model.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> DEMConditioningConfig:
    p = argparse.ArgumentParser(
        description="HydroPol2D-oriented DEM conditioning for flood modeling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples
--------
Minimal default run:
  hydrobathydem-condition

Automatic D4 HydroPol2D creek reduction after resampling to 1000 m:
  hydrobathydem-condition \
    --resample-dem --target-resolution-m 1000 --resampling-method average \
    --auto-rivers-d4 --min-area 10000 \
    --beta-1 2.2695 --beta-2 0.4942 \
    --alfa-1 0.1097 --alfa-2 0.3856

Automatic D4 rivers with Lin-calibrated spatial coefficient maps:
  hydrobathydem-prepare-lin2020 --download
  hydrobathydem-calibrate-hydraulics --selected-threshold-km2 5000
  hydrobathydem-condition \
    --resample-dem --target-resolution-m 1000 --auto-rivers-d4 --min-area 100 \
    --river-geometry-source spatial_coefficients_or_power_law \
    --spatial-beta-1-raster Data/Lin2020_bankfull_width/calibration/D4_beta_1_width_5000km2.tif \
    --spatial-beta-2-raster Data/Lin2020_bankfull_width/calibration/D4_beta_2_width_5000km2.tif \
    --spatial-alfa-1-raster Data/Lin2020_bankfull_width/calibration/D4_alfa_1_depth_5000km2.tif \
    --spatial-alfa-2-raster Data/Lin2020_bankfull_width/calibration/D4_alfa_2_depth_5000km2.tif

Print full documentation:
  hydrobathydem-condition --documentation
"""
    )

    p.add_argument("--documentation", "--print-documentation", action="store_true",
                   help="Print the full documentation/manual and exit.")
    p.add_argument("--config", type=Path, default=None,
                   help="Optional JSON/TOML config file. Explicit CLI flags override config values.")
    p.add_argument("--dem", default=DEFAULT_DEM_PATH,
                   help=f"Input DEM GeoTIFF. Default: {DEFAULT_DEM_PATH}")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: an Outputs folder next to the DEM.")

    p.add_argument("--resample-dem", action="store_true",
                   help="Resample the input DEM to the target model resolution before all DEM conditioning steps.")
    p.add_argument("--target-resolution-m", type=float, default=None,
                   help="Target DEM/model resolution in meters used when --resample-dem is active.")
    p.add_argument("--resampling-method", default="bilinear",
                   help="Rasterio resampling method for DEM resampling. Typical DEM choices: average, bilinear, cubic.")

    p.add_argument("--no-fill-nodata", action="store_true", help="Disable small NoData-hole interpolation.")
    p.add_argument("--max-nodata-fill-pixels", type=int, default=10)

    p.add_argument("--no-smooth-artifacts", action="store_true", help="Disable selective extreme-slope smoothing.")
    p.add_argument("--smooth-filter-cells", type=int, default=5)
    p.add_argument("--slope-percentile", type=float, default=99.9)
    p.add_argument("--slope-threshold-deg", type=float, default=None)
    p.add_argument("--protect-stream-buffer-m", type=float, default=2000.0)
    p.add_argument("--allow-stream-smoothing", action="store_true")

    p.add_argument("--streams", default=None, help="Optional stream/river vector file.")
    p.add_argument("--stream-buffer-m", type=float, default=15.0)
    p.add_argument("--stream-burn-depth-m", type=float, default=0.0)

    p.add_argument("--crossings", default=None, help="Optional bridge/culvert/crossing vector file.")
    p.add_argument("--crossing-buffer-m", type=float, default=10.0)
    p.add_argument("--crossing-burn-depth-m", type=float, default=0.0)

    p.add_argument("--auto-rivers-d4", action="store_true",
                   help="Automatically detect idx_facc using D4 flow accumulation and reduce DEM elevations in creeks.")
    p.add_argument(
        "--river-geometry-source",
        choices=[
            "power_law",
            "external",
            "external_or_power_law",
            "spatial_coefficients",
            "spatial_coefficients_or_power_law",
        ],
        default="power_law",
        help="Source for D4 River_Width/River_Depth. Use spatial_coefficients for subcatchment-calibrated coefficient rasters.",
    )
    p.add_argument("--external-river-width-raster", default=None,
                   help="External bankfull river-width raster aligned to, or resampleable onto, the DEM grid.")
    p.add_argument("--external-river-depth-raster", default=None,
                   help="External river-depth raster aligned to, or resampleable onto, the DEM grid.")
    p.add_argument("--external-river-network", default=None,
                   help="Directed river reaches (for example HydroRIVERS) used for D4 topology instead of DEM-derived routing.")
    p.add_argument("--external-network-id-field", default="HYRIV_ID",
                   help="Unique reach-ID field in --external-river-network.")
    p.add_argument("--external-network-next-down-field", default="NEXT_DOWN",
                   help="Downstream reach-ID field in --external-river-network.")
    p.add_argument("--external-network-area-field", default="UPLAND_SKM",
                   help="Upstream drainage-area field [km2] in --external-river-network.")
    p.add_argument("--external-network-snap-radius-cells", type=int, default=0,
                   help="Optional local DEM snap radius for external-network centreline cells (default: 0).")
    p.add_argument("--external-network-profile-from-source-dem", action="store_true",
                   help="Build a descending river-corridor surface from the input DEM before aggregation to the model grid.")
    p.add_argument("--external-geometry-min-width-m", type=float, default=1.0,
                   help="Minimum valid external width in meters.")
    p.add_argument("--external-geometry-min-depth-m", type=float, default=0.01,
                   help="Minimum valid external depth in meters.")
    p.add_argument("--spatial-beta-1-raster", default=None,
                   help="Raster of local beta_1 width coefficients for River_Width = beta_1 * area^beta_2.")
    p.add_argument("--spatial-beta-2-raster", default=None,
                   help="Raster of local beta_2 width exponents for River_Width = beta_1 * area^beta_2.")
    p.add_argument("--spatial-alfa-1-raster", default=None,
                   help="Raster of local alfa_1 depth coefficients for River_Depth = alfa_1 * area^alfa_2.")
    p.add_argument("--spatial-alfa-2-raster", default=None,
                   help="Raster of local alfa_2 depth exponents for River_Depth = alfa_1 * area^alfa_2.")
    p.add_argument("--min-area", "--min-area-km2", dest="min_area", type=float, default=100.0,
                   help="HydroPol2D GIS_data.min_area: minimum D4 upstream fac_area [km2] used to define idx_facc.")
    p.add_argument("--beta-1", "--width-a", dest="beta_1", type=float, default=2.2695,
                   help="HydroPol2D GIS_data.beta_1: River_Width = beta_1 * fac_area^beta_2 [m].")
    p.add_argument("--beta-2", "--width-b", dest="beta_2", type=float, default=0.4942,
                   help="HydroPol2D GIS_data.beta_2: exponent for River_Width = beta_1 * fac_area^beta_2.")
    p.add_argument("--alfa-1", "--depth-a", dest="alfa_1", type=float, default=0.1097,
                   help="HydroPol2D GIS_data.alfa_1: River_Depth = alfa_1 * fac_area^alfa_2 [m].")
    p.add_argument("--alfa-2", "--depth-b", dest="alfa_2", type=float, default=0.3856,
                   help="HydroPol2D GIS_data.alfa_2: exponent for River_Depth = alfa_1 * fac_area^alfa_2.")
    p.add_argument("--carve-mode", choices=["wide", "manning_exact"], default="wide",
                   help="Equivalent channel carving method. 'wide' matches the MATLAB formula.")
    p.add_argument("--channel-cell-width-m", type=float, default=None,
                   help="Effective width of one carved channel cell. Default is sqrt(dx*dy).")
    p.add_argument("--river-width-cap-m", type=float, default=None,
                   help="Optional cap on estimated river width in meters.")
    p.add_argument("--river-depth-cap-m", type=float, default=None,
                   help="Optional cap on estimated river depth in meters.")
    p.add_argument("--max-H-abg-m", "--max-carve-depth-m", dest="max_H_abg_m", type=float, default=50.0,
                   help="Safety cap for HydroPol2D H_abg, the DEM reduction depth in creeks [m].")
    p.add_argument("--d4-routing-max-raise-m", type=float, default=None,
                   help="Optional maximum artificial elevation raise used only for D4 network extraction.")
    p.add_argument("--no-condition-d4-channel-bed", action="store_true",
                   help="Disable downstream subgrid-bed conditioning after D4 geometry is assigned.")
    p.add_argument("--channel-bed-min-slope", type=float, default=1e-5,
                   help="Minimum downstream Neal channel-bed slope imposed along D4 links.")

    p.add_argument("--no-least-cost-breaching", action="store_true")
    p.add_argument("--breach-dist-cells", type=int, default=10000)
    p.add_argument("--breach-max-cost", type=float, default=None)
    p.add_argument("--no-breach-min-dist", action="store_true")
    p.add_argument("--breach-flat-increment", type=float, default=0.01)
    p.add_argument("--breach-fill-remaining", action="store_true",
                   help="Allow least-cost breaching tool to fill remaining depressions. Use carefully.")
    p.add_argument("--breach-max-depth-m", type=float, default=None)
    p.add_argument("--breach-max-length-cells", type=int, default=None)

    p.add_argument("--fill-residual-depressions", action="store_true",
                   help="Optional limited residual depression filling after breaching.")
    p.add_argument("--fill-max-depth-m", type=float, default=0.25)

    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--allow-geographic-crs", action="store_true",
                   help="Allow DEM in geographic CRS. Not recommended for production.")

    a = p.parse_args()

    if a.documentation:
        print(CURRENT_DOCUMENTATION)
        raise SystemExit(0)

    field_names = set(DEMConditioningConfig.__dataclass_fields__)
    cfg_values = {}
    for key, value in load_config_file(a.config).items():
        if key in field_names:
            cfg_values[key] = value
        else:
            warnings.warn(f"Ignoring unknown config key: {key}")

    if a.config:
        cfg_values["config_path"] = str(Path(a.config).expanduser().resolve())

    flags = explicit_cli_flags()

    def has_flag(*names: str) -> bool:
        return any(name in flags for name in names)

    def put(name: str, value, *flag_names: str) -> None:
        if has_flag(*flag_names) or name not in cfg_values:
            cfg_values[name] = value

    put("dem", a.dem, "--dem")
    put("out_dir", a.out_dir, "--out-dir")
    put("resample_dem", a.resample_dem, "--resample-dem")
    put("target_resolution_m", a.target_resolution_m, "--target-resolution-m")
    put("resampling_method", a.resampling_method, "--resampling-method")
    put("fill_nodata", not a.no_fill_nodata, "--no-fill-nodata")
    put("max_nodata_fill_pixels", a.max_nodata_fill_pixels, "--max-nodata-fill-pixels")
    put("smooth_artifacts", not a.no_smooth_artifacts, "--no-smooth-artifacts")
    put("smooth_filter_cells", a.smooth_filter_cells, "--smooth-filter-cells")
    put("slope_percentile", a.slope_percentile, "--slope-percentile")
    put("slope_threshold_deg", a.slope_threshold_deg, "--slope-threshold-deg")
    put("protect_stream_buffer_m", a.protect_stream_buffer_m, "--protect-stream-buffer-m")
    put("allow_stream_smoothing", a.allow_stream_smoothing, "--allow-stream-smoothing")
    put("streams", a.streams, "--streams")
    put("stream_buffer_m", a.stream_buffer_m, "--stream-buffer-m")
    put("stream_burn_depth_m", a.stream_burn_depth_m, "--stream-burn-depth-m")
    put("crossings", a.crossings, "--crossings")
    put("crossing_buffer_m", a.crossing_buffer_m, "--crossing-buffer-m")
    put("crossing_burn_depth_m", a.crossing_burn_depth_m, "--crossing-burn-depth-m")
    put("auto_rivers_d4", a.auto_rivers_d4, "--auto-rivers-d4")
    put("river_geometry_source", a.river_geometry_source, "--river-geometry-source")
    put("external_river_width_raster", a.external_river_width_raster, "--external-river-width-raster")
    put("external_river_depth_raster", a.external_river_depth_raster, "--external-river-depth-raster")
    put("external_river_network", a.external_river_network, "--external-river-network")
    put("external_network_id_field", a.external_network_id_field, "--external-network-id-field")
    put("external_network_next_down_field", a.external_network_next_down_field, "--external-network-next-down-field")
    put("external_network_area_field", a.external_network_area_field, "--external-network-area-field")
    put("external_network_snap_radius_cells", a.external_network_snap_radius_cells, "--external-network-snap-radius-cells")
    put("external_network_profile_from_source_dem", a.external_network_profile_from_source_dem,
        "--external-network-profile-from-source-dem")
    put("external_geometry_min_width_m", a.external_geometry_min_width_m, "--external-geometry-min-width-m")
    put("external_geometry_min_depth_m", a.external_geometry_min_depth_m, "--external-geometry-min-depth-m")
    put("spatial_beta_1_raster", a.spatial_beta_1_raster, "--spatial-beta-1-raster")
    put("spatial_beta_2_raster", a.spatial_beta_2_raster, "--spatial-beta-2-raster")
    put("spatial_alfa_1_raster", a.spatial_alfa_1_raster, "--spatial-alfa-1-raster")
    put("spatial_alfa_2_raster", a.spatial_alfa_2_raster, "--spatial-alfa-2-raster")
    put("min_area", a.min_area, "--min-area", "--min-area-km2")
    put("beta_1", a.beta_1, "--beta-1", "--width-a")
    put("beta_2", a.beta_2, "--beta-2", "--width-b")
    put("alfa_1", a.alfa_1, "--alfa-1", "--depth-a")
    put("alfa_2", a.alfa_2, "--alfa-2", "--depth-b")
    put("carve_mode", a.carve_mode, "--carve-mode")
    put("channel_cell_width_m", a.channel_cell_width_m, "--channel-cell-width-m")
    put("river_width_cap_m", a.river_width_cap_m, "--river-width-cap-m")
    put("river_depth_cap_m", a.river_depth_cap_m, "--river-depth-cap-m")
    put("max_H_abg_m", a.max_H_abg_m, "--max-H-abg-m", "--max-carve-depth-m")
    put("d4_routing_max_raise_m", a.d4_routing_max_raise_m, "--d4-routing-max-raise-m")
    put("condition_d4_channel_bed", not a.no_condition_d4_channel_bed, "--no-condition-d4-channel-bed")
    put("channel_bed_min_slope", a.channel_bed_min_slope, "--channel-bed-min-slope")
    put("use_least_cost_breaching", not a.no_least_cost_breaching, "--no-least-cost-breaching")
    put("breach_dist_cells", a.breach_dist_cells, "--breach-dist-cells")
    put("breach_max_cost", a.breach_max_cost, "--breach-max-cost")
    put("breach_min_dist", not a.no_breach_min_dist, "--no-breach-min-dist")
    put("breach_flat_increment", a.breach_flat_increment, "--breach-flat-increment")
    put("breach_fill_remaining", a.breach_fill_remaining, "--breach-fill-remaining")
    put("breach_max_depth_m", a.breach_max_depth_m, "--breach-max-depth-m")
    put("breach_max_length_cells", a.breach_max_length_cells, "--breach-max-length-cells")
    put("fill_residual_depressions", a.fill_residual_depressions, "--fill-residual-depressions")
    put("fill_max_depth_m", a.fill_max_depth_m, "--fill-max-depth-m")
    put("make_plots", not a.no_plots, "--no-plots")
    put("allow_geographic_crs", a.allow_geographic_crs, "--allow-geographic-crs")

    return DEMConditioningConfig(**cfg_values)


def main() -> None:
    cfg = parse_args()
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
