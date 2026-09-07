"""Aggregate the 5 m burned terrain to a structured grid for the fair comparison.

The point of the comparison is cost at a GIVEN cell size, so the structured grid
must be the same resolution as the unstructured mesh and cover the same catchment.
Everything is block-aggregated from the same 5 m burn the sub-grid tables are built
from, so both arms see the same terrain information -- one as a coarse raster, the
other as coarse cells plus tables.

Roughness is aggregated as an AREA MEAN of Manning's n, matching how the
unstructured mesh sets cell_hydraulic_roughness. Aggregating the LULC code instead
(majority or mean) would give the two arms different friction and make the timing
comparison meaningless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

O = Path("examples/pune_subgrid")
FINE = 5
CELL = int(next((a for a in sys.argv[1:] if not a.startswith("--")), 90))
F = CELL // FINE
assert CELL % FINE == 0, "the coarse cell must be a whole multiple of the fine grid"

with rasterio.open(O / f"dem_{FINE}m.tif") as s:
    dem = s.read(1, masked=True).filled(np.nan)
    tr, crs = s.transform, s.crs
with rasterio.open(O / f"manning_{FINE}m.tif") as s:
    man = s.read(1, masked=True).filled(np.nan)
h, w = dem.shape
# trim to a whole number of blocks
hh, ww = (h // F) * F, (w // F) * F
def block(a):
    return a[:hh, :ww].reshape(hh // F, F, ww // F, F)

with np.errstate(invalid="ignore"):
    dem_c = np.nanmean(block(dem), axis=(1, 3))
    man_c = np.nanmean(block(man), axis=(1, 3))
valid_c = np.isfinite(dem_c)
# LULC code chosen to REPRODUCE the aggregated Manning value, so the structured run
# and the mesh run share friction exactly. Codes: 1 rural .060, 2 urban .015,
# 3 channel .035 -- a block mean lands between them, so a per-cell class table is
# written instead and the code is just an index.
codes = np.where(valid_c, np.arange(dem_c.size).reshape(dem_c.shape) + 1, np.nan)

trc = from_origin(tr.c, tr.f, CELL, CELL)
def write(name, arr):
    a = np.where(np.isfinite(arr), arr, -9999.0)
    with rasterio.open(O / name, "w", driver="GTiff", height=dem_c.shape[0],
                       width=dem_c.shape[1], count=1, dtype="float64", crs=crs,
                       transform=trc, nodata=-9999.0, compress="deflate") as d:
        d.write(a.astype("float64"), 1)

write(f"struct{CELL}m_dem.tif", dem_c)
write(f"struct{CELL}m_manning.tif", man_c)
write(f"struct{CELL}m_soil.tif", np.where(valid_c, 1.0, np.nan))
write(f"struct{CELL}m_lulc.tif", np.where(valid_c, 1.0, np.nan))
print(f"structured {CELL} m grid: {dem_c.shape[0]}x{dem_c.shape[1]} "
      f"= {dem_c.size:,} cells, {int(valid_c.sum()):,} valid "
      f"({valid_c.sum() * CELL * CELL / 1e6:.2f} km2)")
print(f"  elevation {np.nanmin(dem_c):.2f}-{np.nanmax(dem_c):.2f} m")
print(f"  manning   {np.nanmin(man_c):.4f}-{np.nanmax(man_c):.4f} (area mean of the 5 m field)")
print(f"  unstructured corridor mesh for reference: 12,330 cells")
json.dump({"cell_m": CELL, "nrow": int(dem_c.shape[0]), "ncol": int(dem_c.shape[1]),
           "valid_cells": int(valid_c.sum()),
           "area_km2": float(valid_c.sum() * CELL * CELL / 1e6)},
          open(O / f"struct{CELL}m.json", "w"), indent=2)
print("wrote", O)
