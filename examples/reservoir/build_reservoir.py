"""A reservoir case: the geometry where the sub-grid premise actually holds.

The V-catchment showed the limit of the method -- on 5% planes the water surface
drops 3.85 m inside a 60 m cell, so the level-pool assumption behind the cell
elevation-volume curve fails and depths come out 148% high.

A reservoir is the opposite. The pool surface IS horizontal, exactly as the table
assumes, and the interesting cells are the ones the shoreline cuts: a cell half in
and half out of the pool holds a volume the flat-prism model can only smear over
its whole footprint.  That is the case sub-grid is built for, and it is the case
the Pune catchment has 42 of.

Geometry: 600 x 600 m, gentle 0.3% fall to an east outlet, with a 150 m wide
circular basin 3 m deep in the middle.  Mesh at 60 m, so the basin is 2.5 cells
across and its rim cuts through cells rather than following their edges.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, rasterio
from rasterio.transform import from_origin
from shapely import MultiPoint, intersection, voronoi_polygons
from shapely.geometry import box

OUT = Path("examples/reservoir"); OUT.mkdir(parents=True, exist_ok=True)
CRS = 'LOCAL_CS["Reservoir",UNIT["metre",1]]'
DX_DEM, CELL = 5.0, 60.0
LX = LY = 600.0
BASIN_X, BASIN_Y, BASIN_R, BASIN_D = 300.0, 300.0, 75.0, 3.0
SLOPE = 0.003

def bed(x, y):
    """Gentle plain falling east, with a smooth circular basin cut into it."""
    z = 10.0 + SLOPE * (LX - x)
    r = np.hypot(x - BASIN_X, y - BASIN_Y)
    # a smooth bowl, so the shoreline sweeps across cells as it fills
    bowl = BASIN_D * np.maximum(1.0 - (r / BASIN_R) ** 2, 0.0)
    return z - bowl

n = int(LX / DX_DEM)
xs = (np.arange(n) + 0.5) * DX_DEM
ys = LY - (np.arange(n) + 0.5) * DX_DEM
XX, YY = np.meshgrid(xs, ys)
dem = bed(XX, YY)
tr = from_origin(0.0, LY, DX_DEM, DX_DEM)

def write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float64", crs=CRS, transform=tr, nodata=-9999.0) as d:
        d.write(np.asarray(arr, dtype="float64"), 1)

write(OUT/"reservoir_dem_5m.tif", dem)
manning = np.where(np.hypot(XX-BASIN_X, YY-BASIN_Y) <= BASIN_R, 0.030, 0.050)
write(OUT/"reservoir_manning_5m.tif", manning)
write(OUT/"reservoir_lulc_5m.tif", np.ones_like(dem))
write(OUT/"reservoir_soil_5m.tif", np.ones_like(dem))

domain = box(0.0, 0.0, LX, LY)
g = np.arange(CELL/2, LX, CELL)
seeds = np.column_stack([a.ravel() for a in np.meshgrid(g, g, indexing="ij")])
cells = [intersection(c, domain)
         for c in voronoi_polygons(MultiPoint(seeds), extend_to=domain).geoms]
cells = [c for c in cells if c.geom_type == "Polygon" and c.area > 1e-9]
area = np.array([c.area for c in cells]); per = np.array([c.length for c in cells])
# how many cells does the shoreline actually cut?  those are the ones that matter
rim = 0
for c in cells:
    rr = np.hypot(*(np.asarray(c.exterior.coords).T - np.array([[BASIN_X],[BASIN_Y]])))
    if rr.min() < BASIN_R < rr.max():
        rim += 1
print(f"DEM  {dem.shape[0]}x{dem.shape[1]} at {DX_DEM:.0f} m   elevation {dem.min():.2f}-{dem.max():.2f} m")
print(f"mesh {len(cells)} cells at {CELL:.0f} m   A/P min {np.min(area/per):.2f} m")
print(f"basin {2*BASIN_R:.0f} m across = {2*BASIN_R/CELL:.1f} cells; shoreline cuts {rim} cells")
pickle.dump(cells, open(OUT/"_cells.pkl","wb"))
json.dump({"cells":len(cells),"cell_m":CELL,"dem_m":DX_DEM,"LX":LX,"LY":LY,
           "basin_r":BASIN_R,"basin_d":BASIN_D,"slope":SLOPE,"rim_cells":rim,
           "ap_min":float(np.min(area/per))}, open(OUT/"reservoir_geometry.json","w"), indent=2)
print("wrote", OUT)
