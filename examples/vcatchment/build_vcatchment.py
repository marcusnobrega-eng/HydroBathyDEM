"""Build the Di Giammarco V-catchment: fine terrain, coarse mesh.

Geometry (Di Giammarco et al. 1996, the standard 2-D overland flow benchmark):
  * two 400 m planes falling at 0.05 toward a central channel
  * a 20 m wide channel falling at 0.02 along 1,000 m
  * rain 10.8 mm/h for 90 min, then recession
The point for us is the ratio: the channel is 20 m wide and the mesh cells are
60 m, so every channel cell contains a channel narrower than itself -- the same
situation as Pune, where 100% of 90 m river cells hold a river narrower than the
cell. The DEM is written at 5 m so the terrain detail sub-grid needs is present.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, rasterio
from rasterio.transform import from_origin
from shapely import MultiPoint, STRtree, intersection, voronoi_polygons
from shapely.geometry import Polygon, box

OUT = Path("examples/vcatchment"); OUT.mkdir(parents=True, exist_ok=True)
CRS = 'LOCAL_CS["VCatchment",UNIT["metre",1]]'
DX_DEM = 5.0                 # fine terrain
CELL   = 60.0                # coarse mesh -> channel is 1/3 of a cell
LX, LY = 1000.0, 820.0       # 1000 m along channel, +-410 m across
CHAN_HALF, CHAN_SLOPE, PLANE_SLOPE = 10.0, 0.02, 0.05

def bed(x, y):
    """Channel invert falls along +x; planes rise away from the channel."""
    z = CHAN_SLOPE * (LX - x)
    return z + PLANE_SLOPE * np.maximum(np.abs(y) - CHAN_HALF, 0.0)

# ---- fine DEM -------------------------------------------------------------
nx, ny = int(LX/DX_DEM), int(LY/DX_DEM)
xs = (np.arange(nx) + 0.5) * DX_DEM
ys = LY/2 - (np.arange(ny) + 0.5) * DX_DEM
XX, YY = np.meshgrid(xs, ys)
dem = bed(XX, YY)
tr = from_origin(0.0, LY/2, DX_DEM, DX_DEM)
def write(path, arr, dtype="float64"):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype=dtype, crs=CRS, transform=tr, nodata=-9999.0) as d:
        d.write(np.asarray(arr, dtype=dtype), 1)
write(OUT/"vcatchment_dem_5m.tif", dem)
# roughness: channel rougher than the planes, as in the benchmark
n = np.where(np.abs(YY) <= CHAN_HALF, 0.15, 0.015)
write(OUT/"vcatchment_manning_5m.tif", n)
write(OUT/"vcatchment_lulc_5m.tif", np.ones_like(dem))
write(OUT/"vcatchment_soil_5m.tif", np.ones_like(dem))

# ---- coarse mesh: regular lattice -> Voronoi ------------------------------
domain = box(0.0, -LY/2, LX, LY/2)
gx = np.arange(CELL/2, LX, CELL)
gy = np.arange(-LY/2 + CELL/2, LY/2, CELL)
seeds = np.column_stack([a.ravel() for a in np.meshgrid(gx, gy, indexing="ij")])
cells = [intersection(g, domain) for g in voronoi_polygons(MultiPoint(seeds), extend_to=domain).geoms]
cells = [c for c in cells if c.geom_type == "Polygon" and c.area > 1e-9]
area = np.array([c.area for c in cells]); per = np.array([c.length for c in cells])
print(f"DEM  {dem.shape[0]}x{dem.shape[1]} at {DX_DEM:.0f} m   elevation {dem.min():.2f}-{dem.max():.2f} m")
print(f"mesh {len(cells):,} cells at {CELL:.0f} m   A/P min {np.min(area/per):.2f} m")
print(f"channel is {2*CHAN_HALF:.0f} m wide in a {CELL:.0f} m cell -> {2*CHAN_HALF/CELL:.2f} of a cell")
np.save(OUT/"_seeds.npy", seeds)
import pickle
pickle.dump(cells, open(OUT/"_cells.pkl","wb"))
json.dump({"cells":len(cells),"cell_m":CELL,"dem_m":DX_DEM,"channel_m":2*CHAN_HALF,
           "LX":LX,"LY":LY,"chan_slope":CHAN_SLOPE,"plane_slope":PLANE_SLOPE,
           "ap_min":float(np.min(area/per))}, open(OUT/"vcatchment_geometry.json","w"), indent=2)
print("wrote", OUT)
