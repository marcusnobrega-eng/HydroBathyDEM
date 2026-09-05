"""Write the V-catchment mesh as UGRID + conservative overlap."""
from __future__ import annotations
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, geopandas as gpd, rasterio
from dem_processing.hybrid_mesh import _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap

OUT = Path("examples/vcatchment")
cells = pickle.load(open(OUT/"_cells.pkl","rb"))
geo = json.load(open(OUT/"vcatchment_geometry.json"))
with rasterio.open(OUT/"vcatchment_dem_5m.tif") as src:
    dem = src.read(1); transform, crs = src.transform, src.crs
with rasterio.open(OUT/"vcatchment_manning_5m.tif") as src:
    nras = src.read(1)

faces, edges = _topology(cells)
cells = [__import__("shapely").geometry.Polygon(f) for f in faces]
n = len(cells)
# area-averaged bed and roughness per cell, from the FINE raster
from shapely import STRtree, area as sarea, box as sbox
tree = STRtree(cells)
bed = np.zeros(n); rough = np.zeros(n); wsum = np.zeros(n)
h, w = dem.shape
rows, cols = np.mgrid[0:h, 0:w]
xs = transform.c + (cols.ravel()+0.5)*transform.a
ys = transform.f + (rows.ravel()+0.5)*transform.e
own = tree.query(gpd.points_from_xy(xs, ys), predicate="within")
zf = dem.ravel(); nf = nras.ravel()
np.add.at(bed,   own[1], zf[own[0]])
np.add.at(rough, own[1], nf[own[0]])
np.add.at(wsum,  own[1], 1.0)
ok = wsum > 0
bed[ok] /= wsum[ok]; rough[ok] /= wsum[ok]
bed[~ok] = np.nan
if (~ok).any():   # fall back to centroid sample
    import rasterio.transform as rt
    for i in np.flatnonzero(~ok):
        p = cells[i].representative_point()
        r, c = rt.rowcol(transform, p.x, p.y)
        bed[i] = dem[np.clip(r,0,h-1), np.clip(c,0,w-1)]
        rough[i] = nras[np.clip(r,0,h-1), np.clip(c,0,w-1)]
areas = np.array([c.area for c in cells])
target = np.full(n, geo["cell_m"])
gpkg = OUT/"vcatchment_mesh.gpkg"
for s in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")): s.unlink(missing_ok=True)
gpd.GeoDataFrame({"face_id":range(n),"area_m2":areas,"feature_class":["rural"]*n,
                  "bed_m":bed,"manning":rough}, geometry=cells, crs=crs
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc = OUT/"vcatchment_mesh.nc"
rep = _write_ugrid(nc, faces, edges, cells, crs, target, transform, dem.astype(float),
                   ["rural"]*n, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                   cell_target_width=target)
# overwrite bed and roughness with the area averages, and record the fine paths
import netCDF4
with netCDF4.Dataset(nc, "a") as ds:
    ds["cell_bed_elevation_m"][:] = bed
    ds["cell_hydraulic_roughness"][:] = rough
    ds.subgrid_dem_path = str((OUT/"vcatchment_dem_5m.tif").resolve())
    ds.subgrid_manning_path = str((OUT/"vcatchment_manning_5m.tif").resolve())
ov = write_conservative_overlap(nc, gpkg, OUT/"vcatchment_dem_5m.tif", OUT/"vcatchment_mesh_overlap.nc")
print(f"cells {n}   faces {len(edges)}   bed {np.nanmin(bed):.2f}-{np.nanmax(bed):.2f} m")
print(f"manning {rough.min():.3f}-{rough.max():.3f}   overlap pairs {ov['overlap_count']:,}"
      f"   area err {ov['maximum_mesh_area_relative_error']:.2e}")
print(f"export report: {rep}")
