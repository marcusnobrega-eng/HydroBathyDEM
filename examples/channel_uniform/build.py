"""Analytic test of the sub-grid FACE closure: steady uniform flow in a channel
narrower than a cell.

This is the case sub-grid exists for, and unlike the reservoir it has an answer
that does not come from our own model.

Geometry: a straight prismatic valley 2000 m long, 300 m wide, falling at S=0.001
along x. A rectangular channel 20 m wide and 2 m deep is cut into the middle. The
mesh is 100 m, so the channel is 1/5 of a cell width and 100% of river cells hold
a channel narrower than themselves -- the Pune situation.

Constant rain to steady state. At steady state the outlet discharge is known
exactly (rain x area), and the depth that discharge implies in the TRUE
cross-section is Manning's normal depth:

    Q = (A/n) R^(2/3) sqrt(S),  A = W h,  R = A/(W + 2h)

With rain = 100 mm/h over 2000 x 300 m, Q = 16.667 m3/s, and for W = 20 m,
n = 0.03, S = 0.001 the normal depth is h = 0.900 m. That number is the target.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, geopandas as gpd, rasterio, netCDF4
from rasterio.crs import CRS as RCRS
from rasterio.transform import from_origin
from shapely import MultiPoint, STRtree, intersection, voronoi_polygons
from shapely.geometry import Polygon, box
from dem_processing.hybrid_mesh import _topology, _write_ugrid
from dem_processing.mesh_product import write_conservative_overlap
from dem_processing.subgrid_tables import build_cell_volume_table, build_face_conveyance_table
from scipy.optimize import brentq

OUT = Path("examples/channel_uniform"); OUT.mkdir(parents=True, exist_ok=True)
CRS = 'LOCAL_CS["ChanUni",UNIT["metre",1]]'
DX = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--dem=")), "2.0"))
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
CELL = float(POS[0]) if POS else 100.0
POLICY = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--policy=")), "all")
assert POLICY in {"all", "river", "river_floodplain", "none"}, POLICY
BANK = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--bank=")), "2.0"))
LX, LY = 2000.0, 300.0
W, S, N_CH, N_FP = 20.0, 0.001, 0.03, 0.03
RAIN = 100.0

DEPTH = BANK
def bed(x, y):
    z = S * (LX - x) + DEPTH
    return np.where(np.abs(y - LY/2) <= W/2, z - DEPTH, z)

ncol, nrow = int(LX/DX), int(LY/DX)
xs = (np.arange(ncol)+0.5)*DX; ys = LY - (np.arange(nrow)+0.5)*DX
XX, YY = np.meshgrid(xs, ys)
dem = bed(XX, YY)
man = np.full_like(dem, N_FP); man[np.abs(YY-LY/2) <= W/2] = N_CH
tr = from_origin(0.0, LY, DX, DX)
def w(p, a):
    with rasterio.open(p,"w",driver="GTiff",height=a.shape[0],width=a.shape[1],count=1,
                       dtype="float64",crs=CRS,transform=tr,nodata=-9999.0) as d:
        d.write(a.astype("float64"),1)
w(OUT/f"dem_{int(DX)}m.tif", dem); w(OUT/f"manning_{int(DX)}m.tif", man)
w(OUT/f"lulc_{int(DX)}m.tif", np.where(man < 0.031, 2.0, 1.0)); w(OUT/f"soil_{int(DX)}m.tif", np.ones_like(dem))

Q = LX*LY*RAIN/1000/3600
def conveyance(h):
    """Divided-channel conveyance. Below bank only the channel flows; above it the
    floodplain is added as a separate panel, which is the standard compound-section
    treatment and the same subdivision the face tables perform."""
    if h <= DEPTH:
        A = W*h; P = W + 2*h
        return (A/N_CH)*(A/P)**(2/3)
    Ac = W*h; Pc = W + 2*DEPTH
    Af = (LY - W)*(h - DEPTH); Pf = LY - W
    return (Ac/N_CH)*(Ac/Pc)**(2/3) + (Af/N_FP)*(Af/Pf)**(2/3)
hn = brentq(lambda h: conveyance(h)*np.sqrt(S) - Q, 1e-6, 20.0)
Qbank = conveyance(DEPTH)*np.sqrt(S)
print(f"bank-full capacity {Qbank:.4f} m3/s  ->  flow is "
      f"{'OVERBANK' if Q > Qbank else 'in-bank'}")
print(f"terrain {nrow}x{ncol} at {DX:.0f} m   channel {W:.0f} m wide, {DEPTH:.0f} m deep")
print(f"rain {RAIN:.0f} mm/h over {LX:.0f}x{LY:.0f} m  ->  steady Q = {Q:.4f} m3/s")
print(f"ANALYTIC normal depth above the channel invert: h = {hn:.4f} m")
if hn > DEPTH: print(f"  -> {hn-DEPTH:.4f} m of that is on the floodplain")

domain = box(0.0, 0.0, LX, LY)
gx = np.arange(CELL/2, LX, CELL); gy = np.arange(CELL/2, LY, CELL)
seeds = np.column_stack([a.ravel() for a in np.meshgrid(gx, gy, indexing="ij")])
cells = [intersection(c, domain)
         for c in voronoi_polygons(MultiPoint(seeds), extend_to=domain).geoms]
cells = [c for c in cells if c.geom_type == "Polygon" and c.area > 1e-9]
faces, edges = _topology(cells); cells = [Polygon(ff) for ff in faces]
Ncell = len(cells)
tree = STRtree(cells)
rr, cc = np.mgrid[0:nrow, 0:ncol]
px = tr.c + (cc.ravel()+0.5)*tr.a; py = tr.f + (rr.ravel()+0.5)*tr.e
own = tree.query(gpd.points_from_xy(px, py), predicate="within")
bedc=np.zeros(Ncell); rgh=np.zeros(Ncell); cnt=np.zeros(Ncell)
np.add.at(bedc, own[1], dem.ravel()[own[0]])
np.add.at(rgh, own[1], man.ravel()[own[0]]); np.add.at(cnt, own[1], 1.0)
ok=cnt>0; bedc[ok]/=cnt[ok]; rgh[ok]/=cnt[ok]
areas=np.array([c.area for c in cells]); per=np.array([c.length for c in cells])
gpkg=OUT/f"mesh_{int(CELL)}m_d{int(DX)}.gpkg"
for s_ in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")): s_.unlink(missing_ok=True)
# a cell is "river" if any of its own terrain is channel bed
is_river = np.zeros(Ncell, dtype=bool)
np.logical_or.at(is_river, own[1], (np.abs(py - LY/2) <= W/2)[own[0]])
fclass = np.where(is_river, "river", "floodplain")
if   POLICY == "all":              mask = np.ones(Ncell, dtype=bool)
elif POLICY == "river":            mask = is_river.copy()
elif POLICY == "river_floodplain": mask = np.ones(Ncell, dtype=bool)
else:                              mask = np.zeros(Ncell, dtype=bool)
print(f"feature classes: {int(is_river.sum())} river, {Ncell-int(is_river.sum())} floodplain")
print(f"sub-grid policy '{POLICY}': closure on {int(mask.sum())} of {Ncell} cells")
gpd.GeoDataFrame({"face_id":range(Ncell),"area_m2":areas,"feature_class":fclass,
                  "bed_m":bedc,"manning":rgh}, geometry=cells, crs=CRS
                 ).to_file(gpkg, layer="mesh", driver="GPKG")
nc=OUT/f"mesh_{int(CELL)}m_d{int(DX)}.nc"
_write_ugrid(nc, faces, edges, cells, RCRS.from_wkt(CRS), np.full(Ncell,CELL), tr,
             dem.astype(float), list(fclass), np.zeros(Ncell), np.zeros(Ncell),
             np.zeros(Ncell), np.zeros(Ncell), cell_target_width=np.full(Ncell,CELL))
with netCDF4.Dataset(nc,"a") as ds:
    ds["cell_bed_elevation_m"][:]=bedc; ds["cell_hydraulic_roughness"][:]=rgh
ov=write_conservative_overlap(nc,gpkg,OUT/f"dem_{int(DX)}m.tif",OUT/f"overlap_{int(CELL)}m_d{int(DX)}.nc")
print(f"mesh {Ncell} cells at {CELL:.0f} m   faces {len(edges)}   "
      f"overlap {ov['overlap_count']:,}  area err {ov['maximum_mesh_area_relative_error']:.1e}")

with netCDF4.Dataset(nc) as ds:
    e_own=np.asarray(ds["edge_owner"][:]).reshape(-1); e_nb=np.asarray(ds["edge_neighbor"][:]).reshape(-1)
    e_len=np.asarray(ds["edge_length_m"][:]).reshape(-1)
    mx=np.asarray(ds["edge_midpoint_x"][:]).reshape(-1); my=np.asarray(ds["edge_midpoint_y"][:]).reshape(-1)
    nx_=np.asarray(ds["edge_normal_x"][:]).reshape(-1); ny_=np.asarray(ds["edge_normal_y"][:]).reshape(-1)
    ca=np.asarray(ds["cell_area_m2"][:]).reshape(-1)
with netCDF4.Dataset(OUT/f"overlap_{int(CELL)}m_d{int(DX)}.nc") as ds:
    oc=np.asarray(ds["overlap_mesh_index"][:]).reshape(-1)
    oi=np.asarray(ds["overlap_raster_index"][:]).reshape(-1)
    oa=np.asarray(ds["overlap_area_m2"][:]).reshape(-1)
dsu=np.flipud(dem).reshape(-1); msu=np.flipud(man).reshape(-1)
tabc=build_cell_volume_table(oc,oa,dsu[oi],Ncell,ca,
                             level_pool_mask=mask, cell_bed_mean_m=bedc)
inv=~tr
def samp(a,X,Y):
    c_,r_=inv*(X,Y)
    return a[np.clip(np.floor(r_).astype(int),0,nrow-1), np.clip(np.floor(c_).astype(int),0,ncol-1)]
OFF=0.5*DX; profiles=[]
for i in range(len(e_own)):
    L=float(e_len[i]); npts=max(3,int(np.ceil(L/DX))+1)
    s_=np.linspace(-0.5*L,0.5*L,npts); tx,ty=-ny_[i],nx_[i]
    Xa,Ya=mx[i]+s_*tx-OFF*nx_[i], my[i]+s_*ty-OFF*ny_[i]
    Xb,Yb=mx[i]+s_*tx+OFF*nx_[i], my[i]+s_*ty+OFF*ny_[i]
    z=np.maximum(samp(dem,Xa,Ya),samp(dem,Xb,Yb))
    nn=0.5*(samp(man,Xa,Ya)+samp(man,Xb,Yb))
    profiles.append((s_-s_[0], z, np.where(nn>0,nn,N_FP)))
floor=np.where(e_nb>=0, np.minimum(tabc.datum_m[e_own],tabc.datum_m[np.maximum(e_nb,0)]),
               tabc.datum_m[e_own])
profiles=[(a_,np.maximum(b_,floor[i]),c_) for i,(a_,b_,c_) in enumerate(profiles)]
tabf=build_face_conveyance_table(profiles,e_len)
low=np.minimum(tabc.datum_m[e_own],tabc.datum_m[np.maximum(e_nb,0)])
print(f"invariant violations: {int(((low-tabf.datum_m)>1e-6)[e_nb>=0].sum())}")
with netCDF4.Dataset(OUT/f"subgrid_{int(CELL)}m_d{int(DX)}_{POLICY}.nc","w") as o:
    o.createDimension("cell",Ncell); o.createDimension("cell_pt",tabc.zeta_m.shape[1])
    o.createDimension("face",len(e_own)); o.createDimension("face_pt",tabf.zeta_m.shape[1])
    def v(nm,d,a): o.createVariable(nm,"f8" if a.dtype.kind=="f" else "i8",d)[:]=a
    v("cell_datum_m",("cell",),tabc.datum_m); v("cell_zeta_m",("cell","cell_pt"),tabc.zeta_m)
    v("cell_volume_m3",("cell","cell_pt"),tabc.volume_m3)
    v("cell_wet_area_m2",("cell","cell_pt"),tabc.area_m2)
    v("cell_point_count",("cell",),tabc.count); v("cell_plan_area_m2",("cell",),tabc.plan_area_m2)
    v("face_datum_m",("face",),tabf.datum_m); v("face_zeta_m",("face","face_pt"),tabf.zeta_m)
    v("face_flow_area_m2",("face","face_pt"),tabf.area_m2)
    v("face_perimeter_m",("face","face_pt"),tabf.perimeter_m)
    v("face_conveyance",("face","face_pt"),tabf.conveyance)
    v("face_point_count",("face",),tabf.count); v("face_length_m",("face",),tabf.length_m)
    v("cell_is_subgrid",("cell",),mask.astype(np.int64))
print(f"tables: {Ncell} cells x{tabc.zeta_m.shape[1]}, {len(e_own)} faces x{tabf.zeta_m.shape[1]}")
json.dump({"cells":Ncell,"cell_m":CELL,"dem_m":DX,"within_cell_relief_m":float(S*CELL),"LX":LX,"LY":LY,"W":W,"DEPTH":DEPTH,
           "S":S,"n_ch":N_CH,"n_fp":N_FP,"rain_mm_h":RAIN,"Q_steady_m3_s":Q,
           "h_normal_m":hn,"nrow":nrow,"ncol":ncol,"Q_bankfull_m3_s":Qbank,
           "policy":POLICY,"n_river_cells":int(is_river.sum())},
          open(OUT/f"case_{int(CELL)}m_d{int(DX)}.json","w"), indent=2)
print("wrote", OUT)
