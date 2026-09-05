import os, json
import numpy as np, rasterio, geopandas as gpd
from rasterio import windows, features, transform as rtransform
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union
from scipy import ndimage
import common as C

OUT = os.path.join(C.EX, "data")
shape, prof, recv, level, order, accum, basin = C.topology()
rows, cols = shape
n = rows * cols
A = accum * C.CELL_KM2
lab = np.load(os.path.join(C.CACHE, "cand_labels.npy"))
cand = np.load(os.path.join(C.CACHE, "cand_outlets.npy"))
z = np.load(os.path.join(C.CACHE, "masks.npz"))
urban, fp, rmask, wbr, dom = z["urban"], z["fp"], z["rmask"], z["wbr"], z["dom"]
dem, dem_prof = C.read("dem")
rw, _ = C.read("rw"); rd, _ = C.read("rd")
wb = gpd.read_file(C.P["wb"], layer="waterbody_refinement")
T = prof["transform"]
HALO = 10   # 300 m halo on the clip window

CASES = {"caseA": 0, "caseB": 3}
report = {}

def pct(a, qs=(0, 5, 25, 50, 75, 90, 95, 99, 100)):
    if a.size == 0:
        return {}
    v = np.percentile(a.astype(float), qs)
    return {f"p{q}": round(float(x), 4) for q, x in zip(qs, v)}

def coarse_jumps(elev, valid, f):
    """Block-mean aggregate by factor f, then D4 neighbour |dz| between
    blocks that are both >=50% inside the catchment."""
    r, c = elev.shape
    R, Cc = (r // f) * f, (c // f) * f
    e = np.where(valid, elev, 0.0)[:R, :Cc].reshape(R // f, f, Cc // f, f)
    w = valid[:R, :Cc].astype(float).reshape(R // f, f, Cc // f, f)
    num = e.sum(axis=(1, 3)); den = w.sum(axis=(1, 3))
    ok = den >= 0.5 * f * f
    mean = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    d = []
    for ax in (0, 1):
        a = np.take(mean, range(mean.shape[ax] - 1), axis=ax)
        b = np.take(mean, range(1, mean.shape[ax]), axis=ax)
        oa = np.take(ok, range(ok.shape[ax] - 1), axis=ax)
        ob = np.take(ok, range(1, ok.shape[ax]), axis=ax)
        g = oa & ob
        d.append(np.abs(a[g] - b[g]))
    return np.concatenate(d)

for name, k in CASES.items():
    m = (lab == k)
    out = int(cand[k])
    m2 = m.reshape(shape)
    cnt = int(m.sum())
    # ---------------- verification ----------------
    inside = np.flatnonzero(m)
    r_in = recv[inside]
    # (1) closure downstream: every masked cell except the outlet has a masked receiver
    non_out = inside[inside != out]
    assert np.all(r_in[inside != out] >= 0), "masked cell with no receiver other than outlet"
    assert np.all(m[recv[non_out]]), "masked cell drains outside the mask"
    # (2) outlet leaves the mask
    assert recv[out] >= 0 and not m[recv[out]], "outlet does not exit the mask"
    # (3) closure upstream: nothing outside drains in  => mask == full upstream area
    hasr = recv >= 0
    ext = np.flatnonzero(hasr & ~m)
    assert not np.any(m[recv[ext]]), "external cell drains into the mask"
    # (4) acyclic (level array exists) => every cell reaches the outlet in <=level steps
    # explicit walk from the 200 highest-level cells as a belt-and-braces check
    probe = inside[np.argsort(level[inside])[-200:]]
    steps_max = 0
    for p in probe:
        cur, s = int(p), 0
        while cur != out:
            cur = int(recv[cur]); s += 1
            assert s <= int(level[out]) + 2, "walk did not terminate at outlet"
        steps_max = max(steps_max, s)
    # (5) connectivity
    n4 = ndimage.label(m2, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))[1]
    n8 = ndimage.label(m2, structure=np.ones((3,3)))[1]
    # (6) area agrees with the published D8 upstream-area raster at the outlet
    uaa, _ = C.read("uaa")
    uaa_out = float(uaa.ravel()[out])

    # ---------------- metrics ----------------
    ox, oy = rtransform.xy(T, out // cols, out % cols)
    ll = gpd.GeoSeries(gpd.points_from_xy([ox], [oy]), crs="EPSG:32643").to_crs(4326)
    riv = m2 & rmask
    riv_w = rw.reshape(shape)[riv]; riv_w = riv_w[np.isfinite(riv_w) & (riv_w > 0)]
    riv_d = rd.reshape(shape)[riv]; riv_d = riv_d[np.isfinite(riv_d) & (riv_d > 0)]
    e = dem.reshape(shape)[m2]; e = e[e > -9000]
    wbin = wbr.reshape(shape)[m2]
    wids = np.unique(wbin); wids = wids[wids > 0]
    # 30 m D4 neighbour jumps inside the catchment
    de = []
    E = np.where(m2, dem, np.nan)
    for ax in (0, 1):
        a = np.take(E, range(E.shape[ax]-1), axis=ax); b = np.take(E, range(1, E.shape[ax]), axis=ax)
        d = np.abs(a - b); de.append(d[np.isfinite(d)])
    de = np.concatenate(de)

    rec = dict(
        case=name, outlet_flat=out, outlet_row=out // cols, outlet_col=out % cols,
        outlet_x_utm43n=round(ox, 1), outlet_y_utm43n=round(oy, 1),
        outlet_lon=round(float(ll.x[0]), 6), outlet_lat=round(float(ll.y[0]), 6),
        area_km2=round(cnt * C.CELL_KM2, 3), n_cells_30m=cnt,
        area_km2_published_uaa_raster=round(uaa_out, 3),
        drains_to_cell_flat=int(recv[out]),
        connected_components_4=int(n4), connected_components_8=int(n8),
        max_flow_path_steps_probe=int(steps_max), max_d8_level_in_basin=int(level[inside].max()),
        river_length_km=round(riv.sum() * C.CELL / 1000.0, 2),
        river_cells=int(riv.sum()),
        river_density_km_per_km2=round(riv.sum()*C.CELL/1000.0 / (cnt*C.CELL_KM2), 3),
        river_width_m=pct(riv_w),
        river_width_mean_m=round(float(riv_w.mean()), 3),
        river_pct_width_ge_30m=round(100*float((riv_w >= 30).mean()), 2),
        river_pct_width_ge_90m=round(100*float((riv_w >= 90).mean()), 2),
        bankfull_depth_m=pct(riv_d), bankfull_depth_mean_m=round(float(riv_d.mean()), 4),
        urban_pct=round(100*float(urban.reshape(shape)[m2].mean()), 2),
        urban_km2=round(float(urban.reshape(shape)[m2].sum())*C.CELL_KM2, 2),
        floodplain_pct=round(100*float(fp.reshape(shape)[m2].mean()), 2),
        floodplain_km2=round(float(fp.reshape(shape)[m2].sum())*C.CELL_KM2, 2),
        waterbody_pct=round(100*float((wbin > 0).mean()), 2),
        waterbody_km2=round(float((wbin > 0).sum())*C.CELL_KM2, 3),
        waterbodies=[dict(waterbody_id=int(wb.waterbody_id.iloc[i-1]),
                          waterbody_type=str(wb.waterbody_type.iloc[i-1]),
                          dem_water_level_m=float(wb.dem_water_level_m.iloc[i-1]),
                          full_area_km2=round(float(wb.area_m2.iloc[i-1])/1e6, 3),
                          area_in_case_km2=round(float((wbin == i).sum())*C.CELL_KM2, 3))
                     for i in wids],
        elev_min_m=round(float(e.min()), 2), elev_max_m=round(float(e.max()), 2),
        relief_m=round(float(e.max()-e.min()), 2),
        elev_pct=pct(e),
        elev_at_outlet_m=round(float(dem.ravel()[out]), 2),
        dem_jump_30m_D4_m=pct(de), dem_jump_30m_D4_mean_m=round(float(de.mean()), 3),
        dem_jump_coarse_m={f"{int(30*f)}m": pct(coarse_jumps(dem, m2, f))
                           for f in (1, 3, 5, 7)},
        dem_jump_coarse_max_m={f"{int(30*f)}m": round(float(coarse_jumps(dem, m2, f).max()), 2)
                               for f in (1, 3, 5, 7)},
        inside_valid_domain_pct=round(100*float(dom.reshape(shape)[m2].mean()), 3),
    )
    report[name] = rec

    # ---------------- write clipped products ----------------
    r0, r1 = np.where(m2.any(axis=1))[0][[0, -1]]
    c0, c1 = np.where(m2.any(axis=0))[0][[0, -1]]
    r0, c0 = max(0, r0-HALO), max(0, c0-HALO)
    r1, c1 = min(rows-1, r1+HALO), min(cols-1, c1+HALO)
    win = windows.Window(c0, r0, c1-c0+1, r1-r0+1)
    wt = windows.transform(win, T)
    d = os.path.join(OUT, f"pune_{name}")
    os.makedirs(d, exist_ok=True)
    rec["clip_window_rowcol"] = [int(r0), int(r1), int(c0), int(c1)]
    rec["clip_shape"] = [int(win.height), int(win.width)]
    rec["clip_bounds_utm43n"] = [round(v, 1) for v in windows.bounds(win, T)]

    written = []
    def wr(fn, arr, dtype, nodata):
        p = os.path.join(d, fn)
        with rasterio.open(p, "w", driver="GTiff", height=win.height, width=win.width,
                           count=1, dtype=dtype, crs=prof["crs"], transform=wt,
                           nodata=nodata, compress="deflate", tiled=True,
                           blockxsize=256, blockysize=256) as dst:
            dst.write(np.asarray(arr, dtype=dtype), 1)
        written.append(p)

    sl = (slice(r0, r1+1), slice(c0, c1+1))
    mw = m2[sl]
    wr(f"{name}_catchment_mask.tif", mw, "uint8", 0)
    for key, fn, dt, nd in [
        ("dem", "dem_hydraulic_conditioned_m", "float32", -9999.0),
        ("fdir", "d8_flow_direction", "uint8", 0),
        ("rmask", "d8_river_mask_2km2", "uint8", 0),
        ("rw", "d8_river_width_m", "float32", -9999.0),
        ("rd", "d8_river_depth_m", "float32", -9999.0),
        ("rbed", "d8_river_bed_elevation_m", "float32", -9999.0),
        ("uaa", "d8_upstream_area_km2", "float32", -9999.0),
        ("imp", "impervious_esa_pct", "uint8", 255),
        ("lulc", "lulc_esa", "uint8", 0),
        ("soil", "soil", "uint8", 0),
    ]:
        a, _ = C.read(key)
        wr(f"{name}_{fn}.tif", a[sl], dt, nd)
    wr(f"{name}_urban_mask.tif", urban.reshape(shape)[sl], "uint8", 0)
    wr(f"{name}_floodplain_mask.tif", fp.reshape(shape)[sl], "uint8", 0)
    wr(f"{name}_waterbody_id.tif", wbr.reshape(shape)[sl], "int32", 0)
    # DEM masked to the catchment (nodata outside) for convenience
    demw = np.where(mw, dem[sl], -9999.0)
    wr(f"{name}_dem_catchment_only_m.tif", demw, "float32", -9999.0)

    # domain polygon
    geoms = [shp_shape(g) for g, v in features.shapes(mw.astype("uint8"), mask=mw,
                                                      transform=wt) if v == 1]
    poly = unary_union(geoms)
    gj = os.path.join(OUT, f"pune_{name}_domain.geojson")
    gdf = gpd.GeoDataFrame([{k: v for k, v in rec.items()
                             if isinstance(v, (int, float, str))}],
                           geometry=[poly], crs=prof["crs"])
    gdf.to_file(gj, driver="GeoJSON")
    rec["polygon_area_km2"] = round(poly.area/1e6, 4)
    rec["polygon_n_parts"] = 1 if poly.geom_type == "Polygon" else len(poly.geoms)
    rec["polygon_n_holes"] = (len(poly.interiors) if poly.geom_type == "Polygon"
                              else sum(len(p.interiors) for p in poly.geoms))
    # outlet point
    op = os.path.join(OUT, f"pune_{name}_outlet.geojson")
    gpd.GeoDataFrame([dict(case=name, area_km2=rec["area_km2"])],
                     geometry=gpd.points_from_xy([ox], [oy]),
                     crs=prof["crs"]).to_file(op, driver="GeoJSON")
    rec["files"] = dict(domain_geojson=gj, outlet_geojson=op, rasters=sorted(written))

json.dump(report, open(os.path.join(OUT, "pune_subcatchment_cases.json"), "w"), indent=2)
print(json.dumps(report, indent=1))
