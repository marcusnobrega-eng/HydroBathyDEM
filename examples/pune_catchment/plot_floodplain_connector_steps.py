"""Show the intermediate terrain-to-connector steps for one floodplain patch."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import binary_erosion, distance_transform_edt

from dem_processing.design_hydrograph import receiver_from_d4_direction
from dem_processing.config import load_config_file
from dem_processing.hybrid_mesh import HybridMeshConfig, _smoothed_floodplain_connector_tangents


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "pune_design_corridor_mesh_full_domain_step5" / "diagnostics"
CORRIDOR = ROOT / "outputs" / "pune_design_corridor_gcn250"
HYDRO = ROOT / "outputs" / "hydrobathy_20km2_corrected_dem"
MESH_CONFIG = ROOT / "configs" / "pune_design_corridor_mesh_full_domain_step5.json"


def _read(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    source = rasterio.open(path)
    return source.read(1, masked=True).astype(float).filled(np.nan), source


def _trace(start: int, receiver: np.ndarray, river: np.ndarray, maximum_steps: int = 10_000) -> list[int]:
    path, current, seen = [start], start, {start}
    for _ in range(maximum_steps):
        if river.ravel()[current]:
            break
        target = int(receiver.ravel()[current])
        if target < 0 or target in seen:
            break
        path.append(target); seen.add(target); current = target
    return path


def _one_axis_cell_per_bin(axis: np.ndarray, accumulation: np.ndarray, within: np.ndarray, transform: rasterio.Affine, spacing_m: float) -> np.ndarray:
    """Keep one strongest local connector per map bin so arrows stay legible."""
    rows, cols = np.where(axis & within)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for row, col, x, y in zip(rows, cols, xs, ys, strict=True):
        key = (int(np.floor(x / spacing_m)), int(np.floor(y / spacing_m)))
        score = float(accumulation[row, col])
        index = int(row * axis.shape[1] + col)
        if key not in best or score > best[key][0]:
            best[key] = (score, index)
    return np.asarray([item[1] for item in best.values()], dtype=np.int64)


def main() -> None:
    terrain, source = _read(ROOT / "data" / "pune_corrected_30m" / "pune_dem_corrected_aligned_30m.tif")
    transform, crs, raster_bounds = source.transform, source.crs, source.bounds
    source.close()
    potential, source = _read(CORRIDOR / "D4_design_floodplain_hydraulic_potential_m.tif")
    source.close()
    direction, source = _read(CORRIDOR / "D4_design_floodplain_flow_direction.tif")
    source.close()
    axis, source = _read(CORRIDOR / "D4_design_floodplain_axis_mask.tif")
    source.close()
    river, source = _read(HYDRO / "d4" / "D4_idx_facc.tif")
    source.close()
    components = gpd.read_file(CORRIDOR / "D4_design_floodplain_corridor_100yr.gpkg").explode(index_parts=False).reset_index(drop=True)
    # The uppermost component is the blue patch at the top of the previous
    # Step-5 inspection window. This keeps the diagnostic reproducible.
    component = components.iloc[int(np.argmax(components.centroid.y))].geometry
    mask = geometry_mask([component], out_shape=terrain.shape, transform=transform, invert=True)
    river_mask = (river > 0) & mask
    receiver = receiver_from_d4_direction(np.nan_to_num(direction, nan=0.0))
    edge = mask & ~binary_erosion(mask)
    candidates = np.flatnonzero(edge & np.isfinite(potential))
    candidates = candidates[np.argsort(potential.ravel()[candidates])[::-1]]
    starts: list[int] = []
    for item in candidates:
        row, col = divmod(int(item), mask.shape[1])
        if all(np.hypot(row - divmod(other, mask.shape[1])[0], col - divmod(other, mask.shape[1])[1]) >= 12 for other in starts):
            starts.append(int(item))
        if len(starts) == 8:
            break
    bounds = component.bounds
    pad = 0.12 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    extent = (bounds[0] - pad, bounds[2] + pad, bounds[1] - pad, bounds[3] + pad)
    display = np.where(mask, terrain, np.nan)
    potential_display = np.where(mask, potential, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), layout="constrained")
    panels = axes.ravel()
    raster_extent = (raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top)
    panels[0].imshow(display, extent=raster_extent, origin="upper", cmap="terrain")
    gpd.GeoSeries([component], crs=crs).boundary.plot(ax=panels[0], color="#3594CC", linewidth=1.5)
    panels[0].imshow(np.where(river_mask, 1.0, np.nan), extent=raster_extent, origin="upper", cmap="Blues", alpha=0.9)
    panels[0].set_title("(a) Manchas e canal receptor", fontsize=12, fontname="Helvetica")
    image = panels[1].imshow(potential_display, extent=raster_extent, origin="upper", cmap="viridis")
    panels[1].imshow(np.where(river_mask, 1.0, np.nan), extent=raster_extent, origin="upper", cmap="Blues", alpha=0.85)
    fig.colorbar(image, ax=panels[1], shrink=0.78, label="Potencial hidráulico (m)")
    panels[1].set_title("(b) Terreno condicionado até o canal", fontsize=12, fontname="Helvetica")
    panels[2].imshow(display, extent=raster_extent, origin="upper", cmap="Greys", alpha=0.6)
    gpd.GeoSeries([component], crs=crs).plot(ax=panels[2], color="#8CC5E3", edgecolor="#3594CC", alpha=0.45)
    for start in starts:
        path = _trace(start, receiver, river_mask)
        rows, cols = np.divmod(path, terrain.shape[1])
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        panels[2].plot(xs, ys, color="#9D2C00", linewidth=1.4)
        panels[2].plot(xs[0], ys[0], "o", color="#E25759", markersize=3)
    panels[2].set_title("(c) Trajetórias de máxima descida ao canal", fontsize=12, fontname="Helvetica")
    panels[3].imshow(display, extent=raster_extent, origin="upper", cmap="Greys", alpha=0.55)
    gpd.GeoSeries([component], crs=crs).plot(ax=panels[3], color="#8CC5E3", edgecolor="none", alpha=0.4)
    panels[3].imshow(np.where(axis > 0, 1.0, np.nan), extent=raster_extent, origin="upper", cmap="Blues", alpha=0.95)
    panels[3].set_title("(d) Eixos locais por acumulação", fontsize=12, fontname="Helvetica")
    for axis in panels:
        axis.set_xlim(extent[0], extent[1]); axis.set_ylim(extent[2], extent[3]); axis.set_aspect("equal"); axis.set_axis_off()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "floodplain_connector_steps_upper_patch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_region_04_alignment() -> None:
    """Explain the large Region-4 corridor and the direction actually available to the mesh."""
    terrain, source = _read(ROOT / "data" / "pune_corrected_30m" / "pune_dem_corrected_aligned_30m.tif")
    transform, crs, bounds = source.transform, source.crs, source.bounds
    source.close()
    floodplain, source = _read(CORRIDOR / "D4_design_floodplain_mask_100yr.tif")
    source.close()
    direction, source = _read(CORRIDOR / "D4_design_floodplain_flow_direction.tif")
    source.close()
    accumulation, source = _read(CORRIDOR / "D4_design_floodplain_flow_accumulation_km2.tif")
    source.close()
    axis, source = _read(CORRIDOR / "D4_design_floodplain_axis_mask.tif")
    source.close()
    river, source = _read(HYDRO / "d4" / "D4_idx_facc.tif")
    source.close()
    windows = gpd.read_file(ROOT / "data" / "pune_floodplain_orientation_windows.geojson").to_crs(crs)
    window = windows.geometry.iloc[3]
    within = geometry_mask([window], out_shape=terrain.shape, transform=transform, invert=True)
    floodplain_all = floodplain > 0
    river_all = river > 0
    axis_all = (axis > 0) & floodplain_all
    floodplain = floodplain_all & within
    river = river_all & within
    axis = axis_all & within
    receiver = receiver_from_d4_direction(np.nan_to_num(direction, nan=0.0))
    mesh_config = HybridMeshConfig.from_mapping(load_config_file(MESH_CONFIG))
    tangent_x, tangent_y, connector_axes = _smoothed_floodplain_connector_tangents(
        axis_all, receiver, transform, mesh_config.floodplain_along_river_cell_length_m,
    )
    distance_to_river = distance_transform_edt(~river_all, sampling=(abs(transform.e), abs(transform.a)))
    components = gpd.read_file(CORRIDOR / "D4_design_floodplain_corridor_100yr.gpkg").explode(index_parts=False).reset_index(drop=True)
    component = components[components.intersects(window)].iloc[np.argmax(components[components.intersects(window)].geometry.area)].geometry
    shown = terrain.copy()
    shown[~within] = np.nan
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    raster_extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    ax.imshow(shown, extent=raster_extent, origin="upper", cmap="terrain", alpha=0.43)
    ax.imshow(np.where(floodplain, 1.0, np.nan), extent=raster_extent, origin="upper", cmap="Blues", alpha=0.55, vmin=0, vmax=1)
    ax.contour(np.where(floodplain, distance_to_river, np.nan), levels=[1800.0], extent=raster_extent, origin="upper", colors="#A00000", linewidths=1.25)
    ax.imshow(np.where(river, 1.0, np.nan), extent=raster_extent, origin="upper", cmap="Blues", alpha=1.0, vmin=0, vmax=1)
    gpd.GeoSeries([component.boundary], crs=crs).plot(ax=ax, color="#3594CC", linewidth=1.0)
    gpd.GeoSeries(connector_axes, crs=crs).plot(ax=ax, color="#9D2C00", linewidth=0.9, alpha=0.8, zorder=6)
    cell_m = min(abs(transform.a), abs(transform.e))
    arrow_length_m = max(3.0 * mesh_config.floodplain_along_river_cell_length_m, 2.0 * cell_m)
    arrow_indices = _one_axis_cell_per_bin(axis, accumulation, within, transform, max(8.0 * mesh_config.floodplain_along_river_cell_length_m, 6.0 * cell_m))
    arrow_x: list[float] = []; arrow_y: list[float] = []; arrow_u: list[float] = []; arrow_v: list[float] = []
    for index in arrow_indices:
        row, col = divmod(int(index), terrain.shape[1])
        if not (np.isfinite(tangent_x[row, col]) and np.isfinite(tangent_y[row, col])):
            continue
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        arrow_x.append(x); arrow_y.append(y); arrow_u.append(tangent_x[row, col] * arrow_length_m); arrow_v.append(tangent_y[row, col] * arrow_length_m)
    if arrow_x:
        ax.quiver(arrow_x, arrow_y, arrow_u, arrow_v, angles="xy", scale_units="xy", scale=1, color="#9D2C00", width=0.004, headwidth=3.8, headlength=4.5, zorder=7)
    x0, y0, x1, y1 = window.bounds
    ax.plot([x0 + 250, x0 + 1250], [y0 + 250, y0 + 250], color="0.12", linewidth=2.0)
    ax.text(x0 + 750, y0 + 340, "1 km", ha="center", va="bottom", fontsize=10, fontname="Helvetica")
    ax.text(x0 + 150, y1 - 160, "Setas: tangente dos conectores\nsuavizados (três células)", color="#9D2C00", fontsize=10, fontname="Helvetica", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3})
    ax.text(x1 - 150, y0 + 150, "Contorno vermelho: 1,8 km\ndo rio (próximo ao limite de 2 km)", color="#A00000", fontsize=10, fontname="Helvetica", ha="right", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3})
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "region_04_floodplain_descent_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
    plot_region_04_alignment()
