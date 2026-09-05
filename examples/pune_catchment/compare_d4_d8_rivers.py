"""Compare 2 km² D4 and D8 river extraction on the corrected Pune DEM."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import whitebox
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from rasterio.transform import xy
from shapely.geometry import LineString

from dem_processing.condition_dem import (
    compute_d4_flow_accumulation,
    get_cellsize,
    valid_mask,
    write_raster,
)
from dem_processing.hybrid_mesh import _river_segment_records


ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_DEM = (
    ROOT
    / "examples/pune_catchment/outputs/hydrobathy_20km2_corrected_dem/dem"
    / "DEM_hydraulic_conditioned.tif"
)
D4_ROUTING_DEM = (
    ROOT
    / "examples/pune_catchment/outputs/hydrobathy_20km2_corrected_dem/dem"
    / "DEM_D4_monotonic_routing_surface.tif"
)
D8_SOURCE_DEM = (
    ROOT
    / "examples/pune_catchment/outputs/hydrobathy_20km2_corrected_dem/dem"
    / "DEM_hydrologically_conditioned_pre_bathymetry.tif"
)
DOMAIN = ROOT / "examples/pune_catchment/data/pune_full_valid_domain.geojson"
WINDOWS = ROOT / "examples/pune_catchment/data/pune_floodplain_orientation_windows.geojson"
OUTPUT = ROOT / "examples/pune_catchment/outputs/pune_rivers_2km2_d4_d8"
MIN_AREA_KM2 = 2.0
WIDTH_COEFFICIENT = 2.2695
WIDTH_EXPONENT = 0.4942
DEPTH_COEFFICIENT = 0.1097
DEPTH_EXPONENT = 0.3856


def compute_d8_flow_accumulation(
    dem: np.ndarray, profile: dict, nodata: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return steepest-descent D8 accumulation and flattened receivers."""
    rows, columns = dem.shape
    dx, dy = get_cellsize(profile)
    diagonal = float(np.hypot(dx, dy))
    elevation = dem.astype(np.float64, copy=True)
    valid = valid_mask(elevation, nodata)
    elevation[~valid] = np.nan
    linear = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    receiver = np.full((rows, columns), -1, dtype=np.int64)
    best_slope = np.zeros((rows, columns), dtype=np.float64)

    def update(center: tuple[slice, slice], neighbor: tuple[slice, slice], distance: float) -> None:
        center_elevation = elevation[center]
        neighbor_elevation = elevation[neighbor]
        slope = (center_elevation - neighbor_elevation) / distance
        select = (
            np.isfinite(center_elevation)
            & np.isfinite(neighbor_elevation)
            & (slope > best_slope[center])
        )
        best_slope[center][select] = slope[select]
        receiver[center][select] = linear[neighbor][select]

    update((slice(1, None), slice(None)), (slice(0, -1), slice(None)), dy)
    update((slice(0, -1), slice(None)), (slice(1, None), slice(None)), dy)
    update((slice(None), slice(1, None)), (slice(None), slice(0, -1)), dx)
    update((slice(None), slice(0, -1)), (slice(None), slice(1, None)), dx)
    update((slice(1, None), slice(1, None)), (slice(0, -1), slice(0, -1)), diagonal)
    update((slice(1, None), slice(0, -1)), (slice(0, -1), slice(1, None)), diagonal)
    update((slice(0, -1), slice(1, None)), (slice(1, None), slice(0, -1)), diagonal)
    update((slice(0, -1), slice(0, -1)), (slice(1, None), slice(1, None)), diagonal)

    accumulation = np.zeros(rows * columns, dtype=np.float64)
    flat_valid = valid.ravel()
    accumulation[flat_valid] = 1.0
    flat_elevation = elevation.ravel()
    flat_receiver = receiver.ravel()
    order = np.flatnonzero(flat_valid)
    order = order[np.argsort(flat_elevation[order])[::-1]]
    for index in order:
        target = flat_receiver[index]
        if target >= 0:
            accumulation[target] += accumulation[index]
    return accumulation.reshape(rows, columns), receiver


def receiver_to_d8_direction(receiver: np.ndarray) -> np.ndarray:
    """Encode D8 receivers as 1=N, 2=NE, 3=E, ..., 8=NW."""
    rows, columns = receiver.shape
    index = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    delta = receiver - index
    direction = np.zeros(receiver.shape, dtype=np.uint8)
    for code, offset in enumerate(
        (-columns, -columns + 1, 1, columns + 1, columns, columns - 1, -1, -columns - 1),
        start=1,
    ):
        direction[delta == offset] = code
    return direction


def _raw_records(
    river: np.ndarray, receiver: np.ndarray, transform
) -> gpd.GeoDataFrame:
    columns = river.shape[1]
    lines: list[LineString] = []
    cells: list[int] = []
    for index in np.flatnonzero(river.ravel()):
        target = int(receiver.ravel()[index])
        if target < 0 or not river.ravel()[target]:
            continue
        row, column = divmod(int(index), columns)
        target_row, target_column = divmod(target, columns)
        lines.append(
            LineString(
                (
                    xy(transform, row, column, offset="center"),
                    xy(transform, target_row, target_column, offset="center"),
                )
            )
        )
        cells.append(int(index))
    return gpd.GeoDataFrame({"source_cell": cells}, geometry=lines)


def _smoothed_records(
    river: np.ndarray,
    receiver: np.ndarray,
    accumulation: np.ndarray,
    transform,
    pixel_area_km2: float,
    crs,
) -> gpd.GeoDataFrame:
    records = _river_segment_records(
        river, receiver, transform, np.full(river.shape, 30.0, dtype=np.float64)
    )
    source_cells = np.asarray([item[0] for item in records], dtype=np.int64)
    return gpd.GeoDataFrame(
        {
            "source_cell": source_cells,
            "upstream_area_km2": accumulation.ravel()[source_cells] * pixel_area_km2,
        },
        geometry=[item[1] for item in records],
        crs=crs,
    )


def _line_collection(frame: gpd.GeoDataFrame, color: str, width: float) -> LineCollection:
    return LineCollection(
        [np.asarray(line.coords) for line in frame.geometry],
        colors=color,
        linewidths=width,
        capstyle="round",
        joinstyle="round",
    )


def _draw_background(axis, dem: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    axis.imshow(np.ma.masked_invalid(dem), extent=extent, origin="upper", cmap="Greys", alpha=0.34)
    axis.set_aspect("equal")
    axis.set_axis_off()


def _plot(
    dem: np.ndarray,
    extent: tuple[float, float, float, float],
    d4: gpd.GeoDataFrame,
    d8: gpd.GeoDataFrame,
    domain,
    windows: gpd.GeoDataFrame,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for axis in axes:
        _draw_background(axis, dem, extent)
        gpd.GeoSeries([domain.boundary], crs=d4.crs).plot(
            ax=axis, color="#262626", linewidth=0.7
        )
    axes[0].add_collection(_line_collection(d4, "#E25759", 0.45))
    axes[1].add_collection(_line_collection(d8, "#0B81A2", 0.45))
    axes[2].add_collection(_line_collection(d4, "#E25759", 0.55))
    axes[2].add_collection(_line_collection(d8, "#0B81A2", 0.35))
    for axis, title in zip(
        axes,
        ("D4 rivers", "D8 rivers", "D4 and D8 overlay"),
        strict=True,
    ):
        axis.set_title(f"{title} — upstream area ≥ 2 km²", fontweight="normal")
    axes[2].legend(
        handles=(
            Line2D([0], [0], color="#E25759", lw=2, label="D4"),
            Line2D([0], [0], color="#0B81A2", lw=2, label="D8"),
        ),
        loc="lower left",
        frameon=True,
    )
    _save_figure(figure, "pune_rivers_2km2_d4_d8_full")

    for index, item in windows.iterrows():
        figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
        _draw_background(axis, dem, extent)
        axis.add_collection(_line_collection(d4, "#E25759", 1.3))
        axis.add_collection(_line_collection(d8, "#0B81A2", 0.9))
        xmin, ymin, xmax, ymax = item.geometry.bounds
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_title(
            f"D4–D8 river comparison — region {index + 1}", fontweight="normal"
        )
        axis.legend(
            handles=(
                Line2D([0], [0], color="#E25759", lw=2, label="D4"),
                Line2D([0], [0], color="#0B81A2", lw=2, label="D8"),
            ),
            loc="lower left",
            frameon=True,
        )
        _save_figure(figure, f"pune_rivers_2km2_d4_d8_region_{index + 1:02d}")


def _save_figure(figure, stem: str) -> None:
    for folder, suffix in (("png", "png"), ("pdf", "pdf"), ("svg", "svg")):
        path = OUTPUT / "figures" / folder / f"{stem}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _self_check() -> None:
    dem = np.asarray([[4.0, 3.0, 2.0], [3.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
    profile = {"transform": rasterio.Affine.scale(30.0, -30.0)}
    _, receiver = compute_d8_flow_accumulation(dem, profile, nodata=None)
    assert receiver[1, 1] == 8


def main() -> None:
    _self_check()
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 10})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    d8_routing_path = OUTPUT / "rasters/DEM_D8_filled_routing_surface.tif"
    if not d8_routing_path.exists():
        d8_routing_path.parent.mkdir(parents=True, exist_ok=True)
        tools = whitebox.WhiteboxTools()
        tools.set_verbose_mode(False)
        tools.fill_depressions(
            str(D8_SOURCE_DEM),
            str(d8_routing_path),
            fix_flats=True,
            flat_increment=0.01,
        )
    with rasterio.open(D4_ROUTING_DEM) as source:
        d4_routing_dem = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        profile = source.profile.copy()
        transform, crs = source.transform, source.crs
        west, south, east, north = source.bounds
    with rasterio.open(d8_routing_path) as source:
        d8_routing_dem = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        if source.shape != d4_routing_dem.shape or source.transform != transform:
            raise ValueError("D4 and D8 routing DEM grids do not match.")
    with rasterio.open(BACKGROUND_DEM) as source:
        dem = source.read(1, masked=True).filled(np.nan).astype(np.float64)
        if source.shape != d4_routing_dem.shape or source.transform != transform:
            raise ValueError("Background and routing DEM grids do not match.")
    extent = (west, east, south, north)
    pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    threshold_cells = MIN_AREA_KM2 / pixel_area_km2
    d4_accumulation, d4_receiver = compute_d4_flow_accumulation(
        d4_routing_dem, profile, nodata=np.nan
    )
    d8_accumulation, d8_receiver = compute_d8_flow_accumulation(
        d8_routing_dem, profile, nodata=np.nan
    )
    valid = np.isfinite(d4_routing_dem) & np.isfinite(d8_routing_dem)
    d4_river = valid & (d4_accumulation >= threshold_cells)
    d8_river = valid & (d8_accumulation >= threshold_cells)

    active_boundary = valid & (
        ~np.roll(valid, 1, axis=0)
        | ~np.roll(valid, -1, axis=0)
        | ~np.roll(valid, 1, axis=1)
        | ~np.roll(valid, -1, axis=1)
    )
    active_boundary[[0, -1], :] |= valid[[0, -1], :]
    active_boundary[:, [0, -1]] |= valid[:, [0, -1]]
    d4_internal_terminals = d4_river & (d4_receiver < 0) & ~active_boundary
    d8_internal_terminals = d8_river & (d8_receiver < 0) & ~active_boundary
    if np.any(d4_internal_terminals) or np.any(d8_internal_terminals):
        raise RuntimeError(
            "Filled-routing continuity check failed: "
            f"D4={int(d4_internal_terminals.sum())}, "
            f"D8={int(d8_internal_terminals.sum())} internal river terminals."
        )

    raster_dir = OUTPUT / "rasters"
    write_raster(raster_dir / "D4_river_mask_2km2.tif", d4_river, profile, nodata=0, dtype="uint8")
    write_raster(raster_dir / "D8_river_mask_2km2.tif", d8_river, profile, nodata=0, dtype="uint8")
    write_raster(
        raster_dir / "D8_flow_direction.tif",
        receiver_to_d8_direction(d8_receiver),
        profile,
        nodata=0,
        dtype="uint8",
    )
    write_raster(
        raster_dir / "D4_upstream_area_km2.tif",
        d4_accumulation * pixel_area_km2,
        profile,
    )
    write_raster(
        raster_dir / "D8_upstream_area_km2.tif",
        d8_accumulation * pixel_area_km2,
        profile,
    )
    d8_area_km2 = d8_accumulation * pixel_area_km2
    d8_width = np.where(
        d8_river, WIDTH_COEFFICIENT * np.power(d8_area_km2, WIDTH_EXPONENT), np.nan
    )
    d8_depth = np.where(
        d8_river, DEPTH_COEFFICIENT * np.power(d8_area_km2, DEPTH_EXPONENT), np.nan
    )
    d8_bed = np.where(d8_river, d8_routing_dem - d8_depth, np.nan)
    write_raster(raster_dir / "D8_river_width_m.tif", d8_width, profile)
    write_raster(raster_dir / "D8_river_depth_m.tif", d8_depth, profile)
    write_raster(raster_dir / "D8_river_bed_elevation_m.tif", d8_bed, profile)

    d4_raw = _raw_records(d4_river, d4_receiver, transform).set_crs(crs)
    d8_raw = _raw_records(d8_river, d8_receiver, transform).set_crs(crs)
    d4_smooth = _smoothed_records(
        d4_river, d4_receiver, d4_accumulation, transform, pixel_area_km2, crs
    )
    d8_smooth = _smoothed_records(
        d8_river, d8_receiver, d8_accumulation, transform, pixel_area_km2, crs
    )
    vectors = OUTPUT / "pune_rivers_2km2_d4_d8.gpkg"
    if vectors.exists():
        vectors.unlink()
    for index, (name, frame) in enumerate(
        (
            ("d4_raw", d4_raw),
            ("d8_raw", d8_raw),
            ("d4_smoothed_for_mesh", d4_smooth),
            ("d8_smoothed_for_mesh", d8_smooth),
        )
    ):
        frame.to_file(vectors, layer=name, driver="GPKG", mode="w" if index == 0 else "a")

    domain = gpd.read_file(DOMAIN).to_crs(crs).geometry.union_all()
    windows = gpd.read_file(WINDOWS).to_crs(crs)
    _plot(dem, extent, d4_smooth, d8_smooth, domain, windows)
    overlap = d4_river & d8_river
    summary = (
        "method,river_cells,raw_links,smoothed_segments,total_smoothed_length_km\n"
        f"D4,{d4_river.sum()},{len(d4_raw)},{len(d4_smooth)},{d4_smooth.length.sum()/1000:.6f}\n"
        f"D8,{d8_river.sum()},{len(d8_raw)},{len(d8_smooth)},{d8_smooth.length.sum()/1000:.6f}\n"
        f"shared_cells,{overlap.sum()},,,\n"
        f"D4_internal_terminals,{d4_internal_terminals.sum()},,,\n"
        f"D8_internal_terminals,{d8_internal_terminals.sum()},,,\n"
    )
    (OUTPUT / "river_network_summary.csv").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(vectors)


if __name__ == "__main__":
    main()
