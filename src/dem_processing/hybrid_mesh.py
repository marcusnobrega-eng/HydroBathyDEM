"""Feature-aware hybrid quadrilateral/Voronoi mesh generation.

Regular seed lattices give four-sided Voronoi cells in coherent background
patches.  Feature and transition seeds are left unstructured, yielding one
conforming polygon mesh without user-drawn refinement regions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import netCDF4
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely import (
    MultiPoint, STRtree, box, centroid, get_coordinates,
    get_num_coordinates, get_x, get_y, intersection, point_on_surface,
    unary_union, voronoi_polygons,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

from .condition_dem import compute_d4_flow_accumulation
from .config import load_config_file


@dataclass(frozen=True)
class HybridMeshConfig:
    """All mesh controls; no catchment-specific rule is encoded here."""

    dem: Path
    out_dir: Path
    impervious: Path | None = None
    population: Path | None = None
    river_mask: Path | None = None
    river_direction: Path | None = None
    river_width: Path | None = None
    river_depth: Path | None = None
    river_bank_height: Path | None = None
    floodplain_mask: Path | None = None
    floodplain_vector: Path | None = None
    floodplain_direction: Path | None = None
    floodplain_axis: Path | None = None
    river_network: Path | None = None
    diagnostic_window: Path | None = None
    diagnostic_windows: Path | None = None
    river_width_field: str = "width_m"
    domain_vector: Path | None = None
    background_width_m: float = 120.0
    urban_width_m: float = 45.0
    # Feature-first spacing.  ``along`` follows the directed mapped river;
    # ``cross`` is normal to it.  These are intentionally distinct from the
    # minimum hydraulic width: a 30 m long cell may be 90 m wide.
    river_along_river_cell_length_m: float = 30.0
    river_cross_river_target_width_m: float = 90.0
    floodplain_target_width_m: float = 60.0
    floodplain_align_to_flow: bool = False
    # Legacy directional floodplain controls.  They are used only when
    # ``floodplain_align_to_flow`` is explicitly enabled.
    floodplain_along_river_cell_length_m: float = 40.0
    floodplain_cross_river_target_width_m: float = 90.0
    minimum_hydraulic_width_m: float = 30.0
    maximum_adjacent_size_ratio: float = 2.0
    enforce_feature_boundaries: bool = True
    impervious_threshold_percent: float = 1.0
    population_threshold_per_km2: float = 1000.0
    urban_buffer_m: float = 300.0
    floodplain_hand_stage_m: float = 10.0
    minimum_upstream_area_km2: float = 1.0
    river_source: str = "derive_d4"
    river_preferred_cells_across: int = 3
    unresolved_policy: str = "neal_subgrid"
    minimum_face_length_factor: float = 0.25
    minimum_center_distance_factor: float = 0.5
    quality_repair_max_iterations: int = 6
    # Diagnostic option: write the globally conforming Step 5 Voronoi only.
    # It is deliberately not a production mesh because hard feature-boundary
    # splitting and all topology/quality gates have not yet been applied.
    stop_after_step: int | None = None
    max_seed_count: int = 100_000
    random_seed: int = 7

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "HybridMeshConfig":
        """Read either the compact legacy keys or the grouped case configuration."""
        values = dict(values)
        mesh_mode = values.pop("mesh_mode", "voronoi_fv")
        if mesh_mode != "voronoi_fv":
            raise ValueError(f"Hybrid mesh generator requires mesh_mode='voronoi_fv', got {mesh_mode!r}.")
        grouped = {
            "inputs": {},
            "resolution": {},
            "rivers": {},
            "urban": {},
            "quality": {},
        }
        for group in grouped:
            group_values = values.pop(group, {})
            if group_values:
                if not isinstance(group_values, dict):
                    raise ValueError(f"{group} must be a configuration object.")
                grouped[group].update(group_values)
        values = {
            **grouped["inputs"],
            **grouped["resolution"],
            **grouped["rivers"],
            **grouped["urban"],
            **grouped["quality"],
            **values,
        }
        aliases = {
            "dem_path": "dem",
            "output_dir": "out_dir",
            "urban_impervious_path": "impervious",
            "population_path": "population",
            "river_mask_path": "river_mask",
            "river_direction_path": "river_direction",
            "river_width_path": "river_width",
            "river_depth_path": "river_depth",
            "river_bank_height_path": "river_bank_height",
            "floodplain_mask_path": "floodplain_mask",
            "floodplain_vector_path": "floodplain_vector",
            "floodplain_direction_path": "floodplain_direction",
            "floodplain_axis_path": "floodplain_axis",
            "source": "river_source",
            "initiation_upstream_area_km2": "minimum_upstream_area_km2",
            "minimum_cell_width_m": "minimum_hydraulic_width_m",
            "buffer_m": "urban_buffer_m",
            # Compatibility with the first rejected prototype.  New case
            # files should use the explicit along/cross names above.
            "hand_threshold_m": "floodplain_hand_stage_m",
            "floodplain_width_m": "floodplain_target_width_m",
            "floodplain_along_width_m": "floodplain_along_river_cell_length_m",
            "floodplain_cross_width_m": "floodplain_cross_river_target_width_m",
        }
        values = {aliases.get(key, key): value for key, value in values.items()}
        for key in ("dem", "out_dir", "impervious", "population", "river_mask", "river_direction", "river_width", "river_depth", "river_bank_height", "floodplain_mask", "floodplain_vector", "floodplain_direction", "floodplain_axis", "river_network", "domain_vector", "diagnostic_window", "diagnostic_windows"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        if config.unresolved_policy not in {"neal_subgrid", "none"}:
            raise ValueError("rivers.unresolved_policy must be 'neal_subgrid' or 'none'.")
        if config.river_source not in {"derive_d4", "hydrobathydem_d4"}:
            raise ValueError("rivers.source must be 'derive_d4' or 'hydrobathydem_d4'.")
        if config.stop_after_step not in {None, 5, 6}:
            raise ValueError("quality.stop_after_step may be omitted or set to 5 or 6.")
        if config.minimum_upstream_area_km2 <= 0:
            raise ValueError("rivers.initiation_upstream_area_km2 must be positive.")
        if config.minimum_face_length_factor <= 0 or config.minimum_center_distance_factor <= 0:
            raise ValueError("quality factors must be positive.")
        if min(
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            config.floodplain_target_width_m,
            config.floodplain_along_river_cell_length_m,
            config.floodplain_cross_river_target_width_m,
        ) <= 0:
            raise ValueError("all river and floodplain along/cross targets must be positive.")
        return config


def _aligned(path: Path | None, reference: rasterio.DatasetReader, default: float = 0.0) -> np.ndarray:
    if path is None:
        return np.full(reference.shape, default, dtype=np.float64)
    with rasterio.open(path) as source:
        if source.shape != reference.shape or source.transform != reference.transform:
            raise ValueError(f"Mesh input {path} is not aligned with the DEM grid.")
        # Cast before filling: NumPy cannot represent a NaN fill value in a
        # masked uint8/int8 categorical layer (e.g. ESA LULC or soil class).
        return source.read(1, masked=True).astype(np.float64).filled(np.nan)


def connected_hand(dem: np.ndarray, receiver: np.ndarray, river: np.ndarray) -> np.ndarray:
    """Return HAND only for cells whose D4 path reaches the supplied river mask."""
    flat_dem, flat_receiver, flat_river = dem.ravel(), receiver.ravel(), river.ravel()
    hand = np.full(flat_dem.size, np.nan, dtype=np.float64)
    hand[flat_river] = 0.0
    # A D4 receiver is strictly lower, so one low-to-high pass is sufficient.
    # This keeps continental rasters O(N log N), rather than retracing every path.
    valid = np.flatnonzero(np.isfinite(flat_dem))
    for item in valid[np.argsort(flat_dem[valid])]:
        downstream = int(flat_receiver[item])
        if downstream >= 0 and np.isfinite(hand[downstream]):
            hand[item] = max(flat_dem[item] - (flat_dem[downstream] - hand[downstream]), 0.0)
    return hand.reshape(dem.shape)


def connected_hand_with_reach(
    dem: np.ndarray, receiver: np.ndarray, river: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return HAND and the first mapped-river cell reached by every D4 path.

    A single D4 network can contain many nearby channels.  A binary HAND mask
    loses that provenance and was the reason the old generator turned large,
    unrelated valley fragments into one jagged ribbon.  The reach label is the
    flattened index of the first river cell encountered downstream; ``-1``
    means the cell has no mapped-river connection.
    """
    flat_dem = dem.ravel()
    flat_receiver = receiver.ravel()
    flat_river = river.ravel()
    hand = np.full(flat_dem.size, np.nan, dtype=np.float64)
    reach = np.full(flat_dem.size, -1, dtype=np.int64)
    base = np.full(flat_dem.size, np.nan, dtype=np.float64)
    river_index = np.flatnonzero(flat_river)
    hand[river_index] = 0.0
    base[river_index] = flat_dem[river_index]
    reach[river_index] = river_index
    valid = np.flatnonzero(np.isfinite(flat_dem))
    for item in valid[np.argsort(flat_dem[valid])]:
        downstream = int(flat_receiver[item])
        if reach[item] >= 0 or downstream < 0 or reach[downstream] < 0:
            continue
        reach[item] = reach[downstream]
        base[item] = base[downstream]
        hand[item] = max(flat_dem[item] - base[item], 0.0)
    return hand.reshape(dem.shape), reach.reshape(dem.shape)


def receiver_from_d4_direction(direction: np.ndarray) -> np.ndarray:
    """Convert HydroBathyDEM's D4 direction codes to flattened receivers."""
    rows, cols = direction.shape
    receiver = np.full(direction.shape, -1, dtype=np.int64)
    index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    receiver[1:, :][direction[1:, :] == 1] = index[:-1, :][direction[1:, :] == 1]
    receiver[:-1, :][direction[:-1, :] == 3] = index[1:, :][direction[:-1, :] == 3]
    receiver[:, :-1][direction[:, :-1] == 2] = index[:, 1:][direction[:, :-1] == 2]
    receiver[:, 1:][direction[:, 1:] == 4] = index[:, :-1][direction[:, 1:] == 4]
    return receiver


def _smooth_size_field(target: np.ndarray, valid: np.ndarray, ratio: float) -> np.ndarray:
    """Propagate fine targets outward until every D4 neighbour obeys the ratio cap."""
    result = target.copy()
    for _ in range(result.size):
        previous = result.copy()
        for axis in (0, 1):
            a = np.take(result, range(result.shape[axis] - 1), axis=axis)
            b = np.take(result, range(1, result.shape[axis]), axis=axis)
            a_new, b_new = np.minimum(a, ratio * b), np.minimum(b, ratio * a)
            index_a = [slice(None)] * 2
            index_b = [slice(None)] * 2
            index_a[axis], index_b[axis] = slice(0, -1), slice(1, None)
            result[tuple(index_a)] = np.where(valid[tuple(index_a)], a_new, result[tuple(index_a)])
            result[tuple(index_b)] = np.where(valid[tuple(index_b)], b_new, result[tuple(index_b)])
        if np.allclose(previous, result):
            break
    return result


def _lattice(bounds: tuple[float, float, float, float], spacing: float, mask: np.ndarray, transform: Any) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    xs = np.arange(xmin + spacing / 2.0, xmax, spacing)
    ys = np.arange(ymin + spacing / 2.0, ymax, spacing)
    xx, yy = np.meshgrid(xs, ys)
    row, col = rasterio.transform.rowcol(transform, xx.ravel(), yy.ravel())
    row, col = np.asarray(row), np.asarray(col)
    inside = (row >= 0) & (row < mask.shape[0]) & (col >= 0) & (col < mask.shape[1])
    keep = np.zeros(inside.size, dtype=bool)
    keep[inside] = mask[row[inside], col[inside]]
    return np.column_stack((xx.ravel()[keep], yy.ravel()[keep]))


def _ribbon_reservation(river: np.ndarray, transform: Any, half_width_m: float) -> np.ndarray:
    """Reserve a buffer so a river ribbon replaces, rather than collides with, lattice seeds."""
    return distance_transform_edt(~river, sampling=(abs(transform.e), abs(transform.a))) <= half_width_m


def _drop_near(points: np.ndarray, blockers: np.ndarray, minimum_distance_m: float) -> np.ndarray:
    """Keep candidate seeds far enough from already-prioritized seeds."""
    if not len(points) or not len(blockers):
        return points
    keep = cKDTree(blockers).query(points, distance_upper_bound=minimum_distance_m)[0] >= minimum_distance_m
    return points[keep]


def _minimum_separated(points: np.ndarray, minimum_distance_m: float) -> np.ndarray:
    """Greedy spatial-hash thinning; avoids all-pairs work for continental river networks."""
    if not len(points):
        return points
    bins: dict[tuple[int, int], list[np.ndarray]] = {}
    accepted: list[np.ndarray] = []
    for point in points:
        key = tuple(np.floor(point / minimum_distance_m).astype(int))
        nearby = [other for row in range(key[0] - 1, key[0] + 2) for col in range(key[1] - 1, key[1] + 2) for other in bins.get((row, col), [])]
        if not nearby or np.all(np.hypot(np.asarray(nearby)[:, 0] - point[0], np.asarray(nearby)[:, 1] - point[1]) >= minimum_distance_m):
            accepted.append(point)
            bins.setdefault(key, []).append(point)
    return np.asarray(accepted, dtype=np.float64)


def _flow_aligned_ribbon_seeds(
    mask: np.ndarray, receiver: np.ndarray, transform: Any, cross_width_m: np.ndarray,
    along_width_m: np.ndarray, minimum_distance_m: float,
) -> np.ndarray:
    """Place a protected, flow-aligned seed ribbon over a directed feature mask.

    The caller supplies a completed feature region (a river core or its HAND
    floodplain), before any background sites are considered.  Cross-stream
    rows determine the cell width and longitudinal thinning determines length.
    This is deliberately a feature-first construction: later rural/urban sites
    may meet the ribbon only through the graded transition field.
    """
    rows, cols = np.where(mask)
    ncols, nitems = mask.shape[1], mask.size
    centres: list[tuple[float, float, float, float, float]] = []
    for row, col in zip(rows, cols, strict=True):
        start = row * ncols + col
        downstream = int(receiver[row, col])
        if downstream < 0:
            continue
        terminal = downstream
        for _ in range(2):  # three D4 links smooth staircase directions without changing topology
            candidate = int(receiver.flat[terminal])
            if candidate < 0 or candidate >= nitems or not mask.flat[candidate]:
                break
            terminal = candidate
        trow, tcol = divmod(terminal, ncols)
        x0, y0 = xy(transform, row, col, offset="center")
        x1, y1 = xy(transform, trow, tcol, offset="center")
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length > 0:
            centres.append((
                x0, y0, dx / length, dy / length,
                float(cross_width_m[row, col]), float(along_width_m[row, col]),
            ))
    if not centres:
        return np.empty((0, 2), dtype=np.float64)

    centre_array = np.asarray(centres, dtype=np.float64)
    # A single conservative separation avoids coincident D4 sites at bends and
    # confluences.  Individual seed rows remain no closer than the hydraulic
    # floor after the side rows are added below.
    interval = max(minimum_distance_m, float(np.nanmin(centre_array[:, 5])))
    selected_points = _minimum_separated(centre_array[:, :2], interval)
    selected_keys = {tuple(point) for point in selected_points}
    selected = [candidate for candidate in centres if candidate[:2] in selected_keys]
    rows_of_seeds: list[tuple[float, float]] = []
    for x0, y0, tx, ty, cross, _ in selected:
        nx, ny = -ty, tx
        # Three rows make intentionally anisotropic floodplain cells.  The
        # cross-stream spacing, not the longitudinal spacing, is the local
        # hydraulic width relevant to the surface CFL condition.
        rows_of_seeds.extend((
            (x0 - cross * nx, y0 - cross * ny),
            (x0, y0),
            (x0 + cross * nx, y0 + cross * ny),
        ))
    return _minimum_separated(np.asarray(rows_of_seeds, dtype=np.float64), minimum_distance_m)


def _river_seeds(
    river: np.ndarray, receiver: np.ndarray, transform: Any, cross_width_m: np.ndarray, minimum_distance_m: float,
) -> np.ndarray:
    """Backward-compatible river-core wrapper for the protected ribbon builder."""
    return _flow_aligned_ribbon_seeds(
        river, receiver, transform, cross_width_m,
        np.maximum(2.0 * cross_width_m, minimum_distance_m), minimum_distance_m,
    )


def _mask_geometry(mask: np.ndarray, transform: Any) -> Polygon:
    """Polygonise a raster feature without changing its selected cells."""
    pieces = [
        Polygon(geometry["coordinates"][0])
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform)
        if value
    ]
    geometry = unary_union(pieces) if pieces else Polygon()
    return geometry if geometry.geom_type == "Polygon" else geometry


def _river_segment_records(
    river: np.ndarray, receiver: np.ndarray, transform: Any, width: np.ndarray,
) -> list[tuple[int, LineString, float, tuple[float, float]]]:
    """Build smoothed, directed reaches from an orthogonal D4 river network.

    D4 is retained for hydrologic connectivity, not used as the visible river
    geometry.  Each between-junction reach is smoothed once with a
    corner-cutting pass (fixed endpoints), then resampled with its original
    mapped widths.  This removes artificial 90-degree mesh bends while keeping
    every D4 link represented by a directed physical channel.
    """
    ncols = river.shape[1]
    river_indices = np.flatnonzero(river.ravel())
    downstream: dict[int, int] = {}
    indegree = {int(item): 0 for item in river_indices}
    for item in river_indices:
        target = int(receiver.ravel()[item])
        if target in indegree:
            downstream[int(item)] = target
            indegree[target] += 1

    def smooth(points: np.ndarray) -> np.ndarray:
        current = points
        # One fixed-endpoint Chaikin pass removes the D4 right angle without
        # creating a sub-30 m polyline that would itself control the mesh.
        for _ in range(1 if len(current) > 2 else 0):
            result = [current[0]]
            for left, right in zip(current[:-1], current[1:], strict=True):
                result.append(0.75 * left + 0.25 * right)
                result.append(0.25 * left + 0.75 * right)
            result.append(current[-1])
            current = np.asarray(result, dtype=np.float64)
        return current

    visited_links: set[tuple[int, int]] = set()
    records: list[tuple[int, LineString, float, tuple[float, float]]] = []
    starts = [item for item in river_indices if indegree[int(item)] != 1]
    for initial in starts:
        current = int(initial)
        path = [current]
        while current in downstream:
            target = downstream[current]
            link = (current, target)
            if link in visited_links:
                break
            visited_links.add(link)
            path.append(target)
            current = target
            if indegree[current] != 1:
                break
        if len(path) < 2:
            continue
        centres = np.asarray([xy(transform, *divmod(item, ncols), offset="center") for item in path], dtype=np.float64)
        smoothed = smooth(centres)
        # The smoothing creates two short segments per source D4 link.  Assign
        # each the corresponding mapped bankfull width, never a mesh width.
        for segment_id, (left, right) in enumerate(zip(smoothed[:-1], smoothed[1:], strict=True)):
            source_index = path[min(len(path) - 2, segment_id // 2)]
            row, col = divmod(source_index, ncols)
            mapped_width = float(width[row, col]) if np.isfinite(width[row, col]) else 0.0
            dx, dy = right - left
            length = float(np.hypot(dx, dy))
            if mapped_width > 0 and length > 0:
                records.append((source_index, LineString((left, right)), mapped_width, (dx / length, dy / length)))

    # A simple one-link reach at a cut domain edge can be missed above.  Add it
    # unchanged rather than silently dropping a mapped river.
    for start, target in downstream.items():
        if (start, target) in visited_links:
            continue
        row, col = divmod(start, ncols)
        drow, dcol = divmod(target, ncols)
        x0, y0 = xy(transform, row, col, offset="center")
        x1, y1 = xy(transform, drow, dcol, offset="center")
        length = float(np.hypot(x1 - x0, y1 - y0))
        mapped_width = float(width[row, col]) if np.isfinite(width[row, col]) else 0.0
        if mapped_width > 0 and length > 0:
            records.append((start, LineString(((x0, y0), (x1, y1))), mapped_width, ((x1 - x0) / length, (y1 - y0) / length)))
    return records


def _line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _vector_river_records(
    network_path: Path, width_field: str, crs: Any, domain, dem: np.ndarray, transform: Any,
) -> list[tuple[int, LineString, float, tuple[float, float]]]:
    """Read the mapped river network and orient each reach down the DEM."""
    network = gpd.read_file(network_path)
    if network.empty:
        raise ValueError(f"Mapped river network is empty: {network_path}")
    network = network.to_crs(crs)
    if width_field not in network.columns:
        raise ValueError(f"Mapped river network {network_path} lacks width field {width_field!r}.")
    records: list[tuple[int, LineString, float, tuple[float, float]]] = []
    for fid, item in network.iterrows():
        geometry = item.geometry.intersection(domain)
        for line in _line_parts(geometry):
            if line.length <= 1.0:
                continue
            coordinates = list(line.coords)
            start_row, start_col = rasterio.transform.rowcol(transform, coordinates[0][0], coordinates[0][1])
            end_row, end_col = rasterio.transform.rowcol(transform, coordinates[-1][0], coordinates[-1][1])
            start_elevation = dem[np.clip(start_row, 0, dem.shape[0] - 1), np.clip(start_col, 0, dem.shape[1] - 1)]
            end_elevation = dem[np.clip(end_row, 0, dem.shape[0] - 1), np.clip(end_col, 0, dem.shape[1] - 1)]
            if np.isfinite(start_elevation) and np.isfinite(end_elevation) and end_elevation > start_elevation:
                line = LineString(coordinates[::-1])
            width = float(item[width_field])
            if not np.isfinite(width) or width <= 0:
                continue
            start = np.asarray(line.coords[0], dtype=np.float64)
            end = np.asarray(line.coords[-1], dtype=np.float64)
            vector = end - start
            length = float(np.hypot(*vector))
            if length > 0:
                records.append((int(fid), line, width, (vector[0] / length, vector[1] / length)))
    if not records:
        raise ValueError(f"No mapped river reaches overlap the mesh domain: {network_path}")
    return records


def _vector_reach_orientations(
    reach: np.ndarray, transform: Any, records: list[tuple[int, LineString, float, tuple[float, float]]],
) -> dict[int, tuple[float, float]]:
    """Map D4 HAND labels to the local tangent of the nearest mapped reach."""
    labels = np.unique(reach[reach >= 0])
    if not len(labels):
        return {}
    lines = [record[1] for record in records]
    tree = gpd.GeoSeries(lines).sindex
    orientations: dict[int, tuple[float, float]] = {}
    ncols = reach.shape[1]
    for label in labels:
        row, col = divmod(int(label), ncols)
        point = Point(xy(transform, row, col, offset="center"))
        nearest_index = list(tree.nearest(point, return_all=False))[1][0]
        line = lines[int(nearest_index)]
        distance = line.project(point)
        before = line.interpolate(max(0.0, distance - 1.0))
        after = line.interpolate(min(line.length, distance + 1.0))
        dx, dy = after.x - before.x, after.y - before.y
        length = float(np.hypot(dx, dy))
        if length > 0:
            orientations[int(label)] = (dx / length, dy / length)
    return orientations


def _physical_river_geometry(records: list[tuple[int, LineString, float, tuple[float, float]]]):
    """Union actual mapped bankfull footprints; never widen a narrow river."""
    if not records:
        return Polygon()
    return unary_union([line.buffer(width / 2.0, cap_style="flat", join_style="round") for _, line, width, _ in records])


def _river_core_seeds(
    records: list[tuple[int, LineString, float, tuple[float, float]]],
    along_m: float, cross_target_m: float, minimum_width_m: float,
) -> np.ndarray:
    """Place physical-width cells across each resolved mapped river segment."""
    candidates: list[tuple[float, float]] = []
    for _, line, width, tangent in records:
        if width < minimum_width_m:
            continue  # retained as Neal subgrid, never widened for the mesh
        tx, ty = tangent
        nx, ny = -ty, tx
        n_along = max(1, int(np.ceil(line.length / along_m)))
        n_cross = max(1, int(np.ceil(width / cross_target_m)))
        actual_cross = width / n_cross
        for step in range(n_along):
            centre = line.interpolate((step + 0.5) * line.length / n_along)
            for band in range(n_cross):
                offset = -0.5 * width + (band + 0.5) * actual_cross
                candidates.append((centre.x + offset * nx, centre.y + offset * ny))
    return np.unique(np.asarray(candidates, dtype=np.float64), axis=0) if candidates else np.empty((0, 2), dtype=np.float64)


def _feature_lattice_seeds(
    mask: np.ndarray,
    reach: np.ndarray,
    reach_orientation: dict[int, tuple[float, float]],
    transform: Any,
    along_m: float,
    cross_m: float,
    geometry,
) -> np.ndarray:
    """Select an anisotropic, reach-aligned lattice inside one feature zone.

    Cells are not generated from a generic point cloud.  Each candidate is
    snapped to a lattice in local streamwise/cross-stream coordinates of the
    mapped river reached by its D4 drainage path.  This makes the river and
    HAND corridor the first, locked part of the mesh rather than a later
    refinement mask.
    """
    rows, cols = np.where(mask & (reach >= 0))
    if not len(rows):
        return np.empty((0, 2), dtype=np.float64)
    xs, ys = xy(transform, rows, cols, offset="center")
    points = np.column_stack((xs, ys)).astype(np.float64)
    labels = reach[rows, cols]
    selected: list[np.ndarray] = []
    for label in np.unique(labels):
        tangent = reach_orientation.get(int(label))
        if tangent is None:
            continue
        local = points[labels == label]
        origin_row, origin_col = divmod(int(label), mask.shape[1])
        ox, oy = xy(transform, origin_row, origin_col, offset="center")
        tx, ty = tangent
        nx, ny = -ty, tx
        stream = (local[:, 0] - ox) * tx + (local[:, 1] - oy) * ty
        cross = (local[:, 0] - ox) * nx + (local[:, 1] - oy) * ny
        stream_bin = np.rint(stream / along_m).astype(np.int64)
        cross_bin = np.rint(cross / cross_m).astype(np.int64)
        best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
        for point, s, c, sb, cb in zip(local, stream, cross, stream_bin, cross_bin, strict=True):
            key = (int(sb), int(cb))
            distance = abs(s - sb * along_m) + abs(c - cb * cross_m)
            if key not in best or distance < best[key][0]:
                best[key] = (float(distance), point)
        selected.extend(value[1] for value in best.values())
    if not selected:
        return np.empty((0, 2), dtype=np.float64)
    candidates = np.unique(np.asarray(selected, dtype=np.float64), axis=0)
    return np.asarray([point for point in candidates if geometry.covers(Point(point))], dtype=np.float64)


def _axis_connector_paths(axis: np.ndarray, receiver: np.ndarray) -> list[list[int]]:
    """Return directed, junction-to-junction paths through a thin connector raster."""
    active = axis.ravel().astype(bool)
    links: dict[int, int] = {}
    indegree = np.zeros(active.size, dtype=np.int32)
    for node in np.flatnonzero(active):
        target = int(receiver.ravel()[node])
        if 0 <= target < active.size:
            links[int(node)] = target
            if active[target]:
                indegree[target] += 1
    starts = [int(node) for node in np.flatnonzero(active) if indegree[node] != 1]
    visited: set[tuple[int, int]] = set()
    paths: list[list[int]] = []

    def trace(start: int) -> None:
        path = [start]
        current = start
        while current in links:
            target = links[current]
            edge = (current, target)
            if edge in visited:
                break
            visited.add(edge)
            path.append(target)
            if not active[target] or indegree[target] != 1:
                break
            current = target
        if len(path) >= 2:
            paths.append(path)

    for start in starts:
        trace(start)
    for start in links:
        if (start, links[start]) not in visited:
            trace(start)
    return paths


def _line_tangent(line: LineString, point: Point, span_m: float) -> tuple[float, float] | None:
    """Centered tangent of a smoothed directed connector line."""
    distance = line.project(point)
    before = line.interpolate(max(0.0, distance - span_m))
    after = line.interpolate(min(line.length, distance + span_m))
    dx, dy = after.x - before.x, after.y - before.y
    length = float(np.hypot(dx, dy))
    return (dx / length, dy / length) if length > 0 else None


def _smoothed_floodplain_connector_tangents(
    axis: np.ndarray, receiver: np.ndarray, transform: Any, along_cell_length_m: float,
) -> tuple[np.ndarray, np.ndarray, list[LineString]]:
    """Orient floodplain sites from smooth, directed accumulation connectors.

    The smoothing scale is derived from the requested streamwise cell length:
    a centered three-cell tangent suppresses D4 stair steps without imposing a
    catchment-specific distance such as the former 240 m look-ahead.
    """
    tangent_x = np.full(axis.shape, np.nan, dtype=np.float64)
    tangent_y = np.full(axis.shape, np.nan, dtype=np.float64)
    cell_m = min(abs(transform.a), abs(transform.e))
    tangent_span_m = max(1.5 * along_cell_length_m, cell_m)
    simplify_tolerance_m = tangent_span_m
    paths = _axis_connector_paths(axis, receiver)
    connector_lines: list[LineString] = []
    for path in paths:
        rows, cols = np.divmod(np.asarray(path), axis.shape[1])
        xs, ys = xy(transform, rows, cols, offset="center")
        raw_line = LineString(np.column_stack((xs, ys)))
        line = raw_line.simplify(simplify_tolerance_m, preserve_topology=False)
        if not isinstance(line, LineString) or line.length <= 0:
            line = raw_line
        if line.length <= 0:
            continue
        connector_lines.append(line)
        for path_index, node in enumerate(path):
            if not axis.ravel()[node]:
                continue
            row, col = divmod(node, axis.shape[1])
            x, y = xy(transform, row, col, offset="center")
            vector = _line_tangent(line, Point(x, y), tangent_span_m)
            if vector is None:
                continue
            # At a confluence, the downstream segment starts at the junction
            # and must set the outgoing orientation rather than an incoming arm.
            if not np.isfinite(tangent_x[row, col]) or path_index == 0:
                tangent_x[row, col], tangent_y[row, col] = vector
    valid_axis = axis & np.isfinite(tangent_x) & np.isfinite(tangent_y)
    if not valid_axis.any():
        return tangent_x, tangent_y, connector_lines
    _, nearest = distance_transform_edt(~valid_axis, return_indices=True)
    return tangent_x[nearest[0], nearest[1]], tangent_y[nearest[0], nearest[1]], connector_lines


def _complete_floodplain_connector_axis(
    axis: np.ndarray, receiver: np.ndarray, floodplain: np.ndarray, transform: Any, cross_m: float,
) -> np.ndarray:
    """Extend the supplied high-accumulation axes until they cover each floodplain.

    The high-accumulation skeleton alone leaves low-contributing overbank cells
    on an unoriented Cartesian fallback.  Add only the few receiver paths
    needed to bring every connected floodplain cell within one cross-stream
    spacing of an axis.  The path direction remains the conditioned D4
    receiver, while later tangent smoothing removes its staircase appearance.
    """
    completed = axis.astype(bool) & floodplain
    active = floodplain.ravel()
    flat_receiver = receiver.ravel()
    # Every connected component starts with its cells immediately upstream of
    # the river.  This gives the coverage procedure a physical downstream sink.
    downstream_active = np.zeros_like(active)
    linked = flat_receiver >= 0
    downstream_active[linked] = active[flat_receiver[linked]]
    terminal = ~downstream_active
    completed.ravel()[active & terminal] = True
    # Select one source in every cross-width raster block, then trace only its
    # receiver path to the existing network.  This is linear in the selected
    # paths; the previous repeated global distance transform was correct but
    # made a full catchment mesh needlessly slow.
    cell_m = min(abs(transform.a), abs(transform.e))
    stride = max(1, int(np.floor(cross_m / cell_m)))
    rows, cols = floodplain.shape
    padded = np.pad(floodplain, ((0, (-rows) % stride), (0, (-cols) % stride)))
    block_rows, block_cols = padded.shape[0] // stride, padded.shape[1] // stride
    blocks = padded.reshape(block_rows, stride, block_cols, stride).transpose(0, 2, 1, 3).reshape(block_rows, block_cols, -1)
    has_cell = blocks.any(axis=2)
    block_row, block_col = np.where(has_cell)
    chosen = blocks.argmax(axis=2)[block_row, block_col]
    source_rows = block_row * stride + chosen // stride
    source_cols = block_col * stride + chosen % stride
    sources = source_rows * cols + source_cols
    for start in sources:
        current = int(start)
        visited: set[int] = set()
        while current >= 0 and current not in visited:
            visited.add(current)
            if not active[current] or completed.ravel()[current]:
                break
            completed.ravel()[current] = True
            current = int(flat_receiver[current])
    return completed


def _floodplain_connector_ribbon_seeds(
    connector_axes: list[LineString], floodplain_geometry, floodplain_mask: np.ndarray, transform: Any,
    along_m: float, cross_m: float, minimum_distance_m: float,
) -> np.ndarray:
    """Fill floodplains with streamwise/cross-stream lattices grown from connectors.

    Each smooth connector is sampled at the requested streamwise spacing and
    expanded normal to that line until it leaves the same floodplain polygon.
    This creates actual 40 x 90 m-style seed ribbons; it does not thin a
    Cartesian raster lattice and relabel it as flow-aligned.  Callers must
    provide a complete, receiver-connected axis network: no unoriented
    floodplain fallback is permitted.
    """
    candidates: list[tuple[float, float]] = []
    tangent_span_m = max(1.5 * along_m, min(abs(transform.a), abs(transform.e)))
    for line in sorted(connector_axes, key=lambda item: item.length, reverse=True):
        count = max(1, int(np.ceil(line.length / along_m)))
        for step in range(count):
            point = line.interpolate((step + 0.5) * line.length / count)
            tangent = _line_tangent(line, point, tangent_span_m)
            if tangent is None:
                continue
            nx, ny = -tangent[1], tangent[0]
            if floodplain_geometry.covers(point):
                candidates.append((point.x, point.y))
            for sign in (-1.0, 1.0):
                offset = cross_m
                while True:
                    candidate = Point(point.x + sign * offset * nx, point.y + sign * offset * ny)
                    if not floodplain_geometry.covers(candidate):
                        break
                    candidates.append((candidate.x, candidate.y))
                    offset += cross_m
    primary = _minimum_separated(np.unique(np.asarray(candidates), axis=0), minimum_distance_m) if candidates else np.empty((0, 2), dtype=np.float64)
    rows, cols = np.where(floodplain_mask)
    if not len(rows):
        return primary
    if not len(primary):
        raise ValueError("A non-empty floodplain requires receiver-connected ribbon seeds.")
    xs, ys = xy(transform, rows, cols, offset="center")
    floodplain_points = np.column_stack((xs, ys)).astype(np.float64)
    if np.any(cKDTree(primary).query(floodplain_points)[0] > 2.0 * cross_m):
        raise ValueError("Floodplain connector ribbons left an unoriented coverage gap.")
    return primary


def _boundary_support_seeds(boundaries, spacing_m: float, offset_m: float, domain) -> np.ndarray:
    """Add paired sites next to constrained feature boundaries to avoid slivers."""
    result: list[tuple[float, float]] = []
    lines = [boundaries] if isinstance(boundaries, LineString) else list(getattr(boundaries, "geoms", []))
    for line in lines:
        if not isinstance(line, LineString) or line.length <= 0:
            continue
        for distance in np.arange(0.0, line.length + 0.5 * spacing_m, spacing_m):
            point = line.interpolate(min(distance, line.length))
            before = line.interpolate(max(0.0, distance - min(1.0, 0.2 * spacing_m)))
            after = line.interpolate(min(line.length, distance + min(1.0, 0.2 * spacing_m)))
            dx, dy = after.x - before.x, after.y - before.y
            length = float(np.hypot(dx, dy))
            if length <= 0:
                continue
            nx, ny = -dy / length, dx / length
            for sign in (-1.0, 1.0):
                candidate = (point.x + sign * offset_m * nx, point.y + sign * offset_m * ny)
                if domain.covers(Point(candidate)):
                    result.append(candidate)
    return np.unique(np.asarray(result, dtype=np.float64), axis=0) if result else np.empty((0, 2), dtype=np.float64)


def _split_by_feature_boundaries(polygons: list[Polygon], boundaries, minimum_piece_area_m2: float) -> list[Polygon]:
    """Apply the same hard lines to every Voronoi cell, preserving conformity."""
    lines = [line for line in _line_parts(boundaries) if line.length > 0]
    if not lines:
        return list(polygons)
    # Only a tiny fraction of a large Voronoi mesh intersects a river or
    # floodplain boundary.  Query those candidates once; the old nested loop
    # tested every polygon against every boundary segment.
    # A boundary line that merely touches a polygon edge cannot create a new
    # finite-volume face.  Restrict the expensive split work to real crossings.
    pairs = STRtree(polygons).query(np.asarray(lines, dtype=object), predicate="crosses")
    cuts_by_polygon: dict[int, list[LineString]] = {}
    for line_index, polygon_index in zip(pairs[0], pairs[1], strict=True):
        cuts_by_polygon.setdefault(int(polygon_index), []).append(lines[int(line_index)])
    result: list[Polygon] = []
    for index, polygon in enumerate(polygons):
        cuts = cuts_by_polygon.get(index)
        if not cuts:
            result.append(polygon)
            continue
        pieces = [polygon]
        for line in cuts:
            updated: list[Polygon] = []
            for piece in pieces:
                if not piece.crosses(line):
                    updated.append(piece)
                    continue
                try:
                    split_pieces = [part for part in split(piece, line).geoms if part.geom_type == "Polygon"]
                except Exception:
                    split_pieces = [piece]
                if len(split_pieces) > 1 and min(part.area for part in split_pieces) >= minimum_piece_area_m2:
                    updated.extend(split_pieces)
                else:
                    updated.append(piece)
            pieces = updated
        result.extend(pieces)
    return result


def _feature_classes(polygons: list[Polygon], river_geometry, floodplain_geometry, urban: np.ndarray, transform: Any) -> list[str]:
    """Classify after hard splitting on the aligned source grid.

    A Python ``covers`` call per polygon made the class pass dominate the Pune
    build.  The feature geometries are already raster-aligned, so classifying
    each representative location against one shared raster gives the same
    priority (river, floodplain, urban, rural) without repeated complex-geometry
    predicates.
    """
    objects = np.asarray(polygons, dtype=object)
    points = point_on_surface(objects)
    rows, cols = rasterio.transform.rowcol(transform, get_x(points), get_y(points))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    inside = (rows >= 0) & (rows < urban.shape[0]) & (cols >= 0) & (cols < urban.shape[1])
    result = np.full(len(objects), "rural", dtype=object)
    valid_index = np.flatnonzero(inside)
    if len(valid_index):
        result[valid_index[urban[rows[valid_index], cols[valid_index]]]] = "urban"
    if not floodplain_geometry.is_empty:
        floodplain = geometry_mask(
            [floodplain_geometry], out_shape=urban.shape, transform=transform, invert=True,
        )
        result[valid_index[floodplain[rows[valid_index], cols[valid_index]]]] = "floodplain"
    if not river_geometry.is_empty:
        river = geometry_mask(
            [river_geometry], out_shape=urban.shape, transform=transform, invert=True,
        )
        result[valid_index[river[rows[valid_index], cols[valid_index]]]] = "river"
    return result.tolist()


def _polygons_from_seeds(seeds: np.ndarray, domain: Polygon) -> list[Polygon]:
    diagram = voronoi_polygons(MultiPoint(seeds), extend_to=domain)
    cells = np.asarray(diagram.geoms, dtype=object)
    # The clipped cells are only those crossing the catchment boundary.  A
    # predicate against the *whole* complex domain for every cell was almost as
    # costly as intersecting every cell.  Index its individual boundary
    # segments instead: the spatial query selects only the narrow edge band.
    boundary_segments: list[LineString] = []
    for line in _line_parts(domain.boundary):
        coordinates = list(line.coords)
        boundary_segments.extend(
            LineString((left, right))
            for left, right in zip(coordinates[:-1], coordinates[1:], strict=True)
            if left != right
        )
    clipped = cells.copy()
    if boundary_segments:
        pairs = STRtree(boundary_segments).query(cells, predicate="intersects")
        boundary_cells = np.unique(pairs[0]) if pairs.size else np.empty(0, dtype=np.int64)
        if len(boundary_cells):
            clipped[boundary_cells] = intersection(cells[boundary_cells], domain)
    return [geometry for geometry in clipped if geometry.geom_type == "Polygon" and geometry.area > 1e-8]


def _bad_face_seed_removals(
    seed_pairs: np.ndarray, protected: np.ndarray, seed_target: np.ndarray,
) -> np.ndarray:
    """Greedily remove few generators while covering every repairable bad face."""
    pairs = np.unique(np.sort(np.asarray(seed_pairs, dtype=np.int64), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    uncovered = np.ones(len(pairs), dtype=bool)
    selected: list[int] = []
    while np.any(uncovered):
        active = pairs[uncovered]
        active_candidates, counts = np.unique(active.ravel(), return_counts=True)
        repairable = ~protected[active_candidates]
        candidates, candidate_counts = active_candidates[repairable], counts[repairable]
        if not len(candidates):
            # A river bend or confluence can create a bad face between two
            # otherwise protected river sites.  Keeping both would make the
            # hard geometry gate impossible to satisfy, so thin only that
            # protected/protected conflict.
            candidates, candidate_counts = active_candidates, counts
        best_count = candidate_counts.max()
        best = candidates[candidate_counts == best_count]
        chosen = int(best[np.argmax(seed_target[best])])
        selected.append(chosen)
        uncovered &= ~np.any(pairs == chosen, axis=1)
    return np.asarray(selected, dtype=np.int64)


def _repair_short_cells(
    seeds: np.ndarray, domain: Polygon, minimum_cell_width_m: float, max_iterations: int = 6, protected_seed_count: int = 0,
    target: np.ndarray | None = None, transform: Any | None = None, minimum_face_length_factor: float = 0.25,
    minimum_center_distance_factor: float = 0.5, protected_seed_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Polygon], int]:
    """Remove generators responsible for undersized cells or internal faces.

    For a square cell, 2A/P is half its side length, hence the corresponding
    quality floor is one half of the user-facing minimum physical width.
    """
    floor = 0.5 * minimum_cell_width_m
    current = seeds
    protected = np.zeros(len(current), dtype=bool)
    if protected_seed_count:
        protected[-protected_seed_count:] = True
    if protected_seed_indices is not None:
        protected[np.asarray(protected_seed_indices, dtype=np.int64)] = True
    for iteration in range(max_iterations + 1):
        polygons = _polygons_from_seeds(current, domain)
        scale = 2.0 * np.asarray([item.area for item in polygons]) / np.maximum(np.asarray([item.length for item in polygons]), 1e-12)
        bad = np.flatnonzero(scale < floor)
        remove = np.empty(0, dtype=np.int64)
        forced_protected_removals = np.empty(0, dtype=np.int64)
        if len(bad):
            representative_points = np.asarray([(polygons[index].representative_point().x, polygons[index].representative_point().y) for index in bad])
            remove = cKDTree(current).query(representative_points)[1]
        if target is not None and transform is not None:
            face_qa = _edge_quality(
                polygons, current, target, transform,
                minimum_face_length_factor * minimum_cell_width_m,
                minimum_center_distance_factor * minimum_cell_width_m,
            )
            if len(face_qa["bad_owners"]):
                rows, cols = rasterio.transform.rowcol(transform, current[:, 0], current[:, 1])
                rows = np.clip(rows, 0, target.shape[0] - 1)
                cols = np.clip(cols, 0, target.shape[1] - 1)
                seed_target = target[np.asarray(rows), np.asarray(cols)]
                owner_seed = face_qa["face_seed"]
                edge_remove = _bad_face_seed_removals(
                    owner_seed[face_qa["bad_owners"]], protected, seed_target,
                )
                forced_protected_removals = edge_remove[protected[edge_remove]]
                remove = np.r_[remove, edge_remove]
        remove = np.unique(remove)
        if not len(remove) or iteration == max_iterations:
            return current, polygons, iteration
        # Callers may reserve genuine external anchors.  Feature geometry is
        # not tied to a single seed, so tight-bend refinement can be thinned.
        remove = remove[
            (~protected[remove]) | np.isin(remove, forced_protected_removals)
        ]
        if not len(remove):
            return current, polygons, iteration
        keep = np.ones(len(current), dtype=bool)
        keep[remove] = False
        current, protected = current[keep], protected[keep]
    raise AssertionError("unreachable")


def _topology(polygons: Iterable[Polygon]) -> tuple[list[list[tuple[float, float]]], dict[tuple[tuple[float, float], tuple[float, float]], list[int]]]:
    faces: list[list[tuple[float, float]]] = []
    edges: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for face_id, polygon in enumerate(polygons):
        coordinates = [(round(x, 8), round(y, 8)) for x, y in list(polygon.exterior.coords)[:-1]]
        coordinates = [
            point for index, point in enumerate(coordinates)
            if index == 0 or point != coordinates[index - 1]
        ]
        if len(coordinates) < 3:
            continue
        faces.append(coordinates)
        for a, b in zip(coordinates, coordinates[1:] + coordinates[:1], strict=True):
            if a != b:
                edges.setdefault(tuple(sorted((a, b))), []).append(face_id)
    return faces, edges


def _edge_quality(
    polygons: list[Polygon], seeds: np.ndarray, target: np.ndarray, transform: Any,
    minimum_face_length_m: float, minimum_center_distance_m: float,
) -> dict[str, Any]:
    """Evaluate finite-volume internal faces against their two local target sizes."""
    objects = np.asarray(polygons, dtype=object)
    coordinate_count = get_num_coordinates(objects)
    coordinates = get_coordinates(objects)
    if not len(coordinates):
        return {"min_internal_face_length_m": 0.0, "min_internal_center_distance_m": 0.0, "short_internal_face_count": 0, "close_internal_center_count": 0, "bad_owners": np.empty((0, 2), dtype=np.int64), "short_owners": np.empty((0, 2), dtype=np.int64), "face_seed": np.empty(0, dtype=np.int64), "face_target_m": np.empty(0)}
    # The former implementation built a Python dictionary by iterating every
    # vertex of every polygon.  This vector form keeps the same rounded-edge
    # convention while making the 1.2 M-cell Pune quality check practical.
    first = np.r_[0, np.cumsum(coordinate_count[:-1])]
    coordinate_owner = np.repeat(np.arange(len(objects), dtype=np.int64), coordinate_count)
    edge_start = np.ones(len(coordinates), dtype=bool)
    edge_start[first + coordinate_count - 1] = False  # closed polygon ring
    start = np.flatnonzero(edge_start)
    edge_owner = coordinate_owner[start]
    edge_start_xy = coordinates[start]
    edge_end_xy = coordinates[start + 1]
    nonzero = np.hypot(
        edge_start_xy[:, 0] - edge_end_xy[:, 0], edge_start_xy[:, 1] - edge_end_xy[:, 1],
    ) > 1e-8
    edge_owner = edge_owner[nonzero]
    edge_start_xy = edge_start_xy[nonzero]
    edge_end_xy = edge_end_xy[nonzero]
    if not len(edge_owner):
        return {"min_internal_face_length_m": 0.0, "min_internal_center_distance_m": 0.0, "short_internal_face_count": 0, "close_internal_center_count": 0, "bad_owners": np.empty((0, 2), dtype=np.int64), "short_owners": np.empty((0, 2), dtype=np.int64), "face_seed": np.empty(0, dtype=np.int64), "face_target_m": np.empty(0)}
    swap = (edge_start_xy[:, 0] > edge_end_xy[:, 0]) | (
        (edge_start_xy[:, 0] == edge_end_xy[:, 0]) & (edge_start_xy[:, 1] > edge_end_xy[:, 1])
    )
    edge_key = np.round(np.column_stack((
        np.where(swap, edge_end_xy[:, 0], edge_start_xy[:, 0]),
        np.where(swap, edge_end_xy[:, 1], edge_start_xy[:, 1]),
        np.where(swap, edge_start_xy[:, 0], edge_end_xy[:, 0]),
        np.where(swap, edge_start_xy[:, 1], edge_end_xy[:, 1]),
    )), 8)
    order = np.lexsort((edge_key[:, 3], edge_key[:, 2], edge_key[:, 1], edge_key[:, 0]))
    edge_key = edge_key[order]
    edge_owner = edge_owner[order]
    group_start = np.r_[0, np.flatnonzero(np.any(edge_key[1:] != edge_key[:-1], axis=1)) + 1]
    group_end = np.r_[group_start[1:], len(edge_key)]
    paired = (group_end - group_start) == 2
    if not np.any(paired):
        return {"min_internal_face_length_m": 0.0, "min_internal_center_distance_m": 0.0, "short_internal_face_count": 0, "close_internal_center_count": 0, "bad_owners": np.empty((0, 2), dtype=np.int64), "short_owners": np.empty((0, 2), dtype=np.int64), "face_seed": np.empty(0, dtype=np.int64), "face_target_m": np.empty(0)}
    owner_index = group_start[paired]
    owners = np.column_stack((edge_owner[owner_index], edge_owner[owner_index + 1]))
    internal_key = edge_key[owner_index]
    lengths = np.hypot(internal_key[:, 0] - internal_key[:, 2], internal_key[:, 1] - internal_key[:, 3])
    representative_geometry = point_on_surface(objects)
    representative = np.column_stack((get_x(representative_geometry), get_y(representative_geometry)))
    face_seed = cKDTree(seeds).query(representative)[1]
    rows, cols = rasterio.transform.rowcol(transform, seeds[:, 0], seeds[:, 1])
    rows = np.clip(rows, 0, target.shape[0] - 1); cols = np.clip(cols, 0, target.shape[1] - 1)
    face_target = target[np.asarray(rows), np.asarray(cols)]
    centres_geometry = centroid(objects)
    centres = np.column_stack((get_x(centres_geometry), get_y(centres_geometry)))
    distance = np.hypot(centres[owners[:, 0], 0] - centres[owners[:, 1], 0], centres[owners[:, 0], 1] - centres[owners[:, 1], 1])
    short = lengths < minimum_face_length_m
    close = distance < minimum_center_distance_m
    return {
        "min_internal_face_length_m": float(lengths.min()),
        "min_internal_center_distance_m": float(distance.min()),
        "minimum_face_length_threshold_m": float(minimum_face_length_m),
        "minimum_center_distance_threshold_m": float(minimum_center_distance_m),
        "short_internal_face_count": int(short.sum()),
        "close_internal_center_count": int(close.sum()),
        "bad_owners": owners[short | close],
        "short_owners": owners[short],
        "face_seed": face_seed,
        "face_target_m": face_target,
    }


def _agglomerate_short_face_cells(
    polygons: list[Polygon], minimum_face_length_m: float, minimum_center_distance_m: float = 0.0,
    minimum_cell_scale_m: float = 0.0, max_merges: int = 256, classes: list[str] | None = None,
    candidate_pairs: np.ndarray | None = None,
) -> tuple[list[Polygon], int]:
    """Replace only pathological Voronoi fragments with their neighbour.

    A short face carries negligible hydraulic information but can control the
    explicit timestep.  Merging its two owners is a local, conservative mesh
    operation: the union exactly preserves area and all other cell boundaries.
    It is intentionally applied after the river/HAND feature sites are placed,
    never by regenerating the global feature mesh.
    """
    current = list(polygons)
    # The full topology is already available from the preceding vector QA.
    # Merge a non-overlapping subset directly instead of rebuilding a million-
    # cell edge dictionary after each local union.  Feature classes are an
    # inviolable boundary: river/floodplain cells may never merge into rural
    # or urban cells during numerical cleanup.
    if candidate_pairs is not None:
        selected: list[tuple[int, int]] = []
        used: set[int] = set()
        for left, right in np.asarray(candidate_pairs, dtype=np.int64):
            if left == right or left in used or right in used:
                continue
            if classes is not None and classes[left] != classes[right]:
                continue
            selected.append((int(left), int(right)))
            used.update((int(left), int(right)))
            if len(selected) == max_merges:
                break
        merged_at: dict[int, Polygon] = {}
        removed: set[int] = set()
        for left, right in selected:
            merged = unary_union((current[left], current[right]))
            if merged.geom_type == "Polygon" and merged.area > 0:
                merged_at[left] = merged
                removed.add(right)
        if not merged_at:
            return current, 0
        return [
            merged_at[index] if index in merged_at else polygon
            for index, polygon in enumerate(current) if index not in removed
        ], len(merged_at)
    merges = 0
    while merges < max_merges:
        _, edges = _topology(current)
        face_candidates = sorted(
            (
                (np.hypot(edge[0][0] - edge[1][0], edge[0][1] - edge[1][1]), owners)
                for edge, owners in edges.items()
                if len(owners) == 2
            ),
            key=lambda item: item[0],
        )
        same_class = lambda owners: classes is None or classes[owners[0]] == classes[owners[1]]
        face_candidates = [item for item in face_candidates if same_class(item[1])]
        if face_candidates and face_candidates[0][0] < minimum_face_length_m:
            _, (left, right) = face_candidates[0]
        elif minimum_center_distance_m > 0 and face_candidates:
            centres = np.asarray([(item.centroid.x, item.centroid.y) for item in current])
            close_candidates = sorted(
                (
                    (np.hypot(*(centres[owners[0]] - centres[owners[1]])), owners)
                    for _, owners in face_candidates
                    if np.hypot(*(centres[owners[0]] - centres[owners[1]])) < minimum_center_distance_m
                ),
                key=lambda item: item[0],
            )
            if not close_candidates:
                close_candidates = []
            if close_candidates:
                _, (left, right) = close_candidates[0]
            else:
                scales = np.asarray([2.0 * item.area / max(item.length, 1e-12) for item in current])
                bad = int(np.argmin(scales))
                if minimum_cell_scale_m <= 0 or scales[bad] >= minimum_cell_scale_m:
                    break
                neighbours = [(length, owners) for length, owners in face_candidates if bad in owners]
                if not neighbours:
                    break
                _, owners = max(neighbours, key=lambda item: item[0])
                left, right = owners
        elif minimum_cell_scale_m > 0:
            scales = np.asarray([2.0 * item.area / max(item.length, 1e-12) for item in current])
            bad = int(np.argmin(scales))
            if scales[bad] >= minimum_cell_scale_m:
                break
            neighbours = [(length, owners) for length, owners in face_candidates if bad in owners]
            if not neighbours:
                break
            _, (left, right) = max(neighbours, key=lambda item: item[0])
        else:
            break
        merged = unary_union((current[left], current[right]))
        if merged.geom_type != "Polygon" or merged.area <= 0:
            break
        current[left] = merged
        del current[right]
        if classes is not None:
            del classes[right]
        merges += 1
    return current, merges


def _write_ugrid(
    path: Path, faces: list[list[tuple[float, float]]], edges: dict[Any, list[int]], polygons: list[Polygon], crs: Any,
    target: np.ndarray, transform: Any, dem: np.ndarray, feature_classes: list[str], river_geometry, floodplain_geometry,
) -> None:
    nodes = sorted({point for face in faces for point in face})
    node_id = {point: index for index, point in enumerate(nodes)}
    connectivity = np.full((len(faces), max(map(len, faces))), -1, dtype=np.int32)
    for index, face in enumerate(faces):
        connectivity[index, : len(face)] = [node_id[point] for point in face]
    usable_edges = [(key, owners) for key, owners in edges.items() if len(owners) <= 2]
    owner = np.asarray([owners[0] for _, owners in usable_edges], dtype=np.int32)
    neighbour = np.asarray([owners[1] if len(owners) == 2 else -1 for _, owners in usable_edges], dtype=np.int32)
    length = np.asarray([np.hypot(a[0] - b[0], a[1] - b[1]) for (a, b), _ in usable_edges])
    centres = np.asarray([(item.centroid.x, item.centroid.y) for item in polygons])
    areas = np.asarray([item.area for item in polygons])
    river_fraction = np.asarray([item.intersection(river_geometry).area / item.area if not river_geometry.is_empty else 0.0 for item in polygons])
    floodplain_fraction = np.asarray([item.intersection(floodplain_geometry).area / item.area if not floodplain_geometry.is_empty else 0.0 for item in polygons])
    rows, cols = rasterio.transform.rowcol(transform, centres[:, 0], centres[:, 1])
    rows, cols = np.clip(rows, 0, dem.shape[0] - 1), np.clip(cols, 0, dem.shape[1] - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    midpoint_x = np.asarray([(a[0] + b[0]) / 2 for (a, b), _ in usable_edges])
    midpoint_y = np.asarray([(a[1] + b[1]) / 2 for (a, b), _ in usable_edges])
    center_distance = np.hypot(
        centres[owner, 0] - centres[np.maximum(neighbour, owner), 0],
        centres[owner, 1] - centres[np.maximum(neighbour, owner), 1],
    )
    boundary = neighbour < 0
    center_distance[boundary] = np.hypot(
        centres[owner[boundary], 0] - midpoint_x[boundary],
        centres[owner[boundary], 1] - midpoint_y[boundary],
    )
    control_width = np.full(len(faces), np.inf)
    for edge_index, (left, right) in enumerate(zip(owner, neighbour)):
        width = 2.0 * center_distance[edge_index] if right < 0 else center_distance[edge_index]
        control_width[left] = min(control_width[left], width)
        if right >= 0:
            control_width[right] = min(control_width[right], width)
    refinement_codes = np.asarray(
        [{"rural": 0, "urban": 1, "floodplain": 2, "river": 3}[item] for item in feature_classes],
        dtype=np.int16,
    )
    normal_x = np.empty(len(usable_edges)); normal_y = np.empty(len(usable_edges))
    for index, ((a, b), owners) in enumerate(usable_edges):
        dx, dy = b[0] - a[0], b[1] - a[1]
        nx, ny = dy / max(np.hypot(dx, dy), 1e-12), -dx / max(np.hypot(dx, dy), 1e-12)
        vector_x, vector_y = midpoint_x[index] - centres[owners[0], 0], midpoint_y[index] - centres[owners[0], 1]
        if nx * vector_x + ny * vector_y < 0:
            nx, ny = -nx, -ny
        normal_x[index], normal_y[index] = nx, ny
    # A neighboring polygon pair can share two collinear segments when one
    # ring retains an intermediate vertex.  HydroPol2D represents that
    # interface as one finite-volume face, so aggregate only those duplicate
    # internal pairs while retaining distinct perimeter segments.
    internal = neighbour >= 0
    if np.any(internal):
        swap = internal & (owner > neighbour)
        owner[swap], neighbour[swap] = neighbour[swap].copy(), owner[swap].copy()
        normal_x[swap] *= -1; normal_y[swap] *= -1
        internal_index = np.flatnonzero(internal)
        order = np.lexsort((neighbour[internal], owner[internal]))
        internal_index = internal_index[order]
        sorted_owner = owner[internal_index]; sorted_neighbour = neighbour[internal_index]
        starts = np.r_[0, np.flatnonzero(
            (sorted_owner[1:] != sorted_owner[:-1]) |
            (sorted_neighbour[1:] != sorted_neighbour[:-1])
        ) + 1]
        weights = length[internal_index]
        total_length = np.add.reduceat(weights, starts)
        aggregate = lambda values: np.add.reduceat(values[internal_index] * weights, starts) / total_length
        merged_owner = sorted_owner[starts]
        merged_neighbour = sorted_neighbour[starts]
        merged_midpoint_x = aggregate(midpoint_x)
        merged_midpoint_y = aggregate(midpoint_y)
        merged_normal_x = aggregate(normal_x)
        merged_normal_y = aggregate(normal_y)
        normal_scale = np.maximum(np.hypot(merged_normal_x, merged_normal_y), 1e-12)
        merged_normal_x /= normal_scale; merged_normal_y /= normal_scale
        merged_distance = center_distance[internal_index[starts]]
        boundary_index = np.flatnonzero(~internal)
        owner = np.r_[merged_owner, owner[boundary_index]]
        neighbour = np.r_[merged_neighbour, neighbour[boundary_index]]
        length = np.r_[total_length, length[boundary_index]]
        midpoint_x = np.r_[merged_midpoint_x, midpoint_x[boundary_index]]
        midpoint_y = np.r_[merged_midpoint_y, midpoint_y[boundary_index]]
        normal_x = np.r_[merged_normal_x, normal_x[boundary_index]]
        normal_y = np.r_[merged_normal_y, normal_y[boundary_index]]
        center_distance = np.r_[merged_distance, center_distance[boundary_index]]
        control_width = np.full(len(faces), np.inf)
        for edge_index, (left, right) in enumerate(zip(owner, neighbour)):
            width = 2.0 * center_distance[edge_index] if right < 0 else center_distance[edge_index]
            control_width[left] = min(control_width[left], width)
            if right >= 0:
                control_width[right] = min(control_width[right], width)
    with netCDF4.Dataset(path, "w") as ds:
        ds.Conventions = "CF-1.13, UGRID-1.0"
        ds.title = "HydroBathyDEM adaptive HydroPol2D mesh"
        ds.crs_wkt = crs.to_wkt() if crs else ""
        ds.createDimension("face", len(faces)); ds.createDimension("node", len(nodes)); ds.createDimension("edge", len(owner)); ds.createDimension("max_face_nodes", connectivity.shape[1])
        ds.createVariable("mesh2d_face_nodes", "i4", ("face", "max_face_nodes"), fill_value=-1)[:] = connectivity
        ds.createVariable("mesh2d_node_x", "f8", ("node",))[:] = [item[0] for item in nodes]
        ds.createVariable("mesh2d_node_y", "f8", ("node",))[:] = [item[1] for item in nodes]
        for name, value in {"cell_area_m2": areas, "cell_bed_elevation_m": dem[rows, cols], "mesh2d_face_x": centres[:, 0], "mesh2d_face_y": centres[:, 1], "cell_target_width_m": target[rows, cols], "cell_cfl_width_m": control_width, "cell_hydraulic_roughness": np.full(len(faces), np.nan)}.items():
            ds.createVariable(name, "f8", ("face",))[:] = value
        ds.createVariable("cell_refinement_source", "i2", ("face",))[:] = refinement_codes
        ds.createVariable("edge_owner", "i4", ("edge",))[:] = owner
        ds.createVariable("edge_neighbor", "i4", ("edge",))[:] = neighbour
        ds.createVariable("edge_length_m", "f8", ("edge",))[:] = length
        ds.createVariable("edge_center_distance_m", "f8", ("edge",))[:] = center_distance
        ds.createVariable("edge_midpoint_x", "f8", ("edge",))[:] = midpoint_x
        ds.createVariable("edge_midpoint_y", "f8", ("edge",))[:] = midpoint_y
        ds.createVariable("edge_normal_x", "f8", ("edge",))[:] = normal_x
        ds.createVariable("edge_normal_y", "f8", ("edge",))[:] = normal_y
        ds.createVariable("edge_boundary_type", "i2", ("edge",))[:] = 0
        ds.createVariable("cell_river_fraction", "f8", ("face",))[:] = river_fraction
        ds.createVariable("cell_floodplain_fraction", "f8", ("face",))[:] = floodplain_fraction
        ds.createVariable("cell_feature_class", str, ("face",))[:] = np.asarray(feature_classes, dtype=object)


def _write_mesh_class_figure(
    path: Path, polygons: list[Polygon], feature_classes: list[str], crs: Any,
    river_geometry, floodplain_geometry, title: str, bounds: tuple[float, float, float, float] | None = None,
    connector_axes: list[LineString] | None = None,
) -> None:
    """Write a compact mesh-stage diagnostic using the same class palette."""
    colours = {"rural": "#ECECEC", "urban": "#E6A0A1", "floodplain": "#8CC5E3", "river": "#0B81A2"}
    fig, ax = plt.subplots(figsize=(10, 8))
    dense_full_domain = bounds is None and len(polygons) > 100_000
    gpd.GeoDataFrame({"feature_class": feature_classes}, geometry=polygons, crs=crs).plot(
        ax=ax, color=[colours[item] for item in feature_classes],
        edgecolor=(0.1, 0.1, 0.1, 0.18) if dense_full_domain else "0.16",
        linewidth=0.05 if dense_full_domain else 0.35,
    )
    if not river_geometry.is_empty:
        gpd.GeoSeries([river_geometry.boundary], crs=crs).plot(ax=ax, color="#082A54", linewidth=1.1)
    if not floodplain_geometry.is_empty:
        gpd.GeoSeries([floodplain_geometry.boundary], crs=crs).plot(ax=ax, color="#3594CC", linewidth=0.75, linestyle="--")
    if connector_axes:
        gpd.GeoSeries(connector_axes, crs=crs).plot(ax=ax, color="#9D2C00", linewidth=0.7, alpha=0.8)
    present = [item for item in ("river", "floodplain", "urban", "rural") if item in set(feature_classes)]
    ax.legend(
        handles=[Patch(facecolor=colours[item], edgecolor="0.2", label=item.capitalize()) for item in present],
        loc="lower left", frameon=True, framealpha=0.9, fontsize=8,
    )
    ax.set_title(title, fontname="Helvetica", fontsize=14, pad=8)
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal"); ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_hybrid_mesh(config: HybridMeshConfig) -> dict[str, Any]:
    """Build a conforming mesh and write UGRID, GeoPackage, QA, and wireframe outputs."""
    started_at = perf_counter()
    with rasterio.open(config.dem) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype(np.float64)
        profile = src.profile.copy(); transform, crs = src.transform, src.crs
        impervious = _aligned(config.impervious, src, 0.0)
        population = _aligned(config.population, src, 0.0)
        supplied_mask = _aligned(config.river_mask, src, np.nan)
        supplied_direction = _aligned(config.river_direction, src, np.nan)
        width = _aligned(config.river_width, src, np.nan)
        _aligned(config.river_depth, src, np.nan)
        _aligned(config.river_bank_height, src, np.nan)
        supplied_floodplain = _aligned(config.floodplain_mask, src, 0.0)
        supplied_floodplain_direction = _aligned(config.floodplain_direction, src, 0.0)
        supplied_floodplain_axis = _aligned(config.floodplain_axis, src, 0.0)
    valid = np.isfinite(dem)
    if config.domain_vector is None:
        domain = box(*rasterio.transform.array_bounds(*dem.shape, transform))
    else:
        boundary = gpd.read_file(config.domain_vector)
        if boundary.empty:
            raise ValueError(f"Mesh domain vector is empty: {config.domain_vector}")
        boundary = boundary.to_crs(crs)
        domain = boundary.geometry.union_all()
        # A raster-derived catchment edge may contain metre-scale teeth.  They
        # cannot support a cell with the requested hydraulic minimum, so remove
        # them before Voronoi clipping rather than exporting boundary slivers.
        edge_filter = 0.5 * config.minimum_hydraulic_width_m
        cleaned_domain = domain.buffer(-edge_filter, join_style="mitre").buffer(edge_filter, join_style="mitre")
        if not cleaned_domain.is_empty:
            domain = cleaned_domain
        valid &= geometry_mask(
            [domain], out_shape=dem.shape, transform=transform, invert=True
        )
    accumulation, receiver = compute_d4_flow_accumulation(dem.astype("float32"), profile, nodata=np.nan)
    pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    if config.river_source == "hydrobathydem_d4":
        river = valid & ((supplied_mask > 0) if np.isfinite(supplied_mask).any() else (np.isfinite(width) & (width > 0)))
        if not river.any():
            raise ValueError("rivers.source='hydrobathydem_d4' requires an aligned river_mask or positive river_width raster.")
        if np.isfinite(supplied_direction).any():
            receiver = receiver_from_d4_direction(supplied_direction)
    else:
        river = valid & (accumulation * pixel_area_km2 >= config.minimum_upstream_area_km2)
    hand, reach = connected_hand_with_reach(dem, receiver, river)
    # A HydroBathyDEM D4 case already carries LiN-derived width/depth.  Its
    # conditioned D4 geometry is the one routing, corridor, and mesh must
    # share; using a second vector centreline here creates visible detached
    # overbank corridors.
    if config.river_network is not None and config.river_source != "hydrobathydem_d4":
        river_records = _vector_river_records(
            config.river_network, config.river_width_field, crs, domain, dem, transform,
        )
        reach_orientation = _vector_reach_orientations(reach, transform, river_records)
    else:
        river_records = _river_segment_records(river, receiver, transform, width)
        reach_orientation = {record[0]: record[3] for record in river_records}
    river_geometry = _physical_river_geometry(river_records)
    # A supplied design-flow mask supersedes HAND.  The external mask is an
    # input artifact with its own provenance, not an interpolation of HAND.
    # HAND remains the explicit fallback for legacy cases.
    supplied_floodplain_geometry = None
    if config.floodplain_vector is not None:
        vector = gpd.read_file(config.floodplain_vector).to_crs(crs)
        supplied_floodplain_geometry = vector.geometry.union_all().intersection(domain).buffer(0)
        floodplain = valid & geometry_mask([supplied_floodplain_geometry], out_shape=dem.shape, transform=transform, invert=True) & ~river
    else:
        floodplain = (
            valid & (supplied_floodplain > 0) & ~river
            if config.floodplain_mask is not None
            else valid & (reach >= 0) & np.isfinite(hand) & (hand <= config.floodplain_hand_stage_m) & ~river
        )
    # The source grid is 30 m.  A topology-preserving half-pixel simplify is a
    # mesh-interface operation, not a change to the selected HAND cells: it
    # removes only sub-cell D4 stair steps that otherwise make zero-length
    # breakline fragments.  The physical width remains controlled by the
    # mapped-width buffered centreline above.
    interface_tolerance = 0.5 * min(abs(transform.a), abs(transform.e))
    river_geometry = river_geometry.simplify(interface_tolerance, preserve_topology=True)
    floodplain_geometry = (
        supplied_floodplain_geometry.difference(river_geometry)
        if supplied_floodplain_geometry is not None
        else _mask_geometry(floodplain, transform).difference(river_geometry).simplify(interface_tolerance, preserve_topology=True)
    )
    urban = valid & ((impervious >= config.impervious_threshold_percent) | (population >= config.population_threshold_per_km2))
    if config.urban_buffer_m > 0 and urban.any():
        urban |= distance_transform_edt(~urban, sampling=(abs(transform.e), abs(transform.a))) <= config.urban_buffer_m
    urban &= valid
    target = np.full(dem.shape, config.background_width_m, dtype=np.float64)
    target[urban] = np.minimum(target[urban], config.urban_width_m)
    target[floodplain] = np.minimum(target[floodplain], config.floodplain_target_width_m)
    # At least one 2-D strip is possible only at the DEM-scale floor.  Smaller
    # mapped channels keep their width/depth in Neal rather than being widened.
    resolved_river = river & np.isfinite(width) & (width >= config.minimum_hydraulic_width_m)
    target[resolved_river] = np.minimum(target[resolved_river], config.river_cross_river_target_width_m)
    target = np.maximum(target, config.minimum_hydraulic_width_m)
    target = _smooth_size_field(target, valid, config.maximum_adjacent_size_ratio)
    xmin, ymin, xmax, ymax = domain.bounds
    # 1. Build the physical river and its reach-specific HAND corridor first.
    # These sites are locked.  No urban/rural candidate may subsequently enter
    # either footprint.  2. Add paired support sites around the hard boundaries
    # so their later split does not create the tiny Voronoi faces that previously
    # controlled the timestep.
    river_ribbon = _river_core_seeds(
        river_records,
        config.river_along_river_cell_length_m,
        config.river_cross_river_target_width_m,
        config.minimum_hydraulic_width_m,
    )
    connector_axes: list[LineString] = []
    if config.floodplain_align_to_flow and config.floodplain_direction is not None:
        floodplain_receiver = receiver_from_d4_direction(supplied_floodplain_direction)
        complete_axis = _complete_floodplain_connector_axis(
            supplied_floodplain_axis > 0, floodplain_receiver, floodplain, transform,
            config.floodplain_cross_river_target_width_m,
        )
        _, _, connector_axes = _smoothed_floodplain_connector_tangents(
            complete_axis, floodplain_receiver, transform, config.floodplain_along_river_cell_length_m,
        )
        floodplain_ribbon = _floodplain_connector_ribbon_seeds(
            connector_axes, floodplain_geometry, floodplain, transform,
            config.floodplain_along_river_cell_length_m, config.floodplain_cross_river_target_width_m,
            config.minimum_hydraulic_width_m,
        )
    elif config.floodplain_align_to_flow:
        floodplain_ribbon = _feature_lattice_seeds(
            floodplain, reach, reach_orientation, transform,
            config.floodplain_along_river_cell_length_m,
            config.floodplain_cross_river_target_width_m, floodplain_geometry,
        )
    else:
        # Floodplain flow is genuinely two-dimensional.  A regular isotropic
        # lattice avoids artificial directional ribbons and leaves alignment
        # only where the mapped river supplies a defensible flow axis.
        floodplain_ribbon = _lattice(
            domain.bounds, config.floodplain_target_width_m, floodplain, transform,
        )
    feature_geometry = unary_union([river_geometry, floodplain_geometry])
    if connector_axes:
        connector_path = config.out_dir / "diagnostics" / "floodplain_connector_axes_smooth.gpkg"
        connector_path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(
            {"connector_id": range(len(connector_axes)), "orientation": "smooth_accumulation_axis"},
            geometry=connector_axes, crs=crs,
        ).to_file(connector_path, layer="connectors", driver="GPKG")
    reservation = valid & geometry_mask([feature_geometry], out_shape=dem.shape, transform=transform, invert=True) if not feature_geometry.is_empty else np.zeros(dem.shape, dtype=bool)
    # Leave one explicit transition band outside the locked feature core.  The
    # old code placed a 45 m urban lattice directly against a 30 m river seed,
    # which made arbitrary 1--5 m Voronoi faces.  The band is filled by the
    # paired support sites below, then connects to the regular graded lattice.
    transition_exclusion = valid & geometry_mask(
        [feature_geometry.buffer(0.5 * config.floodplain_target_width_m)],
        out_shape=dem.shape, transform=transform, invert=True,
    ) if not feature_geometry.is_empty else reservation
    hard_boundaries = unary_union([river_geometry.boundary, floodplain_geometry.boundary]) if not feature_geometry.is_empty else LineString()
    support = (
        _boundary_support_seeds(
            hard_boundaries,
            spacing_m=min(config.floodplain_target_width_m, config.urban_width_m),
            offset_m=0.5 * config.floodplain_target_width_m,
            domain=domain,
        )
        if config.enforce_feature_boundaries else np.empty((0, 2), dtype=np.float64)
    )
    # Feature seeds already occupy both sides of the river/floodplain internal
    # boundary.  Keep support sites only in the outer transition zone; adding
    # them 15 m inside a 30 m river cell was the direct source of the tiny
    # generator separations seen in the rejected candidate.
    if len(support):
        support = np.asarray([point for point in support if not feature_geometry.covers(Point(point))], dtype=np.float64)
    # At confluences, a D4 segment, a connector ribbon, and a boundary-support
    # site can otherwise land almost on top of one another.  Lock the physical
    # river first, then retain only floodplain/support sites that meet the same
    # user-facing hydraulic floor.  This removes zero-area Voronoi fragments at
    # their source rather than hiding them in a later repair.
    river_ribbon = _minimum_separated(river_ribbon, config.minimum_hydraulic_width_m)
    floodplain_ribbon = _drop_near(
        _minimum_separated(floodplain_ribbon, config.minimum_hydraulic_width_m),
        river_ribbon, config.minimum_hydraulic_width_m,
    )
    support = _drop_near(
        _minimum_separated(support, config.minimum_hydraulic_width_m),
        np.vstack([item for item in (river_ribbon, floodplain_ribbon) if len(item)])
        if any(len(item) for item in (river_ribbon, floodplain_ribbon)) else np.empty((0, 2), dtype=np.float64),
        config.minimum_hydraulic_width_m,
    )
    feature_seeds = np.vstack([item for item in (river_ribbon, floodplain_ribbon, support) if len(item)]) if any(len(item) for item in (river_ribbon, floodplain_ribbon, support)) else np.empty((0, 2), dtype=np.float64)
    # 3. Fill the remaining urban/rural domain with the graded target field.
    # Every level is used (for example 45 -> 90 -> 120 m), so the 2:1 transition
    # is explicit rather than a binary refinement mask.
    lattice_seeds: list[np.ndarray] = []
    levels = np.unique(target[valid & ~transition_exclusion])
    blockers = feature_seeds
    for spacing in np.sort(levels):
        candidates = _lattice(
            (xmin, ymin, xmax, ymax), float(spacing),
            valid & ~transition_exclusion & np.isclose(target, spacing), transform,
        )
        candidates = _drop_near(candidates, blockers, config.minimum_hydraulic_width_m)
        lattice_seeds.append(candidates)
        blockers = np.vstack((blockers, candidates)) if len(blockers) else candidates
    seeds = lattice_seeds
    if feature_seeds.size:
        seeds.append(feature_seeds)
    interior_points = np.unique(np.vstack(seeds), axis=0)
    # ``voronoi_polygons(..., extend_to=domain)`` already bounds the diagram.
    # Envelope-corner generators lie outside a non-rectangular catchment and
    # can create four large exterior cells that never cross its true boundary.
    # Keep only physical/interior sites, then clip the narrow boundary band.
    points = interior_points
    # Feature geometry/classification is retained independently of its seed.
    # At a tight bend or confluence, a redundant feature seed may therefore be
    # removed if it creates a control volume below the hydraulic floor.
    protected_core = np.empty(0, dtype=np.int64)
    if len(river_ribbon):
        distance, protected_core = cKDTree(points).query(river_ribbon)
        protected_core = np.unique(protected_core[distance < 1e-6])
    if points.shape[0] > config.max_seed_count:
        raise ValueError(
            f"Mesh preflight generated {points.shape[0]:,} seeds, above max_seed_count="
            f"{config.max_seed_count:,}. Use a smaller domain, coarser background, or a higher gate."
        )
    points, polygons, quality_repair_iterations = _repair_short_cells(
        points, domain, config.minimum_hydraulic_width_m, max_iterations=config.quality_repair_max_iterations,
        protected_seed_count=0, target=target,
        transform=transform, minimum_face_length_factor=config.minimum_face_length_factor,
        minimum_center_distance_factor=config.minimum_center_distance_factor,
        protected_seed_indices=protected_core,
    )
    step5_classes = _feature_classes(polygons, river_geometry, floodplain_geometry, urban, transform)
    _write_mesh_class_figure(
        config.out_dir / "diagnostics" / "mesh_step5_global_voronoi.png",
        polygons, step5_classes, crs, river_geometry, floodplain_geometry,
        "Step 5 — river-aligned and isotropic floodplain Voronoi mesh",
        connector_axes=connector_axes,
    )
    if config.diagnostic_window is not None:
        diagnostic = gpd.read_file(config.diagnostic_window).to_crs(crs).geometry.union_all()
        detail_indices = [index for index, polygon in enumerate(polygons) if polygon.intersects(diagnostic)]
        _write_mesh_class_figure(
            config.out_dir / "diagnostics" / "mesh_step5_detail.png",
            [polygons[index] for index in detail_indices], [step5_classes[index] for index in detail_indices], crs,
            river_geometry.intersection(diagnostic), floodplain_geometry.intersection(diagnostic),
            "Step 5 — detailed global Voronoi candidate", diagnostic.bounds, connector_axes,
        )
    if config.diagnostic_windows is not None:
        windows = gpd.read_file(config.diagnostic_windows).to_crs(crs)
        for index, geometry in enumerate(windows.geometry, start=1):
            detail_indices = [item for item, polygon in enumerate(polygons) if polygon.intersects(geometry)]
            _write_mesh_class_figure(
                config.out_dir / "diagnostics" / f"mesh_step5_region_{index:02d}.png",
                [polygons[item] for item in detail_indices], [step5_classes[item] for item in detail_indices], crs,
                river_geometry.intersection(geometry), floodplain_geometry.intersection(geometry),
                f"Step 5 — mesh quality region {index}", geometry.bounds, connector_axes,
            )
    if config.stop_after_step == 5:
        return {
            "status": "step5_diagnostic_only",
            "step5_figure": str(config.out_dir / "diagnostics" / "mesh_step5_global_voronoi.png"),
            "cells": len(polygons),
        }
    # 4. Convert both physical feature boundaries into common finite-volume
    # faces.  Because exactly the same lines split every affected polygon, the
    # result is conforming; cells are classified only after this operation.
    if config.enforce_feature_boundaries:
        polygons = _split_by_feature_boundaries(
            polygons, hard_boundaries, 0.5 * config.minimum_hydraulic_width_m ** 2,
        )
    feature_classes = _feature_classes(polygons, river_geometry, floodplain_geometry, urban, transform)
    # Seed thinning removes almost all degeneracies while retaining convex
    # Voronoi cells.  Collapse only the small residual set of bad interfaces;
    # keep river cells separate from the surrounding 2-D surface.
    short_face_merges = 0
    for _ in range(config.quality_repair_max_iterations):
        repair_qa = _edge_quality(
            polygons, points, target, transform,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
        )
        if not len(repair_qa["bad_owners"]):
            break
        merge_groups = ["river" if item == "river" else "surface" for item in feature_classes]
        polygons, merged = _agglomerate_short_face_cells(
            polygons,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            0.5 * config.minimum_hydraulic_width_m,
            max_merges=len(repair_qa["bad_owners"]),
            classes=merge_groups,
            candidate_pairs=repair_qa["bad_owners"],
        )
        if not merged:
            break
        short_face_merges += merged
        feature_classes = _feature_classes(polygons, river_geometry, floodplain_geometry, urban, transform)
    cross_feature_short_face_merges = 0
    for _ in range(config.quality_repair_max_iterations):
        repair_qa = _edge_quality(
            polygons, points, target, transform,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
        )
        if not len(repair_qa["bad_owners"]):
            break
        polygons, merged = _agglomerate_short_face_cells(
            polygons,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            0.5 * config.minimum_hydraulic_width_m,
            max_merges=len(repair_qa["bad_owners"]),
            candidate_pairs=repair_qa["bad_owners"],
        )
        if not merged:
            break
        cross_feature_short_face_merges += merged
        feature_classes = _feature_classes(polygons, river_geometry, floodplain_geometry, urban, transform)
    # Always retain the post-split candidate figure.  When the quality gate
    # rejects it, this is the evidence needed to diagnose the problematic
    # transition geometry; it is not a model-ready mesh artifact.
    final_figure = config.out_dir / "diagnostics" / "mesh_step6_final.png"
    _write_mesh_class_figure(
        final_figure,
        polygons, feature_classes, crs, river_geometry, floodplain_geometry,
        "Final Pune adaptive Voronoi mesh",
    )
    if config.diagnostic_window is not None:
        diagnostic = gpd.read_file(config.diagnostic_window).to_crs(crs).geometry.union_all()
        detail_indices = [index for index, polygon in enumerate(polygons) if polygon.intersects(diagnostic)]
        _write_mesh_class_figure(
            config.out_dir / "diagnostics" / "mesh_step6_final_detail.png",
            [polygons[index] for index in detail_indices], [feature_classes[index] for index in detail_indices], crs,
            river_geometry.intersection(diagnostic), floodplain_geometry.intersection(diagnostic),
            "Final adaptive mesh — river, floodplain, urban, and rural cells", diagnostic.bounds,
        )
    if config.diagnostic_windows is not None:
        windows = gpd.read_file(config.diagnostic_windows).to_crs(crs)
        for index, geometry in enumerate(windows.geometry, start=1):
            detail_indices = [item for item, polygon in enumerate(polygons) if polygon.intersects(geometry)]
            _write_mesh_class_figure(
                config.out_dir / "diagnostics" / f"mesh_step6_final_region_{index:02d}.png",
                [polygons[item] for item in detail_indices], [feature_classes[item] for item in detail_indices], crs,
                river_geometry.intersection(geometry), floodplain_geometry.intersection(geometry),
                f"Final adaptive mesh — region {index}", geometry.bounds,
            )
    preflight_edge_qa = _edge_quality(
        polygons, points, target, transform,
        config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
        config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
    )
    polygon_objects = np.asarray(polygons, dtype=object)
    preflight_scale = 2.0 * np.asarray([item.area for item in polygon_objects]) / np.maximum(
        np.asarray([item.length for item in polygon_objects]), 1e-12,
    )
    if preflight_scale.min() < 0.5 * config.minimum_hydraulic_width_m:
        bad = np.flatnonzero(preflight_scale < 0.5 * config.minimum_hydraulic_width_m)
        bad_path = config.out_dir / "diagnostics" / "mesh_step6_bad_control_volumes.gpkg"
        gpd.GeoDataFrame(
            {
                "face_id": bad,
                "cfl_width_m": preflight_scale[bad],
                "feature_class": [feature_classes[index] for index in bad],
            },
            geometry=[polygons[index] for index in bad], crs=crs,
        ).to_file(bad_path, layer="bad_control_volumes", driver="GPKG")
        raise ValueError(
            "Mesh quality failed: minimum hydraulic cell scale is "
            f"{preflight_scale.min():.3f} m, below the "
            f"{0.5 * config.minimum_hydraulic_width_m:.3f} m floor. "
            f"See {bad_path} for the {len(bad)} rejected control volumes."
        )
    if preflight_edge_qa["short_internal_face_count"] or preflight_edge_qa["close_internal_center_count"]:
        bad = np.unique(preflight_edge_qa["bad_owners"])
        bad_path = config.out_dir / "diagnostics" / "mesh_step6_bad_faces.gpkg"
        gpd.GeoDataFrame(
            {
                "face_id": bad,
                "feature_class": [feature_classes[index] for index in bad],
            },
            geometry=[polygons[index] for index in bad], crs=crs,
        ).to_file(bad_path, layer="bad_faces", driver="GPKG")
        rejected_path = config.out_dir / "reports" / "hybrid_mesh_qa_preflight.json"
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(json.dumps({
            "status": "step6_preflight_rejected",
            "cells": len(polygons),
            "generation_time_s": perf_counter() - started_at,
            "quality_repair_iterations": quality_repair_iterations,
            "short_face_agglomerations": short_face_merges,
            "cross_feature_short_face_agglomerations": cross_feature_short_face_merges,
            **{key: value for key, value in preflight_edge_qa.items() if key not in {"bad_owners", "short_owners", "face_seed", "face_target_m"}},
        }, indent=2), encoding="utf-8")
        raise ValueError(
            "Mesh quality failed after local repair: "
            f"{preflight_edge_qa['short_internal_face_count']} internal faces are shorter than "
            f"{preflight_edge_qa['minimum_face_length_threshold_m']:.3f} m and "
            f"{preflight_edge_qa['close_internal_center_count']} centre pairs are closer than "
            f"{preflight_edge_qa['minimum_center_distance_threshold_m']:.3f} m. See {bad_path}."
        )
    if config.stop_after_step == 6:
        qa_path = config.out_dir / "reports" / "hybrid_mesh_qa_preflight.json"
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(json.dumps({
            "status": "step6_preflight_passed",
            "cells": len(polygons),
            "generation_time_s": perf_counter() - started_at,
            "quality_repair_iterations": quality_repair_iterations,
            "short_face_agglomerations": short_face_merges,
            "cross_feature_short_face_agglomerations": cross_feature_short_face_merges,
            "minimum_cfl_length_m": float(preflight_scale.min()),
            "p05_cfl_length_m": float(np.quantile(preflight_scale, 0.05)),
            **{key: value for key, value in preflight_edge_qa.items() if key not in {"bad_owners", "short_owners", "face_seed", "face_target_m"}},
        }, indent=2), encoding="utf-8")
        return {
            "status": "step6_preflight_passed",
            "cells": len(polygons),
            "qa": str(qa_path),
            "step6_figure": str(final_figure),
        }
    faces, edges = _topology(polygons)
    polygons = [Polygon(face) for face in faces]
    gpkg = config.out_dir / "mesh" / "hydropol_hybrid_mesh.gpkg"
    ugrid = config.out_dir / "mesh" / "hydropol_hybrid_mesh.nc"
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    river_fraction = [item.intersection(river_geometry).area / item.area if not river_geometry.is_empty else 0.0 for item in polygons]
    floodplain_fraction = [item.intersection(floodplain_geometry).area / item.area if not floodplain_geometry.is_empty else 0.0 for item in polygons]
    gpd.GeoDataFrame(
        {"face_id": range(len(polygons)), "area_m2": [item.area for item in polygons], "feature_class": feature_classes,
         "river_fraction": river_fraction, "floodplain_fraction": floodplain_fraction},
        geometry=polygons, crs=crs,
    ).to_file(gpkg, layer="mesh", driver="GPKG")
    _write_ugrid(ugrid, faces, edges, polygons, crs, target, transform, dem, feature_classes, river_geometry, floodplain_geometry)
    degree = np.zeros(len(polygons), dtype=int)
    for owners in edges.values():
        for item in owners:
            degree[item] += 1
    perimeters = np.asarray([item.length for item in polygons])
    cfl_width = 2.0 * np.asarray([item.area for item in polygons]) / np.maximum(perimeters, 1e-12)
    edge_qa = _edge_quality(
        polygons, points, target, transform,
        config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
        config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
    )
    qa = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, "cells": len(polygons), "edges": len(edges), "generation_time_s": perf_counter() - started_at, "mean_cell_degree": float(degree.mean()), "rural_quad_fraction": float(np.mean(degree == 4)), "minimum_cfl_length_m": float(cfl_width.min()), "p05_cfl_length_m": float(np.quantile(cfl_width, 0.05)), "river_cells": int(river.sum()), "resolved_2d_river_cells": int(resolved_river.sum()), "neal_subgrid_river_cells": int((river & ~resolved_river).sum()), "urban_cells": int(urban.sum()), "floodplain_cells": int(floodplain.sum()), "river_feature_seed_count": int(len(river_ribbon)), "floodplain_feature_seed_count": int(len(floodplain_ribbon)), "boundary_support_seed_count": int(len(support)), "reserved_feature_cells": int(reservation.sum()), "quality_repair_iterations": quality_repair_iterations, "short_face_agglomerations": short_face_merges, "cross_feature_short_face_agglomerations": cross_feature_short_face_merges, **{key: value for key, value in edge_qa.items() if key not in {"bad_owners", "short_owners", "face_seed", "face_target_m"}}}
    qa_path = config.out_dir / "reports" / "hybrid_mesh_qa.json"; qa_path.parent.mkdir(parents=True, exist_ok=True); qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    _write_mesh_class_figure(
        config.out_dir / "diagnostics" / "hybrid_mesh_wireframe.png",
        polygons, feature_classes, crs, river_geometry, floodplain_geometry,
        "Step 6 — feature-boundary split and sliver repair",
    )
    return {"ugrid": str(ugrid), "geopackage": str(gpkg), "qa": str(qa_path), **qa}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a parameterized hybrid quad/Voronoi HydroPol mesh.")
    parser.add_argument("--config", required=True, type=Path, help="JSON/TOML hybrid-mesh configuration.")
    args = parser.parse_args(argv)
    print(json.dumps(build_hybrid_mesh(HybridMeshConfig.from_mapping(load_config_file(args.config))), indent=2))


if __name__ == "__main__":
    main()
