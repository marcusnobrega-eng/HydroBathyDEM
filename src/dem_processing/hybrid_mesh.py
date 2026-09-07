"""Feature-aware hybrid quadrilateral/Voronoi mesh generation.

Regular seed lattices give four-sided Voronoi cells in coherent background
patches.  Feature and transition seeds are left unstructured, yielding one
conforming polygon mesh without user-drawn refinement regions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import xy
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely import (
    MultiPoint, STRtree, box, centroid, contains, difference, get_coordinates,
    get_num_coordinates, get_x, get_y, intersection, is_valid,
    linestrings as shapely_linestrings, make_valid, point_on_surface, prepare,
    points as shapely_points, shortest_line, unary_union, union, voronoi_polygons,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

from .condition_dem import compute_d4_flow_accumulation
from .config import load_config_file
from .mesh_contract import (
    EDGE_NORMAL_CONVENTION,
    MESH_PRODUCT_SCHEMA,
    apply_contract_attributes,
)


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
    floodplain_enabled: bool = True
    floodplain_mask: Path | None = None
    floodplain_vector: Path | None = None
    floodplain_direction: Path | None = None
    floodplain_axis: Path | None = None
    waterbody_vector: Path | None = None
    waterbody_layer: str | None = None
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
    waterbody_target_width_m: float = 120.0
    minimum_hydraulic_width_m: float = 30.0
    maximum_adjacent_size_ratio: float = 2.0
    enforce_feature_boundaries: bool = True
    enforce_waterbody_boundaries: bool = False
    impervious_threshold_percent: float = 1.0
    population_threshold_per_km2: float = 1000.0
    urban_buffer_m: float = 300.0
    floodplain_hand_stage_m: float = 10.0
    minimum_upstream_area_km2: float = 1.0
    river_source: str = "derive_d4"
    # ``structured_strip`` builds river cells directly from the centreline and
    # mapped width, so cross-stream size is prescribed rather than emerging from
    # a Voronoi diagram.  ``voronoi`` keeps the previous behaviour for cases
    # that still need it.
    # ``graded_rows`` seeds the channel in lanes and steps the cell size up on both
    # banks, then builds one Voronoi over everything -- the HEC-RAS approach.  It
    # replaces ``structured_strip``, which built the ribbon separately and punched
    # it into the background; that cutting cost 26 of the 30 minute build and left
    # a 6.67 size jump where a 30 m channel cell met a 200 m rural cell.
    river_cell_style: str = "graded_rows"
    river_centerline_smoothing_iterations: int = 1
    river_width_smoothing_window_cells: int = 1
    unresolved_policy: str = "neal_subgrid"
    minimum_face_length_factor: float = 0.25
    minimum_center_distance_factor: float = 0.5
    quality_repair_max_iterations: int = 6
    write_diagnostics: bool = True
    # Diagnostic option: write the globally conforming Step 5 Voronoi only.
    # It is deliberately not a production mesh because hard feature-boundary
    # splitting and all topology/quality gates have not yet been applied.
    stop_after_step: int | None = None
    max_seed_count: int = 100_000
    random_seed: int = 7
    enforce_river_boundaries: bool = False

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
            "floodplain": {},
            "urban": {},
            "quality": {},
        }
        for group in grouped:
            group_values = values.pop(group, {})
            if group_values:
                if not isinstance(group_values, dict):
                    raise ValueError(f"{group} must be a configuration object.")
                grouped[group].update(group_values)
        if "enabled" in grouped["floodplain"]:
            grouped["floodplain"]["floodplain_enabled"] = grouped["floodplain"].pop("enabled")
        values = {
            **grouped["inputs"],
            **grouped["resolution"],
            **grouped["rivers"],
            **grouped["floodplain"],
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
            "waterbody_vector_path": "waterbody_vector",
            "source": "river_source",
            "initiation_upstream_area_km2": "minimum_upstream_area_km2",
            "cell_style": "river_cell_style",
            "centerline_smoothing_iterations": "river_centerline_smoothing_iterations",
            "width_smoothing_window_cells": "river_width_smoothing_window_cells",
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
        values.pop("preferred_cells_across", None)
        values.pop("river_preferred_cells_across", None)
        for key in ("dem", "out_dir", "impervious", "population", "river_mask", "river_direction", "river_width", "river_depth", "river_bank_height", "floodplain_mask", "floodplain_vector", "floodplain_direction", "floodplain_axis", "waterbody_vector", "river_network", "domain_vector", "diagnostic_window", "diagnostic_windows"):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        if config.unresolved_policy not in {"neal_subgrid", "none"}:
            raise ValueError("rivers.unresolved_policy must be 'neal_subgrid' or 'none'.")
        if config.river_source not in {"derive_d4", "hydrobathydem_d4", "hydrobathydem_d8"}:
            raise ValueError(
                "rivers.source must be 'derive_d4', 'hydrobathydem_d4', or 'hydrobathydem_d8'."
            )
        if config.river_cell_style not in {"graded_rows", "structured_strip", "voronoi"}:
            raise ValueError(
                "rivers.cell_style must be 'graded_rows', 'structured_strip', or 'voronoi'."
            )
        if config.stop_after_step not in {None, 5, 6}:
            raise ValueError("quality.stop_after_step may be omitted or set to 5 or 6.")
        if config.minimum_upstream_area_km2 <= 0:
            raise ValueError("rivers.initiation_upstream_area_km2 must be positive.")
        if config.river_centerline_smoothing_iterations < 0:
            raise ValueError("rivers.centerline_smoothing_iterations must be non-negative.")
        if config.river_width_smoothing_window_cells < 1:
            raise ValueError("rivers.width_smoothing_window_cells must be at least 1.")
        if config.minimum_face_length_factor <= 0 or config.minimum_center_distance_factor <= 0:
            raise ValueError("quality factors must be positive.")
        if min(
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            config.floodplain_target_width_m,
            config.floodplain_along_river_cell_length_m,
            config.floodplain_cross_river_target_width_m,
            config.waterbody_target_width_m,
        ) <= 0:
            raise ValueError("all river, floodplain, and waterbody targets must be positive.")
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


def receiver_from_d8_direction(direction: np.ndarray) -> np.ndarray:
    """Convert 1=N, 2=NE, 3=E, ..., 8=NW codes to flattened receivers."""
    rows, cols = direction.shape
    receiver = np.full(direction.shape, -1, dtype=np.int64)
    index = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    links = {
        1: (-1, 0), 2: (-1, 1), 3: (0, 1), 4: (1, 1),
        5: (1, 0), 6: (1, -1), 7: (0, -1), 8: (-1, -1),
    }
    for code, (drow, dcol) in links.items():
        row_source = slice(max(0, -drow), min(rows, rows - drow))
        col_source = slice(max(0, -dcol), min(cols, cols - dcol))
        row_target = slice(max(0, drow), min(rows, rows + drow))
        col_target = slice(max(0, dcol), min(cols, cols + dcol))
        selected = direction[row_source, col_source] == code
        receiver[row_source, col_source][selected] = index[row_target, col_target][selected]
    return receiver


def _needs_d4_routing(config: HybridMeshConfig, has_supplied_direction: bool) -> bool:
    return config.river_source == "derive_d4" or not has_supplied_direction


def _needs_reach_labels(config: HybridMeshConfig) -> bool:
    return (
        config.river_network is not None and config.river_source == "derive_d4"
    ) or (
        config.floodplain_enabled
        and (
            (config.floodplain_vector is None and config.floodplain_mask is None)
            or (config.floodplain_align_to_flow and config.floodplain_direction is None)
        )
    )


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


def exclusive_floodplain_geometry(candidate, river_geometry, waterbody_geometry):
    """Floodplain is the candidate minus the channel and minus standing water.

    Priority is waterbody, then river channel, then floodplain, so a cell can
    never be floodplain and river at once and never floodplain inside a
    reservoir.  Keeping the rule in one place makes it checkable rather than an
    incidental property of the order the geometries happen to be built in.
    """
    if candidate is None or candidate.is_empty:
        return Polygon()
    blockers = [item for item in (river_geometry, waterbody_geometry) if item is not None and not item.is_empty]
    if not blockers:
        return candidate.buffer(0)
    return candidate.difference(unary_union(blockers)).buffer(0)


def _mask_geometry(mask: np.ndarray, transform: Any) -> Polygon:
    """Polygonise a raster feature without changing its selected cells."""
    pieces = [
        Polygon(geometry["coordinates"][0])
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform)
        if value
    ]
    geometry = unary_union(pieces) if pieces else Polygon()
    return geometry if geometry.geom_type == "Polygon" else geometry


def _vector_feature_geometry(
    path: Path | None, crs: Any, domain, layer: str | None = None,
):
    if path is None:
        return Polygon()
    frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if frame.empty:
        return Polygon()
    return frame.to_crs(crs).geometry.union_all().intersection(domain).buffer(0)


def _smooth_width_values(widths: np.ndarray, window_cells: int) -> np.ndarray:
    values = np.asarray(widths, dtype=np.float64)
    if values.size == 0:
        return values
    finite = np.isfinite(values) & (values > 0)
    if not finite.any():
        return values
    values = np.where(finite, values, float(np.nanmedian(values[finite])))
    window = min(max(1, int(window_cells)), values.size)
    if window <= 1:
        return values
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.full(window, 1.0 / window), mode="valid")


def _river_segment_records(
    river: np.ndarray,
    receiver: np.ndarray,
    transform: Any,
    width: np.ndarray,
    centerline_smoothing_iterations: int = 1,
    width_smoothing_window_cells: int = 1,
) -> list[tuple[int, LineString, float, tuple[float, float]]]:
    """Build smoothed, directed reaches from an orthogonal D4 river network.

    D4 is retained for hydrologic connectivity, not used as the visible river
    geometry.  Each between-junction reach is smoothed once with a
    corner-cutting pass (fixed endpoints), then resampled with its original
    mapped widths.  This removes artificial 90-degree mesh bends while keeping
    every D4 link represented by a directed physical channel.
    """
    ncols = river.shape[1]
    paths, leftover = _river_reach_paths(river, receiver)
    records: list[tuple[int, LineString, float, tuple[float, float]]] = []
    for path in paths:
        centres = np.asarray([xy(transform, *divmod(item, ncols), offset="center") for item in path], dtype=np.float64)
        smoothed = _chaikin_smoothed(centres, centerline_smoothing_iterations)
        source_widths = _smooth_width_values(
            np.asarray([
                width[row, col] if np.isfinite(width[row, col]) else np.nan
                for row, col in (divmod(item, ncols) for item in path[:-1])
            ], dtype=np.float64),
            width_smoothing_window_cells,
        )
        n_segments = max(1, len(smoothed) - 1)
        for segment_id, (left, right) in enumerate(zip(smoothed[:-1], smoothed[1:], strict=True)):
            width_index = min(len(source_widths) - 1, int(segment_id * len(source_widths) / n_segments))
            source_index = path[width_index]
            mapped_width = float(source_widths[width_index]) if np.isfinite(source_widths[width_index]) else 0.0
            dx, dy = right - left
            length = float(np.hypot(dx, dy))
            if mapped_width > 0 and length > 0:
                records.append((source_index, LineString((left, right)), mapped_width, (dx / length, dy / length)))

    # A simple one-link reach at a cut domain edge can be missed above.  Add it
    # unchanged rather than silently dropping a mapped river.
    for start, target in leftover:
        row, col = divmod(start, ncols)
        drow, dcol = divmod(target, ncols)
        x0, y0 = xy(transform, row, col, offset="center")
        x1, y1 = xy(transform, drow, dcol, offset="center")
        length = float(np.hypot(x1 - x0, y1 - y0))
        mapped_width = float(width[row, col]) if np.isfinite(width[row, col]) else 0.0
        if mapped_width > 0 and length > 0:
            records.append((start, LineString(((x0, y0), (x1, y1))), mapped_width, ((x1 - x0) / length, (y1 - y0) / length)))
    return records


def _river_reach_paths(river: np.ndarray, receiver: np.ndarray) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Split a directed river raster into junction-to-junction index paths.

    A reach ends wherever the network branches or joins, so every path is a
    single-thread channel that can carry one cross-section series.  The second
    return value holds links that no traversal reached, which happens for a
    one-link reach cut by the domain edge.
    """
    river_indices = np.flatnonzero(river.ravel())
    downstream: dict[int, int] = {}
    indegree = {int(item): 0 for item in river_indices}
    for item in river_indices:
        target = int(receiver.ravel()[item])
        if target in indegree:
            downstream[int(item)] = target
            indegree[target] += 1
    visited_links: set[tuple[int, int]] = set()
    paths: list[list[int]] = []
    for initial in (item for item in river_indices if indegree[int(item)] != 1):
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
        if len(path) >= 2:
            paths.append(path)
    leftover = [link for link in downstream.items() if link not in visited_links]
    return paths, leftover


def _chaikin_smoothed(points: np.ndarray, iterations: int) -> np.ndarray:
    """Corner-cut a polyline with fixed endpoints, removing D8 stair steps."""
    current = np.asarray(points, dtype=np.float64)
    for _ in range(max(0, int(iterations)) if len(current) > 2 else 0):
        result = [current[0]]
        for left, right in zip(current[:-1], current[1:], strict=True):
            result.append(0.75 * left + 0.25 * right)
            result.append(0.25 * left + 0.75 * right)
        result.append(current[-1])
        current = np.asarray(result, dtype=np.float64)
    return current


def _river_reach_records(
    river: np.ndarray,
    receiver: np.ndarray,
    transform: Any,
    width: np.ndarray,
    centerline_smoothing_iterations: int = 1,
    width_smoothing_window_cells: int = 1,
) -> list[tuple[int, LineString, np.ndarray]]:
    """Return whole reaches as (reach_id, smoothed centreline, per-vertex width).

    ``_river_segment_records`` deliberately explodes the same network into
    single links because the Voronoi seeder consumes one tangent at a time.  A
    structured strip instead needs the continuous reach so its cross-sections
    stay parallel and its cells stay watertight across a bend.
    """
    ncols = river.shape[1]
    paths, leftover = _river_reach_paths(river, receiver)
    for start, target in leftover:
        paths.append([start, target])
    records: list[tuple[int, LineString, np.ndarray]] = []
    for reach_id, path in enumerate(paths):
        centres = np.asarray(
            [xy(transform, *divmod(item, ncols), offset="center") for item in path], dtype=np.float64,
        )
        widths = _smooth_width_values(
            np.asarray(
                [width[row, col] if np.isfinite(width[row, col]) else np.nan
                 for row, col in (divmod(item, ncols) for item in path)],
                dtype=np.float64,
            ),
            width_smoothing_window_cells,
        )
        # Chaikin subdivision multiplies vertices, so carry the mapped width by
        # normalised arc position rather than by vertex index.
        source_position = _normalised_arc_position(centres)
        smoothed = _chaikin_smoothed(centres, centerline_smoothing_iterations)
        if len(smoothed) < 2:
            continue
        line = LineString(smoothed)
        if line.length <= 0:
            continue
        smoothed_widths = np.interp(_normalised_arc_position(smoothed), source_position, widths)
        records.append((reach_id, line, smoothed_widths))
    return records


def _normalised_arc_position(points: np.ndarray) -> np.ndarray:
    """Return each vertex's cumulative distance along a polyline, scaled to 0-1."""
    steps = np.hypot(*np.diff(points, axis=0).T)
    cumulative = np.r_[0.0, np.cumsum(steps)]
    total = cumulative[-1]
    return cumulative / total if total > 0 else np.linspace(0.0, 1.0, len(points))


def _station_normals(stations: np.ndarray) -> np.ndarray:
    """Unit cross-stream normals from central-difference tangents."""
    tangents = np.empty_like(stations)
    tangents[1:-1] = stations[2:] - stations[:-2]
    tangents[0] = stations[1] - stations[0]
    tangents[-1] = stations[-1] - stations[-2]
    scale = np.maximum(np.hypot(tangents[:, 0], tangents[:, 1]), 1e-12)
    tangents /= scale[:, None]
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def _fold_limited_half_width(
    stations: np.ndarray, normals: np.ndarray, half_width: np.ndarray, safety: float = 0.5,
) -> np.ndarray:
    """Cap the offset where two consecutive cross-sections would cross.

    A normal-offset ribbon is single valued only while the half width stays
    inside the point at which neighbouring cross-sections meet; past it the
    inner bank folds through the centreline and the two cells overlap.  Solving
    ``P_i + s1 n_i == P_i+1 + s2 n_i+1`` gives that point exactly, so the cap is
    a statement about what a centreline plus a width can represent rather than a
    quality fudge.  The clamped width is exported per cell, which keeps a
    genuinely over-wide bend visible instead of silently thinned.
    """
    limit = np.full(len(stations), np.inf)
    if len(stations) < 2:
        return np.minimum(half_width, limit)
    delta = stations[1:] - stations[:-1]
    lead, follow = normals[:-1], normals[1:]
    turn = lead[:, 0] * follow[:, 1] - lead[:, 1] * follow[:, 0]
    parallel = np.abs(turn) < 1e-12
    safe_turn = np.where(parallel, 1.0, turn)
    lead_offset = (delta[:, 0] * follow[:, 1] - delta[:, 1] * follow[:, 0]) / safe_turn
    follow_offset = (delta[:, 0] * lead[:, 1] - delta[:, 1] * lead[:, 0]) / safe_turn
    lead_limit = np.where(parallel, np.inf, safety * np.abs(lead_offset))
    follow_limit = np.where(parallel, np.inf, safety * np.abs(follow_offset))
    limit[:-1] = np.minimum(limit[:-1], lead_limit)
    limit[1:] = np.minimum(limit[1:], follow_limit)
    return np.minimum(half_width, limit)


def build_river_strip_cells(
    centerline: LineString,
    widths: np.ndarray,
    along_m: float = 30.0,
    cross_target_m: float = 30.0,
    minimum_width_m: float = 30.0,
    reach_id: int = 0,
    minimum_cell_scale_m: float = 0.0,
) -> tuple[list[Polygon], list[dict[str, Any]], LineString, LineString]:
    """Build the structured cell ribbon for one reach plus its two bank lines.

    Cross-stream size is set by the mapped channel width and along-stream size
    by the station spacing, so neither is an emergent property of a Voronoi
    diagram.  Adjacent cells share their station offset points exactly, which
    makes the ribbon watertight by construction rather than by later repair.

    ``minimum_cell_scale_m`` guards a collision that is easy to miss.  The
    quality floor is on 2A/P, which for a w-by-l cell is w*l/(w+l); a 30 m
    square gives exactly 15.0, which is exactly the floor for a 30 m minimum
    width.  The default configuration therefore sits precisely on the threshold,
    and on a curved reach the chord between two stations is slightly shorter
    than the arc, so cells land just under it -- 125 of the 165 cells the Pune
    gate rejected were between 14.0 and 15.0.  Consecutive cells in a band are
    merged until they clear the floor, which lengthens a few cells rather than
    leaving them to be rejected or dissolved into the floodplain.
    """
    if centerline.length <= 0:
        return [], [], LineString(), LineString()
    length = centerline.length
    # Derive the along-flow step from the narrowest band this reach will carry,
    # not from ``along_m`` alone.  The quality floor is on 2A/P, which for a
    # w-by-l cell is w*l/(w+l), so clearing a floor f needs l > f*w/(w-f):
    # a 30 m band needs more than 30 m, a 27 m band more than 33.8 m, a 25 m band
    # more than 37.5 m.  Asking for 30 m everywhere therefore puts every cell on
    # or under the threshold, and they all get merged away to twice the length --
    # which is how a 30 m request silently became a 60 m cell.  Sizing from the
    # band makes the request honest and leaves the merge for genuine outliers.
    raw_width = np.asarray(widths, dtype=np.float64).ravel()
    raw_width = raw_width[np.isfinite(raw_width)]
    raw_width = np.maximum(
        raw_width if raw_width.size else np.full(1, minimum_width_m), minimum_width_m,
    )
    band_count = np.maximum(1, np.rint(raw_width / cross_target_m))
    band_floor = float((raw_width / band_count).min())
    required_along = along_m
    if minimum_cell_scale_m > 0 and band_floor > minimum_cell_scale_m:
        # 8% over the analytic minimum: the chord between two stations is a
        # little shorter than the arc on a curved reach, and without the margin
        # every bend lands back under the floor.
        required_along = 1.08 * minimum_cell_scale_m * band_floor / (band_floor - minimum_cell_scale_m)
    n_along = max(1, int(length // max(along_m, required_along)))
    station_distance = np.linspace(0.0, length, n_along + 1)
    stations = np.asarray(
        [[point.x, point.y] for point in (centerline.interpolate(item) for item in station_distance)],
        dtype=np.float64,
    )
    # ``widths`` normally arrives one value per centreline vertex.  Accept any
    # other count as an evenly sampled series along the reach so a caller can
    # pass a coarser mapped-width profile without silently misaligning it.
    values = np.asarray(widths, dtype=np.float64).ravel()
    coordinates = np.asarray(centerline.coords, dtype=np.float64)
    if values.size == 0:
        values = np.full(2, minimum_width_m)
    if values.size == 1:
        values = np.repeat(values, 2)
    source_position = (
        _normalised_arc_position(coordinates) if values.size == len(coordinates)
        else np.linspace(0.0, 1.0, values.size)
    )
    station_width = np.interp(station_distance / length, source_position, values)
    station_width = np.maximum(np.nan_to_num(station_width, nan=minimum_width_m), minimum_width_m)
    normals = _station_normals(stations)
    half_width = _fold_limited_half_width(stations, normals, 0.5 * station_width)
    left_bank = stations + half_width[:, None] * normals
    right_bank = stations - half_width[:, None] * normals
    cells: list[Polygon] = []
    attributes: list[dict[str, Any]] = []
    band_cells: dict[int, list[Polygon]] = {}
    band_attributes: dict[int, list[dict[str, Any]]] = {}
    for index in range(n_along):
        interval_width = 0.5 * (2.0 * half_width[index] + 2.0 * half_width[index + 1])
        bands = max(1, int(round(interval_width / cross_target_m)))
        for band in range(bands):
            lower = -1.0 + 2.0 * band / bands
            upper = -1.0 + 2.0 * (band + 1) / bands
            corners = [
                stations[index] + lower * half_width[index] * normals[index],
                stations[index] + upper * half_width[index] * normals[index],
                stations[index + 1] + upper * half_width[index + 1] * normals[index + 1],
                stations[index + 1] + lower * half_width[index + 1] * normals[index + 1],
            ]
            cell = Polygon(corners)
            if not cell.is_valid or cell.area <= 0:
                continue
            band_cells.setdefault(band, []).append(cell)
            band_attributes.setdefault(band, []).append({
                "reach_id": int(reach_id),
                "station_start_m": float(station_distance[index]),
                "station_end_m": float(station_distance[index + 1]),
                "cross_band": int(band),
                "cross_band_count": int(bands),
                "channel_width_m": float(half_width[index] + half_width[index + 1]),
                "mapped_width_m": float(0.5 * (station_width[index] + station_width[index + 1])),
            })
    for band in sorted(band_cells):
        emitted: list[int] = []
        pending = None
        pending_attribute = None
        for cell, attribute in zip(band_cells[band], band_attributes[band], strict=True):
            if pending is None:
                pending, pending_attribute = cell, dict(attribute)
            else:
                union = unary_union((pending, cell))
                if union.geom_type != "Polygon":
                    emitted.append(len(cells))
                    cells.append(pending)
                    attributes.append(pending_attribute)
                    pending, pending_attribute = cell, dict(attribute)
                else:
                    pending = union
                    pending_attribute["station_end_m"] = attribute["station_end_m"]
                    pending_attribute["channel_width_m"] = max(
                        pending_attribute["channel_width_m"], attribute["channel_width_m"],
                    )
            if 2.0 * pending.area / max(pending.length, 1e-12) >= minimum_cell_scale_m:
                emitted.append(len(cells))
                cells.append(pending)
                attributes.append(pending_attribute)
                pending, pending_attribute = None, None
        if pending is None or pending_attribute is None:
            continue
        # A band can end short.  Fold the tail back into the previous cell of the
        # same band rather than exporting it undersized.
        union = unary_union((cells[emitted[-1]], pending)) if emitted else None
        if union is not None and union.geom_type == "Polygon":
            cells[emitted[-1]] = union
            attributes[emitted[-1]]["station_end_m"] = pending_attribute["station_end_m"]
        else:
            cells.append(pending)
            attributes.append(pending_attribute)
    return cells, attributes, LineString(left_bank), LineString(right_bank)

def _line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _line_parts(item)]
    return []


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for item in geometry.geoms for part in _polygon_parts(item)]
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


def _physical_river_geometry(
    records: list[tuple[int, LineString, float, tuple[float, float]]], minimum_width_m: float = 0.0,
):
    """Union mapped river footprints, optionally widened only for mesh hosts."""
    if not records:
        return Polygon()
    return unary_union([
        line.buffer(max(width, minimum_width_m) / 2.0, cap_style="flat", join_style="round")
        for _, line, width, _ in records
    ])


def _river_cross_layout(
    width: float, cross_target_m: float, minimum_seed_distance_m: float,
) -> tuple[np.ndarray, float]:
    if not np.isfinite(width) or width <= 0:
        return np.empty(0, dtype=np.float64), 0.0
    n_cross = max(1, int(np.ceil(width / cross_target_m)))
    if minimum_seed_distance_m > 0:
        n_cross = min(n_cross, max(1, int(np.floor(width / minimum_seed_distance_m))))
    actual_cross = width / n_cross
    offsets = -0.5 * width + (np.arange(n_cross, dtype=np.float64) + 0.5) * actual_cross
    return offsets, actual_cross


def _river_core_seeds(
    records: list[tuple[int, LineString, float, tuple[float, float]]],
    along_m: float,
    cross_target_m: float,
    minimum_width_m: float,
    minimum_seed_distance_m: float = 0.0,
) -> np.ndarray:
    """Place flow-aligned host-cell seeds across each represented river segment."""
    candidates: list[tuple[float, float]] = []
    for _, line, width, tangent in records:
        if width < minimum_width_m:
            continue  # retained as Neal subgrid, never widened for the mesh
        tx, ty = tangent
        nx, ny = -ty, tx
        n_along = max(1, int(np.ceil(line.length / along_m)))
        offsets, _ = _river_cross_layout(width, cross_target_m, minimum_seed_distance_m)
        for step in range(n_along):
            centre = line.interpolate((step + 0.5) * line.length / n_along)
            for offset in offsets:
                candidates.append((centre.x + offset * nx, centre.y + offset * ny))
    return np.unique(np.asarray(candidates, dtype=np.float64), axis=0) if candidates else np.empty((0, 2), dtype=np.float64)


def _river_bank_support_seeds(
    records: list[tuple[int, LineString, float, tuple[float, float]]],
    along_m: float,
    cross_target_m: float,
    minimum_seed_distance_m: float,
) -> np.ndarray:
    """Place non-river support sites just outside both physical banks."""
    candidates: list[tuple[float, float]] = []
    for _, line, width, tangent in records:
        offsets, actual_cross = _river_cross_layout(width, cross_target_m, minimum_seed_distance_m)
        if not len(offsets):
            continue
        tx, ty = tangent
        nx, ny = -ty, tx
        n_along = max(1, int(np.ceil(line.length / along_m)))
        support_offset = max(0.5 * width + 0.5 * actual_cross, 0.5 * width + minimum_seed_distance_m)
        for step in range(n_along):
            centre = line.interpolate((step + 0.5) * line.length / n_along)
            candidates.append((centre.x - support_offset * nx, centre.y - support_offset * ny))
            candidates.append((centre.x + support_offset * nx, centre.y + support_offset * ny))
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


def _clip_cells_to_feature_polygon(polygons: list[Polygon], feature, minimum_piece_area_m2: float) -> list[Polygon]:
    """Make a polygon feature a real cell boundary without line-by-line splitting."""
    if feature is None or feature.is_empty:
        return list(polygons)
    feature = feature.buffer(0)
    candidates = set(STRtree(polygons).query(feature.boundary, predicate="intersects").tolist())
    result: list[Polygon] = []
    for index, polygon in enumerate(polygons):
        if index not in candidates:
            result.append(polygon)
            continue
        pieces = _polygon_parts(polygon.intersection(feature)) + _polygon_parts(polygon.difference(feature))
        if len(pieces) > 1 and min(piece.area for piece in pieces) >= minimum_piece_area_m2:
            result.extend(pieces)
        else:
            result.append(polygon)
    return result


def graded_row_plan(
    cross_target_m: float, maximum_target_m: float, size_ratio: float,
) -> list[float]:
    """Cell widths stepping up from the channel to the background, HEC-RAS style.

    Each row is at most ``size_ratio`` times the one inside it, so no two adjacent
    cells break the grading rule.  Removing the river from the background diagram
    is what created that problem: a 30 m channel cell ended up touching a 200 m
    rural cell directly, a jump of 6.67.  These rows are the missing middle.
    """
    ratio = max(float(size_ratio), 1.25)
    widths: list[float] = []
    width = float(cross_target_m)
    while width < maximum_target_m * (1.0 - 1e-9) and len(widths) < 12:
        width = min(width * ratio, float(maximum_target_m))
        widths.append(width)
    return widths


def river_graded_seeds(
    reach_records: list[tuple[int, LineString, np.ndarray]],
    along_m: float,
    cross_target_m: float,
    minimum_width_m: float,
    maximum_target_m: float,
    size_ratio: float,
    domain,
    waterbody_geometry=None,
    minimum_cell_scale_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Seed the channel in lanes and step the size up on both banks.

    Returns the in-channel seeds, the graded bank seeds, and how far the graded
    band reaches from the centreline.  Nothing is cut here: one Voronoi over these
    points plus the ordinary background lattice produces channel cells that already
    line up and a bank that already falls on a cell boundary.  That is the whole
    point of the rewrite -- the previous approach built the ribbon separately and
    then punched it into the background, and the repair afterwards cost 26 of the
    30 minutes.

    Offset rows can cross each other on a bend tighter than the offset.  Unlike the
    ribbon that is harmless here: a seed on the wrong side simply loses its cell to
    a closer neighbour, so the diagram stays valid.  Seeds that land inside the
    channel or a waterbody are dropped anyway.
    """
    plan = graded_row_plan(cross_target_m, maximum_target_m, size_ratio)
    channel: list[np.ndarray] = []
    graded: list[np.ndarray] = []
    reach_extent = 0.0
    for _, line, widths in reach_records:
        if line.length <= 0:
            continue
        values = np.asarray(widths, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        values = np.maximum(values if values.size else np.full(1, minimum_width_m), minimum_width_m)
        bands = np.maximum(1, np.rint(values / cross_target_m))
        band_floor = float((values / bands).min())
        step = along_m
        if minimum_cell_scale_m > 0 and band_floor > minimum_cell_scale_m:
            step = max(along_m, 1.08 * minimum_cell_scale_m * band_floor / (band_floor - minimum_cell_scale_m))
        n = max(1, int(line.length // step))
        distance = np.linspace(0.0, line.length, n + 1)
        stations = np.asarray(
            [[p.x, p.y] for p in (line.interpolate(item) for item in distance)], dtype=np.float64,
        )
        if len(stations) < 2:
            continue
        coordinates = np.asarray(line.coords, dtype=np.float64)
        source = (
            _normalised_arc_position(coordinates) if values.size == len(coordinates)
            else np.linspace(0.0, 1.0, values.size)
        )
        station_width = np.maximum(
            np.interp(distance / line.length, source, values), minimum_width_m,
        )
        normals = _station_normals(stations)
        half = _fold_limited_half_width(stations, normals, 0.5 * station_width)
        # one lane of seeds per band, at the lane centre
        n_bands = int(np.rint(np.median(station_width) / cross_target_m))
        n_bands = max(1, n_bands)
        for band in range(n_bands):
            fraction = -1.0 + (2.0 * band + 1.0) / n_bands
            channel.append(stations + (fraction * half)[:, None] * normals)
        # graded rows outside each bank, thinned along-flow to match their width
        offset = half.copy()
        previous = float(np.median(2.0 * half / n_bands))
        for width in plan:
            offset = offset + 0.5 * (previous + width)
            stride = max(1, int(round(width / max(step, 1e-9))))
            index = np.arange(0, len(stations), stride)
            for sign in (-1.0, 1.0):
                graded.append(stations[index] + (sign * offset[index])[:, None] * normals[index])
            previous = width
        reach_extent = max(reach_extent, float(offset.max()))
    def _clean(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty((0, 2), dtype=np.float64)
        points = np.unique(np.vstack(parts), axis=0)
        inside = np.asarray([domain.covers(Point(item)) for item in points], dtype=bool)
        points = points[inside]
        if waterbody_geometry is not None and not waterbody_geometry.is_empty and len(points):
            keep = ~np.asarray(
                [waterbody_geometry.covers(Point(item)) for item in points], dtype=bool,
            )
            points = points[keep]
        return points
    return _clean(channel), _clean(graded), reach_extent


def build_river_strip_mesh(
    reach_records: list[tuple[int, LineString, np.ndarray]],
    along_m: float,
    cross_target_m: float,
    minimum_width_m: float,
    domain,
    waterbody_geometry=None,
    minimum_cell_area_m2: float = 0.0,
    minimum_cell_scale_m: float = 0.0,
) -> tuple[list[Polygon], list[dict[str, Any]], list[LineString], float]:
    """Assemble every reach ribbon into one non-overlapping river cell set.

    Two reaches meeting at a confluence write overlapping cross-sections onto
    the same ground.  The wider reach is treated as the main stem and keeps its
    cells intact; the tributary yields the overlap.  Reservoir bodies are
    removed here rather than downstream so no river cell is ever created inside
    standing water.
    """
    cells: list[Polygon] = []
    attributes: list[dict[str, Any]] = []
    banks: list[LineString] = []

    def stem_rank(index: int) -> float:
        """Widest reach first, so the main stem keeps its cross-section whole."""
        widths = np.asarray(reach_records[index][2], dtype=np.float64)
        finite = widths[np.isfinite(widths)]
        return -float(finite.max()) if finite.size else 0.0

    order = sorted(range(len(reach_records)), key=stem_rank)
    for priority, record_index in enumerate(order):
        reach_id, line, widths = reach_records[record_index]
        reach_cells, reach_attributes, left, right = build_river_strip_cells(
            line, widths, along_m, cross_target_m, minimum_width_m, reach_id,
            minimum_cell_scale_m,
        )
        for attribute in reach_attributes:
            attribute["reach_priority"] = priority
        cells.extend(reach_cells)
        attributes.extend(reach_attributes)
        banks.extend(item for item in (left, right) if not item.is_empty)
    # The bank lines are meant to be usable as breaklines, so trim them to the
    # same extent as the cells they bound.  An untrimmed bank runs on through a
    # reservoir and would constrain a later mesh where no bank exists.
    #
    # Intersecting every bank against the catchment outline is far too slow -- it
    # is a complex polygon and there are two banks per reach.  Prepare it once
    # and let the cheap ``covers``/``intersects`` predicates decide which of the
    # few boundary-crossing banks actually need an overlay.
    prepare(domain)
    blocker = waterbody_geometry if waterbody_geometry is not None and not waterbody_geometry.is_empty else None
    if blocker is not None:
        prepare(blocker)
    trimmed: list[LineString] = []
    for line in banks:
        clipped = line if domain.covers(line) else line.intersection(domain)
        if blocker is not None and not clipped.is_empty and blocker.intersects(clipped):
            clipped = clipped.difference(blocker)
        trimmed.extend(part for part in _line_parts(clipped) if part.length > 0)
    banks = trimmed
    if not cells:
        return [], [], banks, 0.0
    # Confluence overlap: keep the higher-priority (wider) cell whole and trim
    # the lower-priority one.  Only junction neighbourhoods produce pairs, so
    # this stays a small local operation on a large network.
    priorities = np.asarray([item["reach_priority"] for item in attributes], dtype=np.int64)
    tree = STRtree(cells)
    cell_objects = np.asarray(cells, dtype=object)
    # ``overlaps`` covers a partial fold; ``contains`` covers the rarer case of
    # one cross-section swallowing another.  Neither fires for cells that only
    # share an edge, so an adjacent well-formed pair is never trimmed.
    pairs = np.hstack((
        tree.query(cell_objects, predicate="overlaps"),
        tree.query(cell_objects, predicate="contains"),
    ))
    trims: dict[int, list[int]] = {}
    for left_index, right_index in zip(pairs[0], pairs[1], strict=True):
        left_index, right_index = int(left_index), int(right_index)
        if left_index == right_index:
            continue
        loser, winner = (
            (left_index, right_index)
            if (priorities[left_index], left_index) > (priorities[right_index], right_index)
            else (right_index, left_index)
        )
        trims.setdefault(loser, []).append(winner)
    blocker = waterbody_geometry if waterbody_geometry is not None and not waterbody_geometry.is_empty else None
    result_cells: list[Polygon] = []
    result_attributes: list[dict[str, Any]] = []
    # A trim can break one cross-section cell into two disjoint pieces.  Only the
    # larger piece stays a river cell; the remainder is reported rather than
    # dropped quietly, because the channel polygon is built from what is kept and
    # the discarded area silently becomes floodplain or surface.
    discarded_area_m2 = 0.0
    for index, cell in enumerate(cells):
        original_area = cell.area
        if index in trims:
            cell = cell.difference(unary_union([cells[item] for item in trims[index]]))
        if blocker is not None:
            cell = cell.difference(blocker)
        if not cell.is_empty and not domain.covers(cell):
            cell = cell.intersection(domain)
        parts = [part for part in _polygon_parts(cell) if part.area > minimum_cell_area_m2]
        if not parts:
            continue
        keep = max(parts, key=lambda item: item.area)
        discarded_area_m2 += sum(part.area for part in parts) - keep.area
        result_cells.append(keep)
        result_attributes.append(attributes[index])
    return result_cells, result_attributes, banks, discarded_area_m2


def _ring_edges(polygons: list[Polygon]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-cell exterior-ring segments as (owner, local position, start, end).

    Interior rings are excluded on purpose.  ``_topology`` and the UGRID export
    both read only ``exterior``, so a cell that still carries a hole is a defect
    for ``hole_free_polygons`` to remove, not a source of finite-volume faces --
    and ``position`` has to index the exterior ring for the callers that rebuild
    a ring from it.
    """
    objects = np.asarray([polygon.exterior for polygon in polygons], dtype=object)
    counts = get_num_coordinates(objects)
    coordinates = get_coordinates(objects)
    if not len(coordinates):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty((0, 2)), np.empty((0, 2))
    first = np.r_[0, np.cumsum(counts[:-1])]
    owner_of_coordinate = np.repeat(np.arange(len(objects), dtype=np.int64), counts)
    is_edge_start = np.ones(len(coordinates), dtype=bool)
    is_edge_start[first + counts - 1] = False
    start = np.flatnonzero(is_edge_start)
    owner = owner_of_coordinate[start]
    return owner, start - first[owner], coordinates[start], coordinates[start + 1]


def _unpaired_ring_edges(
    owner: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray, decimals: int = 4,
) -> np.ndarray:
    """Indices of ring segments that no second cell currently matches."""
    if not len(owner):
        return np.empty(0, dtype=np.int64)
    swap = (start_xy[:, 0] > end_xy[:, 0]) | (
        (start_xy[:, 0] == end_xy[:, 0]) & (start_xy[:, 1] > end_xy[:, 1])
    )
    key = np.round(np.column_stack((
        np.where(swap, end_xy[:, 0], start_xy[:, 0]), np.where(swap, end_xy[:, 1], start_xy[:, 1]),
        np.where(swap, start_xy[:, 0], end_xy[:, 0]), np.where(swap, start_xy[:, 1], end_xy[:, 1]),
    )), decimals)
    order = np.lexsort((key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    sorted_key = key[order]
    group_start = np.r_[0, np.flatnonzero(np.any(sorted_key[1:] != sorted_key[:-1], axis=1)) + 1]
    group_end = np.r_[group_start[1:], len(sorted_key)]
    single = (group_end - group_start) == 1
    return order[group_start[single]]


def _despiked_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop the out-and-back spurs that welding two vertices together leaves.

    Collapsing one endpoint of a short segment onto the other can make a ring
    walk out to a vertex and straight back along the same line.  The ring then
    contains that segment twice, so ``_edge_quality`` pairs the cell with
    itself and no merge can repair it.  Removing the spur tip restores a simple
    ring and changes the cell area by nothing.
    """
    current = [point for index, point in enumerate(ring) if index == 0 or point != ring[index - 1]]
    if len(current) > 1 and current[0] == current[-1]:
        current.pop()
    while len(current) >= 3:
        count = len(current)
        tip = next(
            (
                index for index in range(count)
                if current[(index - 1) % count] == current[(index + 1) % count]
            ),
            None,
        )
        if tip is None:
            break
        del current[tip]
        current = [
            point for index, point in enumerate(current) if index == 0 or point != current[index - 1]
        ]
        if len(current) > 1 and current[0] == current[-1]:
            current.pop()
    return current


def _snapped_vertices(
    polygons: list[Polygon], snap: dict[tuple[float, float], tuple[float, float]],
) -> list[Polygon]:
    """Move a few vertices onto a neighbouring endpoint, removing a short face."""
    result: list[Polygon] = []
    for polygon in polygons:
        ring = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        if not any(vertex in snap for vertex in ring):
            result.append(polygon)
            continue
        cleaned = _despiked_ring([snap.get(vertex, vertex) for vertex in ring])
        candidate = Polygon(cleaned) if len(cleaned) >= 3 else None
        # GEOS calls a ring that walks out and back along the same line valid and
        # gives it a positive area of about a square millimetre.  Area alone
        # therefore does not detect a collapsed cell; the 2A/P scale does.
        collapsed = candidate is not None and (
            2.0 * candidate.area / max(candidate.length, 1e-12) < 1e-3
        )
        result.append(
            polygon
            if candidate is None or collapsed or not candidate.is_valid or candidate.area <= 0
            else candidate
        )
    return result


def insert_hanging_nodes(
    polygons: list[Polygon], tolerance: float = 1e-3, minimum_face_length_m: float = 0.0,
    max_passes: int = 6,
) -> tuple[list[Polygon], int, int]:
    """Make every internal interface a two-owner face, without creating slivers.

    Feature-boundary splitting, channel clipping, and sliver agglomeration all
    leave hanging nodes: cell A keeps one long edge where cells B and C each
    meet only half of it.  ``_topology`` pairs segments by identical endpoints,
    so all three stay unpaired and export as domain boundary -- a closed wall in
    the middle of the mesh, invisible to ``_edge_quality`` because that only
    inspects two-owner faces.

    Inserting the neighbouring vertex fixes the pairing but would cut a 30 m
    ribbon face into a 4 m and a 26 m face whenever the neighbour's vertex lands
    just past a station.  A node closer to an endpoint than
    ``minimum_face_length_m`` is therefore snapped onto that endpoint instead:
    the interface still pairs, and no face below the hydraulic floor is created.
    """
    result = list(polygons)
    inserted_total = 0
    snapped_total = 0
    for _ in range(max_passes):
        owner, position, start_xy, end_xy = _ring_edges(result)
        if not len(owner):
            break
        unpaired = _unpaired_ring_edges(owner, start_xy, end_xy)
        if not len(unpaired):
            break
        nodes = np.unique(np.round(np.vstack((start_xy, end_xy)), 4), axis=0)
        segment_start, segment_end = start_xy[unpaired], end_xy[unpaired]
        # A KD-tree over plain node coordinates costs tens of megabytes; an
        # STRtree over 2.5 M point *geometries* costs hundreds, and this loop
        # runs several times per build.  Each unpaired segment asks for the nodes
        # inside the disc that covers it, then the exact perpendicular test below
        # discards the corners of that disc.
        # Sample along each segment rather than putting one ball around its
        # midpoint.  A midpoint ball needs a radius of half the segment length, so
        # once agglomeration produces long edges the candidate set explodes -- a
        # 1 km edge pulls in every node within 500 m.  Fixed-radius balls spaced
        # along the segment cover the same neighbourhood in work proportional to
        # its length.
        span = np.hypot(
            segment_end[:, 0] - segment_start[:, 0], segment_end[:, 1] - segment_start[:, 1],
        )
        sample_step = max(4.0 * tolerance, 30.0)
        samples_per_segment = np.maximum(1, np.ceil(span / sample_step)).astype(np.int64)
        sample_owner = np.repeat(np.arange(len(unpaired)), samples_per_segment)
        offset = np.concatenate([
            (np.arange(count) + 0.5) / count for count in samples_per_segment
        ]) if len(samples_per_segment) else np.empty(0)
        sample_points = (
            segment_start[sample_owner]
            + offset[:, None] * (segment_end[sample_owner] - segment_start[sample_owner])
        )
        radius = 0.5 * span[sample_owner] / samples_per_segment[sample_owner] + tolerance
        candidates = cKDTree(nodes).query_ball_point(sample_points, radius, workers=-1)
        counts = np.fromiter((len(item) for item in candidates), dtype=np.int64, count=len(candidates))
        if not counts.sum():
            break
        segment_index = np.repeat(sample_owner, counts)
        node_index = np.fromiter(
            (item for group in candidates for item in group), dtype=np.int64, count=int(counts.sum()),
        )
        # One node can fall in two neighbouring balls on the same segment.
        segment_index, node_index = np.unique(
            np.column_stack((segment_index, node_index)), axis=0,
        ).T
        a = segment_start[segment_index]
        b = segment_end[segment_index]
        point = nodes[node_index]
        direction = b - a
        squared = np.einsum("ij,ij->i", direction, direction)
        parameter = np.divide(
            np.einsum("ij,ij->i", point - a, direction), squared,
            out=np.zeros(len(point)), where=squared > 0,
        )
        span = np.sqrt(np.maximum(squared, 1e-24))
        offset = np.abs(
            (point[:, 0] - a[:, 0]) * direction[:, 1] - (point[:, 1] - a[:, 1]) * direction[:, 0]
        ) / span
        margin = tolerance / span
        interior = (parameter > margin) & (parameter < 1.0 - margin) & (offset <= tolerance)
        if not np.any(interior):
            break
        segment_index = segment_index[interior]
        parameter = parameter[interior]
        point = point[interior]
        a, b, span = a[interior], b[interior], span[interior]
        # Decide insert-versus-snap per edge rather than per node.  Two
        # neighbouring vertices landing 3 m apart on the same 30 m ribbon edge
        # are each far from its endpoints, yet inserting both would still cut a
        # 3 m face between them.
        by_edge: dict[int, list[tuple[float, int]]] = {}
        for local, item in enumerate(segment_index):
            by_edge.setdefault(int(item), []).append((float(parameter[local]), local))
        snap: dict[tuple[float, float], tuple[float, float]] = {}
        keep = np.zeros(len(segment_index), dtype=bool)
        for item, entries in by_edge.items():
            entries.sort()
            edge_span = float(span[entries[0][1]])
            edge_start, edge_end = a[entries[0][1]], b[entries[0][1]]
            accepted_distance = 0.0
            accepted_point = (round(float(edge_start[0]), 4), round(float(edge_start[1]), 4))
            for value, local in entries:
                distance = value * edge_span
                source = (float(point[local][0]), float(point[local][1]))
                if edge_span - distance < minimum_face_length_m:
                    destination = (round(float(edge_end[0]), 4), round(float(edge_end[1]), 4))
                elif distance - accepted_distance < minimum_face_length_m:
                    destination = accepted_point
                else:
                    keep[local] = True
                    accepted_distance = distance
                    accepted_point = source
                    continue
                if source != destination:
                    snap.setdefault(source, destination)
        # Insertions first, and on their own pass.  Inserting a vertex does not
        # move any geometry, so it can never invalidate another edge, whereas a
        # snap can.  Deferring insertion behind snapping -- as an earlier version
        # did -- means one too-short candidate anywhere in a 300,000-cell mesh
        # preempts every insertion in that pass, so the pass budget is spent
        # moving tens of thousands of vertices instead of pairing faces for free.
        # Snapping is the fallback for the faces insertion cannot fix.
        if not np.any(keep):
            if not snap:
                break
            result = _snapped_vertices(result, snap)
            snapped_total += len(snap)
            continue
        insertions: dict[tuple[int, int], list[tuple[float, tuple[float, float]]]] = {}
        for item, value, vertex in zip(segment_index[keep], parameter[keep], point[keep], strict=True):
            edge = int(unpaired[int(item)])
            insertions.setdefault((int(owner[edge]), int(position[edge])), []).append(
                (float(value), (float(vertex[0]), float(vertex[1]))),
            )
        for cell_index in {key[0] for key in insertions}:
            ring = list(result[cell_index].exterior.coords)[:-1]
            rebuilt: list[tuple[float, float]] = []
            for local, vertex in enumerate(ring):
                rebuilt.append(vertex)
                extra = insertions.get((cell_index, local))
                if extra:
                    rebuilt.extend(item for _, item in sorted(extra))
                    inserted_total += len(extra)
            candidate = Polygon(rebuilt)
            if candidate.is_valid and candidate.area > 0:
                result[cell_index] = candidate
    return result, inserted_total, snapped_total


def _straight_cuts(polygon: Polygon, through) -> list[Polygon]:
    """Split a polygon with an axis-aligned line through a given point."""
    xmin, ymin, xmax, ymax = polygon.bounds
    span = max(xmax - xmin, ymax - ymin) + 1.0
    for cut in (
        LineString(((through.x, ymin - span), (through.x, ymax + span))),
        LineString(((xmin - span, through.y), (xmax + span, through.y))),
    ):
        try:
            parts = [
                part for part in split(polygon, cut).geoms
                if part.geom_type == "Polygon" and part.area > 0
            ]
        except Exception:
            parts = []
        if len(parts) > 1:
            return parts
    return [polygon]


def split_cells_without_interior_centroid(
    polygons: list[Polygon], max_cuts: int = 4,
) -> tuple[list[Polygon], int]:
    """Split any cell whose centroid falls outside it.

    A cell-centred finite-volume scheme puts the unknown at the cell centre and
    measures every face distance from it.  Cutting the river channel out of a
    background cell can leave a crescent that wraps the ribbon, and such a cell's
    centroid lies outside it -- frequently inside the very ribbon cell across the
    face being measured.  The exported ``mesh2d_face_x/y`` and
    ``edge_center_distance_m`` are then meaningless for that cell, and the
    quality gate reads the pair as two cells sharing a centre, which no merge can
    repair because merging them moves the centroid no closer to either.

    One straight cut through the centroid restores the property.  Both pieces own
    the cut, so the new face is internal and the mesh stays conforming.

    NOT wired into ``conform_and_repair_cells``, deliberately.  On the Pune mesh
    an axis-aligned cut through a crescent's centroid produces thin pieces that
    are themselves crescents, and the repair diverges: 519 bad face pairs became
    16,016 over three rounds while the cell count grew by 17,000.  Splitting
    these cells needs a cut chosen from the cell's shape -- along its medial axis,
    or along the ribbon it wraps -- not an axis-aligned line.  The helper is kept
    because the invariant it enforces is the right one and
    ``cells_with_exterior_centroid`` reports how far the mesh is from it.
    """
    objects = np.asarray(polygons, dtype=object)
    outside = np.flatnonzero(~contains(objects, centroid(objects)))
    if not len(outside):
        return list(polygons), 0
    pending = {int(index) for index in outside}
    result: list[Polygon] = [polygon for index, polygon in enumerate(polygons) if index not in pending]
    queue = [polygons[index] for index in sorted(pending)]
    split_count = 0
    for _ in range(max(1, max_cuts)):
        if not queue:
            break
        following: list[Polygon] = []
        for polygon in queue:
            if polygon.contains(polygon.centroid):
                result.append(polygon)
                continue
            parts = _straight_cuts(polygon, polygon.centroid)
            if len(parts) == 1:
                result.append(polygon)  # not separable; reported, not hidden
                continue
            split_count += 1
            following.extend(parts)
        queue = following
    result.extend(queue)
    return result, split_count


def hole_free_parts(polygon: Polygon, max_cuts: int = 8) -> list[Polygon]:
    """Split a cell that encloses a feature into simply connected pieces.

    Cutting the structured channel out of a background cell can leave the
    ribbon entirely inside one remnant, so the remnant carries an interior
    ring.  ``_topology`` reads only ``exterior``, so such a ring would export
    as an invisible interface and its neighbours would look like domain
    boundary.  One straight cut per hole restores a simple ring; both pieces
    own the cut, so the new face is internal and conforming.
    """
    pending = [polygon]
    result: list[Polygon] = []
    for _ in range(max_cuts):
        if not pending:
            break
        following: list[Polygon] = []
        for piece in pending:
            if not piece.interiors:
                result.append(piece)
                continue
            parts = _straight_cuts(piece, Polygon(piece.interiors[0]).representative_point())
            following.extend(parts if len(parts) > 1 else [])
            if len(parts) <= 1:
                # Neither cut separated the ring.  Keep the outer ring only:
                # the enclosed feature cells already own that area, so dropping
                # the ring would double-count it.
                result.append(piece)
        pending = following
    result.extend(pending)
    return result


def self_paired_cell_indices(polygons: list[Polygon]) -> np.ndarray:
    """Cells the quality scan would pair with themselves.

    Uses exactly the coordinate set and rounding ``_edge_quality`` uses: all
    rings, eight decimals.  A cell whose ring walks one segment twice appears
    there as a face whose two owners are the same index, and no merge can
    repair that because there is only one cell -- so the caller's repair loop
    freezes on it unless the cell is dealt with here.
    """
    objects = np.asarray(polygons, dtype=object)
    counts_per_cell = get_num_coordinates(objects)
    coordinates = get_coordinates(objects)
    if not len(coordinates):
        return np.empty(0, dtype=np.int64)
    first = np.r_[0, np.cumsum(counts_per_cell[:-1])]
    coordinate_owner = np.repeat(np.arange(len(objects), dtype=np.int64), counts_per_cell)
    is_edge_start = np.ones(len(coordinates), dtype=bool)
    is_edge_start[first + counts_per_cell - 1] = False
    start = np.flatnonzero(is_edge_start)
    owner = coordinate_owner[start]
    start_xy, end_xy = coordinates[start], coordinates[start + 1]
    keep = np.hypot(start_xy[:, 0] - end_xy[:, 0], start_xy[:, 1] - end_xy[:, 1]) > 1e-8
    owner, start_xy, end_xy = owner[keep], start_xy[keep], end_xy[keep]
    if not len(owner):
        return np.empty(0, dtype=np.int64)
    swap = (start_xy[:, 0] > end_xy[:, 0]) | (
        (start_xy[:, 0] == end_xy[:, 0]) & (start_xy[:, 1] > end_xy[:, 1])
    )
    edge_key = np.round(np.column_stack((
        np.where(swap, end_xy[:, 0], start_xy[:, 0]), np.where(swap, end_xy[:, 1], start_xy[:, 1]),
        np.where(swap, start_xy[:, 0], end_xy[:, 0]), np.where(swap, start_xy[:, 1], end_xy[:, 1]),
    )), 8)
    order = np.lexsort((edge_key[:, 3], edge_key[:, 2], edge_key[:, 1], edge_key[:, 0]))
    sorted_key, sorted_owner = edge_key[order], owner[order]
    group_start = np.r_[0, np.flatnonzero(np.any(sorted_key[1:] != sorted_key[:-1], axis=1)) + 1]
    group_end = np.r_[group_start[1:], len(sorted_key)]
    index = group_start[(group_end - group_start) == 2]
    return np.unique(sorted_owner[index][sorted_owner[index] == sorted_owner[index + 1]])


def _robust_union(left, right, grid_size: float = 1e-4, bridge_m: float = 0.01):
    """``left | right`` as a single polygon, retrying under precision reduction.

    Two cells that look adjacent can be separated by a hair-thin gap left by an
    earlier overlay, and their plain union is then a multipolygon.  Snapping
    both to the 0.1 mm grid ``_topology`` already rounds to closes the gap and
    changes no coordinate the export would have kept.  Returns ``None`` if the
    two genuinely cannot be joined.
    """
    for attempt in (0, 1, 2):
        try:
            if attempt == 0:
                merged = unary_union((left, right))
            elif attempt == 1:
                merged = union(make_valid(left), make_valid(right), grid_size=grid_size)
            else:
                # The two are genuinely separated by a hairline gap, not merely
                # imprecise: closing it adds the gap's area to the result.  That
                # is the right answer, not a fudge -- the gap is unclaimed
                # partition area, and the cell it isolates cannot otherwise be
                # absorbed by any predicate.  The tolerance is capped so this can
                # only ever close a gap far below the hydraulic floor.
                # Bridge along the shortest line between the two.  Buffering the
                # union out and back in does not work -- eroding by the same
                # amount reopens the bridge -- and a wider buffer would distort
                # both cells.  This adds only the gap itself.
                merged = unary_union((
                    left, right,
                    shortest_line(left, right).buffer(bridge_m, join_style="mitre"),
                ))
        except Exception:
            continue
        if merged.geom_type == "Polygon" and merged.area > 0:
            return merged
    return None

def absorb_named_cells(
    polygons: list[Polygon], targets: set[int], report: dict[str, int] | None = None,
    classes: list[str] | None = None,
) -> tuple[list[Polygon], int]:
    """Merge each named cell into whichever neighbour shares the most boundary.

    Used for cells that carry no hydraulic information but do control the
    timestep, and for cells whose ring cannot be repaired at all.  The union
    preserves area exactly.

    ``classes`` makes the choice of partner prefer a neighbour of the same class,
    falling back to any neighbour only when there is none.  Without it this
    function ate 5,601 of the 45,633 river cells on the Pune build: a river cell
    just under the floor was absorbed into the urban cell it shared most boundary
    with, the merged cell no longer classified as river, and the channel was left
    with a one-cell hole between two river cells -- 3,248 of them, all inside a
    single reach, which is why they looked like random breaks rather than junction
    defects.  Crossing classes is still allowed as a last resort, because a cell
    that cannot be exported is worse than a slightly larger neighbour.

    ``report`` records why a cell could not be absorbed.  It is the only way to
    tell an isolated sliver -- every neighbour is itself a sliver -- from one
    whose union with each neighbour is not a single polygon.  Those two need
    different fixes, and the absorbed count reads the same either way.
    """
    if report is not None:
        for key in ("targets", "no_free_neighbour", "union_not_polygon"):
            report.setdefault(key, 0)
        report["targets"] += len(targets)
    if not targets:
        return list(polygons), 0
    tree = STRtree(polygons)
    merged: dict[int, Polygon] = {}
    removed: set[int] = set()
    for index in sorted(targets):
        if index in removed:
            continue
        neighbours = [
            int(item) for item in tree.query(polygons[index], predicate="intersects")
            if int(item) != index and int(item) not in removed
        ]
        candidates = [item for item in neighbours if item not in targets]
        if classes is not None and candidates:
            same = [item for item in candidates if classes[item] == classes[index]]
            if same:
                candidates = same
                if report is not None:
                    report.setdefault("kept_in_class", 0)
                    report["kept_in_class"] += 1
        if not candidates:
            # Every neighbour is itself a target.  Merging two of them still
            # helps: the result is larger than either, and a later pass can
            # absorb it into something that clears the floor.  Refusing here
            # strands clusters of slivers permanently.
            if report is not None:
                report["no_free_neighbour"] += 1
            candidates = neighbours
        if not candidates:
            # No cell intersects this one at all: it sits behind a hairline gap
            # an earlier overlay left, so no predicate reaches it and it can
            # never be absorbed.  Fall back to the nearest cell and let the
            # precision-reduced union close the gap.  Without this a cell like a
            # 25 m triangle is stranded permanently, and a triangle can never
            # satisfy a 2A/P floor anyway: 2A/P is its inradius, so it would need
            # roughly 52 m sides to clear 15 m.  Merging is the only repair.
            nearest = np.atleast_1d(
                tree.query_nearest(polygons[index], exclusive=True, all_matches=False)
            )
            candidates = [
                int(item) for item in nearest
                if int(item) != index and int(item) not in removed
            ]
            if report is not None and candidates:
                report.setdefault("reached_by_nearest", 0)
                report["reached_by_nearest"] += 1
        if not candidates:
            continue
        # Best-first over every neighbour, not just the best one.  A thin sliver
        # often cannot union with the neighbour it shares most boundary with -- the
        # result is a multipolygon -- while a different neighbour works fine.
        # Giving up on the first failure left 35 slivers in the Pune mesh that the
        # gate then rejected.
        ordered = sorted(
            candidates,
            key=lambda item: polygons[index].boundary.intersection(polygons[item].boundary).length,
            reverse=True,
        )
        for target in ordered:
            joined = _robust_union(merged.get(target, polygons[target]), polygons[index])
            if joined is None:
                continue
            merged[target] = joined
            removed.add(index)
            break
        else:
            if report is not None:
                report["union_not_polygon"] += 1
    if not removed:
        return list(polygons), 0
    if classes is not None:
        # Keep the caller's class list index-aligned with the polygons we return,
        # the same way _agglomerate_short_face_cells does. Without this the next
        # pass reads a class off the wrong cell.
        classes[:] = [item for index, item in enumerate(classes) if index not in removed]
    return [
        merged.get(index, polygon)
        for index, polygon in enumerate(polygons) if index not in removed
    ], len(removed)

def split_self_touching_cells(polygons: list[Polygon]) -> tuple[list[Polygon], int]:
    """Split a ring that walks the same segment twice into separate cells.

    Welding two vertices together can pinch a ring so that it touches itself.
    GEOS still reports such a ring valid, but the repeated segment makes the
    cell its own finite-volume neighbour: ``_edge_quality`` reports a face whose
    two owners are the same index, and no merge can repair that because there is
    only one cell.

    The doubled traversal is removed from the ring first and ``buffer(0)`` then
    resolves whatever genuine self-intersection is left.  ``buffer(0)`` alone is
    not enough: GEOS treats an out-and-back ring as valid and canonical and
    returns it unchanged, which is how this defect survived several repair passes.

    The returned count is *repairs*, not detections -- a cell can be detected and
    still resist repair.  Watch the quality scan's self-paired count, not this
    number, to tell whether the defect is gone.
    """
    pinched = self_paired_cell_indices(polygons)
    pinched = np.union1d(
        pinched, np.flatnonzero(~is_valid(np.asarray(polygons, dtype=object))),
    )
    if not len(pinched):
        return list(polygons), 0
    pinched_set = set(pinched.tolist())
    result: list[Polygon] = []
    repaired_count = 0
    for index, polygon in enumerate(polygons):
        if index not in pinched_set:
            result.append(polygon)
            continue
        ring = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        cleaned = _despiked_ring(ring)
        candidate = Polygon(cleaned) if len(cleaned) >= 3 else polygon
        if not candidate.is_valid or candidate.area <= 0:
            candidate = polygon
        parts = [part for part in _polygon_parts(candidate.buffer(0)) if part.area > 0]
        if not parts:
            result.append(polygon)
            continue
        if len(parts) > 1 or not polygon.equals(parts[0]):
            repaired_count += 1
        result.extend(parts)
    return result, repaired_count


def collapse_short_faces(
    polygons: list[Polygon], minimum_length_m: float, max_passes: int = 4,
) -> tuple[list[Polygon], int]:
    """Weld vertex pairs that would export a face below the hydraulic floor.

    A confluence trim, or an earlier hanging-node snap, can leave two ring
    vertices under a metre apart.  Every cell along that boundary then carries a
    sub-metre finite-volume face which controls the explicit timestep while
    carrying no flow.  Welding the pair removes the face for all of its owners at
    once, so the mesh stays conforming.  The alternative -- merging the two
    owners -- would destroy a well-formed kilometre-scale cell to repair a
    one-metre edge.
    """
    result = list(polygons)
    welded_total = 0
    for _ in range(max_passes):
        _, _, start_xy, end_xy = _ring_edges(result)
        if not len(start_xy):
            break
        length = np.hypot(end_xy[:, 0] - start_xy[:, 0], end_xy[:, 1] - start_xy[:, 1])
        short = np.flatnonzero((length > 0.0) & (length < minimum_length_m))
        if not len(short):
            break
        parent: dict[tuple[float, float], tuple[float, float]] = {}

        def representative(vertex: tuple[float, float]) -> tuple[float, float]:
            root = vertex
            while parent.get(root, root) != root:
                root = parent[root]
            return root

        for index in short:
            left = (round(float(start_xy[index][0]), 4), round(float(start_xy[index][1]), 4))
            right = (round(float(end_xy[index][0]), 4), round(float(end_xy[index][1]), 4))
            left_root, right_root = representative(left), representative(right)
            if left_root == right_root:
                continue
            # Deterministic winner keeps the weld independent of ring order.
            loser, winner = sorted((left_root, right_root), reverse=True)
            parent[loser] = winner
        if not parent:
            break
        snap = {vertex: representative(vertex) for vertex in parent}
        snap = {source: target for source, target in snap.items() if source != target}
        if not snap:
            break
        result = _snapped_vertices(result, snap)
        welded_total += len(snap)
    return result, welded_total


def drop_null_area_cells(
    polygons: list[Polygon], minimum_area_m2: float = 1e-6,
) -> tuple[list[Polygon], int]:
    """Remove cells that have collapsed onto a line.

    A repair pass can leave three collinear vertices: 2e-10 m2 of area on a
    27 m perimeter.  That is not a control volume.  It cannot be merged either,
    because its union with a neighbour is not a single polygon, and while it
    exists its edges give the real cells on either side of that line a third
    owner, so those two never pair with each other.  Removing it returns no
    area to anybody, because it has none.
    """
    keep = [polygon for polygon in polygons if polygon.area > minimum_area_m2]
    return keep, len(polygons) - len(keep)

def absorb_degenerate_cells(
    polygons: list[Polygon], minimum_scale_m: float, max_passes: int = 4,
    report: dict[str, int] | None = None, classes: list[str] | None = None,
) -> tuple[list[Polygon], int]:
    """Absorb sub-floor fragments into whichever neighbour shares most boundary.

    Cutting the ribbon out of the background diagram, and snapping a vertex onto
    its neighbour, can both leave a fragment far below the hydraulic floor -- in
    the worst case a sub-square-metre triangle.  Such a cell carries no
    hydraulic information yet still controls the explicit timestep, and it is
    too small to pair with a same-class neighbour reliably, so the merge is
    allowed to cross feature classes.  The union preserves area exactly.

    Delegates to ``absorb_named_cells`` rather than repeating its loop.  It had
    its own copy, which meant the best-first retry over every neighbour was
    added to one and not the other -- and this is the function that handles the
    sub-floor slivers, so the improvement never reached them.
    """
    result = list(polygons)
    absorbed_total = 0
    for _ in range(max_passes):
        objects = np.asarray(result, dtype=object)
        scale = 2.0 * np.asarray([item.area for item in objects]) / np.maximum(
            np.asarray([item.length for item in objects]), 1e-12,
        )
        degenerate = set(np.flatnonzero(scale < minimum_scale_m).tolist())
        if not degenerate:
            break
        result, absorbed = absorb_named_cells(result, degenerate, report, classes)
        if not absorbed:
            break
        absorbed_total += absorbed
    return result, absorbed_total

def hole_free_polygons(polygons: list[Polygon]) -> list[Polygon]:
    """Split every cell that has acquired an interior ring.

    Merging two cells that between them surround a third leaves a ring that
    ``_topology`` never reads, so the enclosed interface would export as domain
    boundary and the enclosing cell would later be rebuilt from its exterior
    alone -- silently overlapping its neighbour.
    """
    if not any(polygon.interiors for polygon in polygons):
        return list(polygons)
    result: list[Polygon] = []
    for polygon in polygons:
        result.extend(hole_free_parts(polygon) if polygon.interiors else [polygon])
    return result


def rerouted_repair_pairs(
    polygons: list[Polygon], pairs: np.ndarray, groups: list[str], protected_group: str,
) -> np.ndarray:
    """Send a cross-group bad face to a mergeable same-group neighbour instead.

    A short face between a ribbon cell and a surface cell cannot be repaired by
    merging the two, because the ribbon is immutable feature geometry.  Merging
    the surface cell into its longest surface neighbour removes the same face
    and leaves the river cell exactly as built.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    reroute = [
        int(item)
        for left, right in pairs if groups[left] != groups[right]
        for item in (left, right) if groups[item] != protected_group
    ]
    if not reroute:
        return np.empty((0, 2), dtype=np.int64)
    return _undersized_cell_repair_pairs(
        polygons, np.unique(reroute), groups, same_class_only=True,
    )


def _robust_difference(left, right, grid_size: float = 1e-4):
    """``left - right``, retrying under precision reduction, or ``None``.

    A union of a few thousand repaired mesh cells can be complex enough that the
    GEOS overlay raises "unable to assign free hole to a shell".  Snapping both
    operands to the 0.1 mm grid that ``_topology`` already rounds to is the
    standard remedy and changes no coordinate the export would have kept anyway.
    """
    for attempt in (0, 1, 2):
        try:
            if attempt == 0:
                return left.difference(right)
            if attempt == 1:
                return difference(make_valid(left), make_valid(right))
            return difference(make_valid(left), make_valid(right), grid_size=grid_size)
        except Exception:
            continue
    return None


def fill_partition_voids(
    polygons: list[Polygon], domain, search_radius_m: float, max_faces: int = 4000,
) -> tuple[list[Polygon], int, float, bool]:
    """Give back any area the cell set lost while vertices were being moved.

    Welding a short face, or snapping a hanging node, moves a vertex in every
    cell that carries it.  A neighbour whose straight edge merely passes through
    that vertex does not carry it, so the two can part company and leave a thin
    void.  A void is a hole in the finite-volume partition and mass entering it
    is simply lost, so each one is merged back into the neighbour that shares the
    most boundary with it.

    Only the neighbourhood of the unpaired faces is examined: a void always has
    one, and differencing the whole domain against a 300,000-cell union to find a
    100 m2 triangle is not affordable.

    Above ``max_faces`` unpaired faces the search is skipped, and the caller is
    told so.  This is a scope decision, not a silent cap: a mesh with tens of
    thousands of unpaired faces is not yet conforming, so voids are not the
    defect to chase there, and building a search region around that many
    segments costs more than the entire rest of the build.  The skip is reported
    in QA so it can never be mistaken for "no voids found".
    """
    _, _, start_xy, end_xy = _ring_edges(polygons)
    if not len(start_xy):
        return list(polygons), 0, 0.0, True
    unpaired = _unpaired_ring_edges(np.arange(len(start_xy)), start_xy, end_xy)
    if not len(unpaired):
        return list(polygons), 0, 0.0, True
    segments = shapely_linestrings(
        np.stack((start_xy[unpaired], end_xy[unpaired]), axis=1).reshape(-1, 2),
        indices=np.repeat(np.arange(len(unpaired)), 2),
    )
    # Most unpaired faces are the real catchment edge.  Including them would
    # buffer a band around the whole 385 km perimeter and union every cell along
    # it -- minutes of work to look for voids that by definition are interior.
    boundary_parts = _line_parts(domain.boundary)
    if boundary_parts:
        midpoints = shapely_points(np.column_stack((
            0.5 * (start_xy[unpaired][:, 0] + end_xy[unpaired][:, 0]),
            0.5 * (start_xy[unpaired][:, 1] + end_xy[unpaired][:, 1]),
        )))
        on_boundary = np.unique(
            STRtree(boundary_parts).query(midpoints, predicate="dwithin", distance=1.0)[0]
        )
        interior = np.setdiff1d(np.arange(len(unpaired)), on_boundary)
        if not len(interior):
            return list(polygons), 0, 0.0, True
        segments = segments[interior]
    if len(segments) > max_faces:
        return list(polygons), 0, 0.0, False
    # Square caps and mitred joins with one segment per quadrant: a round buffer
    # of thousands of lines generates arc vertices whose union is where this
    # stalls, and the region only has to be a generous neighbourhood.
    region = unary_union(segments).buffer(
        max(search_radius_m, 1.0), quad_segs=1, cap_style="square", join_style="mitre",
    ).intersection(domain)
    if region.is_empty:
        return list(polygons), 0, 0.0, True
    tree = STRtree(polygons)
    local = np.unique(tree.query(region, predicate="intersects"))
    if not len(local):
        return list(polygons), 0, 0.0, True
    # An overlay difference can emit slivers that are collapsed rather than thin:
    # a square millimetre of area spread along a 60 m perimeter.  They are not
    # lost partition area, so they are dropped rather than reclaimed as cells.
    remainder = _robust_difference(region, unary_union([polygons[int(item)] for item in local]))
    if remainder is None:
        # The overlay could not be completed even with precision reduction.  Do
        # not guess: leaving the void unreclaimed keeps it visible in the
        # ``interior_unpaired_face_count`` QA field, whereas a partial subtraction
        # could delete real cells.
        return list(polygons), 0, 0.0, False
    voids = [
        part for part in _polygon_parts(remainder)
        if part.area > 1e-6 and 2.0 * part.area / max(part.length, 1e-12) > 1e-3
    ]
    if not voids:
        return list(polygons), 0, 0.0, True
    result = list(polygons)
    filled = 0
    filled_area = 0.0
    for void in voids:
        candidates = [int(item) for item in tree.query(void, predicate="intersects")]
        touching = [
            (result[index].boundary.intersection(void.boundary).length, index)
            for index in candidates
        ]
        touching = [item for item in touching if item[0] > 0]
        if not touching:
            continue
        _, target = max(touching)
        union = unary_union((result[target], void))
        if union.geom_type != "Polygon" or union.area <= 0:
            continue
        result[target] = union
        filled += 1
        filled_area += void.area
    return result, filled, filled_area, True


def conform_and_repair_cells(
    polygons: list[Polygon], minimum_face_length_m: float, minimum_cell_scale_m: float, domain=None,
    max_rounds: int = 5, classes: list[str] | None = None,
) -> tuple[list[Polygon], dict[str, float]]:
    """Make a cell set conforming, then clear sub-floor faces, voids, and fragments.

    Within a round the order is the substance:

    1. Interior rings first, because every later step rebuilds a cell from its
       exterior ring and would delete a hole without returning the area.
    2. Then node insertion.  Welding a short face moves a vertex in every cell
       that carries it, which is only gap-free once each of those cells actually
       has the shared vertex; welding a mesh that still has hanging nodes leaves
       slivers between a straight segment and the subdivided one beside it.
       Weld, then split pinched rings, then remove rings again -- each step can
       create the defect the next one repairs.
    3. Reclaim voids before absorbing fragments, so a reclaimed void that is
       itself below the floor gets absorbed rather than exported.

    The round then repeats while anything is still moving geometry.  This matters
    for more than tidiness: the caller's repair loop re-classifies the whole mesh
    and re-runs the quality scan after every call, which on a 400,000-cell mesh
    costs more than several rounds here.  Returning in a state where a snap has
    just created a new short face guarantees the caller another full iteration,
    and the two can trade work back and forth without converging.
    """
    totals = {
        "inserted": 0, "snapped": 0, "welded": 0, "pinched": 0, "absorbed": 0,
        "collapsed_dropped": 0, "voids_filled": 0, "void_area_m2": 0.0,
        "absorb_why": {},
        "void_search_skipped": 0, "rounds": 0,
    }
    for _ in range(max(1, max_rounds)):
        totals["rounds"] += 1
        polygons = hole_free_polygons(polygons)
        polygons, inserted, snapped = insert_hanging_nodes(
            polygons, minimum_face_length_m=minimum_face_length_m,
        )
        polygons, welded = collapse_short_faces(polygons, minimum_face_length_m)
        polygons, pinched = split_self_touching_cells(polygons)
        polygons = hole_free_polygons(polygons)
        polygons, voids_filled, void_area, void_search_ran = (
            fill_partition_voids(polygons, domain, 2.0 * minimum_face_length_m)
            if domain is not None else (polygons, 0, 0.0, True)
        )
        polygons, collapsed = drop_null_area_cells(polygons)
        polygons, absorbed = absorb_degenerate_cells(
            polygons, minimum_cell_scale_m, classes=classes,
        )
        # Pair up whatever the welds, splits, and merges above have just broken.
        # This has to be inside the round, not after it: the pass may snap a
        # vertex, and a snap can create the very short face a weld exists to
        # remove, so its effect must be able to trigger another round.
        polygons, closing_inserted, closing_snapped = insert_hanging_nodes(
            polygons, minimum_face_length_m=minimum_face_length_m,
        )
        # Check for self-pairing again afterwards.  Inserting a node into a ring
        # can make it walk one segment twice, and the quality scan then reports a
        # face whose two owners are the same cell -- which no merge can repair,
        # because there is only one cell.  Leaving that to the caller freezes its
        # repair loop: on the Pune mesh 164 of 179 residual bad pairs were this,
        # and the loop cycled on them for twenty iterations.
        polygons, closing_pinched = split_self_touching_cells(polygons)
        # Whatever is still self-paired after de-spiking and buffer(0) cannot be
        # repaired as a ring at all.  Absorbing it into a neighbour removes the
        # cell outright, which always terminates, and 90 such cells out of
        # 328,000 is a far smaller compromise than a mesh the gate rejects.
        polygons, stubborn = absorb_named_cells(
            polygons, set(self_paired_cell_indices(polygons).tolist()),
        )
        # Sub-floor absorption again, last.  The closing insert, the weld, and the
        # void filler all run after the first absorption and all create thin
        # remnants of their own; absorbing only before them left about forty
        # slivers in the Pune mesh that the gate rejected, unchanged however many
        # rounds were allowed.
        polygons, collapsed_late = drop_null_area_cells(polygons)
        polygons, absorbed_late = absorb_degenerate_cells(
            polygons, minimum_cell_scale_m, report=totals["absorb_why"], classes=classes,
        )
        totals["inserted"] += inserted + closing_inserted
        totals["snapped"] += snapped + closing_snapped
        totals["welded"] += welded
        totals["pinched"] += pinched + closing_pinched
        totals["absorbed"] += stubborn
        totals["collapsed_dropped"] += collapsed + collapsed_late
        totals["absorbed"] += absorbed_late
        totals["absorbed"] += absorbed
        totals["voids_filled"] += voids_filled
        totals["void_area_m2"] += void_area
        totals["void_search_skipped"] += 0 if void_search_ran else 1
        # Inserting a node does not move geometry, so it cannot create a defect
        # for the next round.  Every other step can.
        if not (
            snapped or closing_snapped or welded or pinched or closing_pinched
            or absorbed or absorbed_late or stubborn or collapsed or collapsed_late
            or voids_filled
        ):
            break
    # The tail of the last round -- the closing insert, the pinch split, the void
    # fill -- can leave a sub-floor cell with nothing after it to absorb it, so
    # the caller measures a defect the round had no chance to clear.  One final
    # pass here, then re-pair whatever it merged.
    polygons, closing_collapsed = drop_null_area_cells(polygons)
    polygons, closing_absorbed = absorb_degenerate_cells(
        polygons, minimum_cell_scale_m, report=totals["absorb_why"], classes=classes,
    )
    totals["collapsed_dropped"] += closing_collapsed
    totals["absorbed"] += closing_absorbed
    if closing_collapsed or closing_absorbed:
        polygons, final_inserted, final_snapped = insert_hanging_nodes(
            polygons, minimum_face_length_m=minimum_face_length_m,
        )
        totals["inserted"] += final_inserted
        totals["snapped"] += final_snapped
    return polygons, totals


def substitute_river_strip(
    background: list[Polygon], channel_geometry, strip_cells: list[Polygon],
) -> tuple[list[Polygon], np.ndarray]:
    """Cut the structured channel out of the background mesh and drop it in.

    The channel outline is the union of the strip cells themselves, so every
    background remnant meets the ribbon along segments the strip already owns.
    Remnants are kept whatever their size: dropping a sliver here would leave a
    hole, and a hole reads to the solver as an interior wall.  Undersized
    remnants are left for the normal agglomeration pass instead.

    Never subtract the merged channel: ``polygon - channel`` and
    ``polygon - (ribbon cells meeting polygon)`` describe the same set, but the
    merged outline has no vertex where two ribbon cells meet, so the remnant gets
    one long edge spanning several ribbon faces and the interface stops pairing.
    """
    if channel_geometry is None or channel_geometry.is_empty or not strip_cells:
        return list(background), np.zeros(len(background), dtype=bool)
    # Find the ribbon cells touching each background cell with one bulk query.
    # ``STRtree.query`` prepares its query geometry and crosses the Python/GEOS
    # boundary on every call, so asking it 300,000 separate questions took longer
    # than the rest of the build put together; the array form answers all of them
    # inside one C call.
    pairs = STRtree(strip_cells).query(
        np.asarray(background, dtype=object), predicate="intersects",
    )
    touching: dict[int, list[int]] = {}
    for background_index, strip_index in zip(pairs[0], pairs[1], strict=True):
        touching.setdefault(int(background_index), []).append(int(strip_index))
    result: list[Polygon] = []
    for index, polygon in enumerate(background):
        local = touching.get(index)
        if not local:
            result.append(polygon)
            continue
        # One ribbon cell at a time, so its corners stay in the remnant's ring.
        remainder = polygon
        for item in local:
            remainder = _robust_difference(remainder, strip_cells[item])
            if remainder is None or remainder.is_empty:
                break
        if remainder is None:
            remainder = _robust_difference(
                polygon, unary_union([strip_cells[item] for item in local]),
            )
        if remainder is None:
            remainder = _robust_difference(polygon, channel_geometry.buffer(0))
        if remainder is None:
            raise ValueError(
                "Could not cut the structured river channel out of a background cell "
                f"near {polygon.representative_point().wkt}. Leaving it whole would make "
                "the cell overlap the ribbon and double-count its area."
            )
        for part in _polygon_parts(remainder):
            if part.area <= 1e-9:
                continue
            result.extend(hole_free_parts(part) if part.interiors else [part])
    is_strip = np.zeros(len(result) + len(strip_cells), dtype=bool)
    is_strip[len(result):] = True
    result.extend(strip_cells)
    return result, is_strip


def internal_unpaired_edges(
    edges: dict[tuple[tuple[float, float], tuple[float, float]], list[int]], domain, tolerance: float = 1.0,
) -> tuple[int, float]:
    """Count and measure single-owner faces that are not on the domain edge.

    Any such face is a hanging node: the solver sees a closed wall where two
    cells actually meet.  ``_edge_quality`` cannot see them because it only
    inspects two-owner faces, so this is reported separately in QA.
    """
    single = [key for key, owners in edges.items() if len(owners) == 1]
    if not single:
        return 0, 0.0
    midpoints = shapely_points(np.asarray([
        ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in single
    ], dtype=np.float64))
    boundary_parts = _line_parts(domain.boundary)
    if not boundary_parts:
        return len(single), float(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in single))
    near = set(
        STRtree(boundary_parts).query(midpoints, predicate="dwithin", distance=tolerance)[0].tolist()
    )
    interior = [key for index, key in enumerate(single) if index not in near]
    return len(interior), float(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in interior))


def _merge_group_keys(feature_classes: list[str]) -> list[str]:
    """Which cells a repair merge is allowed to combine.

    River and waterbody each get their own group so a merge can never step across
    a bank or a reservoir shoreline; everything else is one surface group, because
    an urban/rural sliver is freely absorbable. Without the waterbody group the
    hard shoreline clip above would be undone by the very next repair pass.
    """
    return [
        item if item in {"river", "waterbody"} else "surface" for item in feature_classes
    ]


def _feature_classes(
    polygons: list[Polygon], river_geometry, floodplain_geometry, urban: np.ndarray,
    transform: Any, river_points: np.ndarray | None = None, waterbody_geometry=None,
) -> list[str]:
    """Classify after hard splitting on the aligned source grid.

    A Python ``covers`` call per polygon made the class pass dominate the Pune
    build.  The feature geometries are already raster-aligned, so classifying
    each representative location against one shared raster gives the same
    priority (waterbody, river, floodplain, urban, rural) without repeated
    complex-geometry predicates.
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
    if waterbody_geometry is not None and not waterbody_geometry.is_empty:
        waterbody = geometry_mask(
            [waterbody_geometry], out_shape=urban.shape, transform=transform, invert=True,
        )
        result[valid_index[waterbody[rows[valid_index], cols[valid_index]]]] = "waterbody"
    return result.tolist()


def _geometry_area_fractions(
    polygons: list[Polygon], geometry, tree: STRtree | None = None,
    feature_parts: list[Polygon] | None = None,
) -> np.ndarray:
    """Exact feature fractions, evaluating only polygons that intersect it.

    The feature is split into its connected components before querying.  A
    single ``query(union, predicate="intersects")`` makes GEOS build one prepared
    polygon for the whole feature and test every candidate against it; when the
    feature is the structured river channel -- a 1,400 km ribbon with hundreds of
    thousands of vertices -- that one call dominates the entire build, and it is
    repeated on each classification pass.  Components are disjoint, so summing
    each cell's overlap with the components it actually meets gives the same
    answer from small, cheap overlays.
    """
    result = np.zeros(len(polygons), dtype=np.float64)
    if not polygons:
        return result
    # ``feature_parts`` lets a caller pass a finer disjoint decomposition it
    # already holds -- the individual ribbon cells rather than the merged ribbon.
    # The parts must be disjoint and cover the same area, or the sum double-counts.
    # ``buffer(0)`` on an already-valid feature is pure cost, and for a
    # mask-derived floodplain covering a whole catchment it is one of the most
    # expensive calls in the build.  Only pay it when the geometry needs it.
    parts = list(feature_parts) if feature_parts else (
        [] if geometry is None or geometry.is_empty
        else _polygon_parts(geometry if geometry.is_valid else geometry.buffer(0))
    )
    if not parts:
        return result
    objects = np.asarray(polygons, dtype=object)
    part_objects = np.asarray(parts, dtype=object)
    pairs = (tree if tree is not None else STRtree(objects)).query(
        part_objects, predicate="intersects",
    )
    if not pairs.size:
        return result
    part_index, cell_index = pairs[0], pairs[1]
    # Clip in chunks.  A single vectorised ``intersection`` over every pair
    # materialises one clipped geometry per pair, and for a mask-derived feature
    # with tens of thousands of components that is millions of live geometries --
    # enough to push a 24 GB machine into swap, where the stage stops being
    # compute-bound and effectively never finishes.  Areas accumulate, so the
    # chunk size only bounds peak memory.
    overlap = np.empty(len(cell_index), dtype=np.float64)
    chunk = 200_000
    for begin in range(0, len(cell_index), chunk):
        end = min(begin + chunk, len(cell_index))
        cells_here, parts_here = cell_index[begin:end], part_index[begin:end]
        try:
            clipped = intersection(objects[cells_here], part_objects[parts_here])
            overlap[begin:end] = np.fromiter(
                (item.area for item in clipped), dtype=np.float64, count=end - begin,
            )
        except Exception:
            overlap[begin:end] = np.fromiter(
                (
                    objects[cell].buffer(0).intersection(part_objects[part].buffer(0)).area
                    for cell, part in zip(cells_here, parts_here, strict=True)
                ),
                dtype=np.float64, count=end - begin,
            )
    numerator = np.zeros(len(polygons), dtype=np.float64)
    np.add.at(numerator, cell_index, overlap)
    touched = np.unique(cell_index)
    denominator = np.fromiter(
        (item.area for item in objects[touched]), dtype=np.float64, count=len(touched),
    )
    result[touched] = np.divide(
        numerator[touched], denominator, out=np.zeros_like(denominator), where=denominator > 0,
    )
    return np.minimum(result, 1.0)


def _fraction_refined_feature_classes(
    feature_classes: list[str], river_channel_fraction: np.ndarray, floodplain_fraction: np.ndarray,
    waterbody_fraction: np.ndarray, feature_class_fraction: float = 0.10,
    river_channel_class_fraction: float = 0.5, waterbody_class_fraction: float = 0.10,
) -> list[str]:
    result = np.asarray(feature_classes, dtype=object)
    river_cells = river_channel_fraction >= river_channel_class_fraction
    result[river_cells] = "river"
    false_river = (result == "river") & ~river_cells
    result[false_river & (floodplain_fraction >= feature_class_fraction)] = "floodplain"
    result[false_river & (floodplain_fraction < feature_class_fraction)] = "rural"
    result[waterbody_fraction > waterbody_class_fraction] = "waterbody"
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
        # GEOS overlay can leave vertices separated by only machine-scale
        # distances.  If retained, two cells that meet at one point appear to
        # share a microscopic finite-volume face.  A 0.1 mm coordinate grid is
        # immaterial for metre-scale meshes and removes that false topology.
        coordinates = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        coordinates = [
            point for index, point in enumerate(coordinates)
            if index == 0 or point != coordinates[index - 1]
        ]
        if len(coordinates) > 1 and coordinates[-1] == coordinates[0]:
            coordinates.pop()
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


def _undersized_cell_repair_pairs(
    polygons: list[Polygon], bad: np.ndarray, classes: list[str], same_class_only: bool = True,
) -> np.ndarray:
    """Pair each undersized cell with its longest same-class neighbour."""
    tree = STRtree(polygons)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for left in np.asarray(bad, dtype=np.int64):
        if int(left) in used:
            continue
        candidates = tree.query(polygons[int(left)], predicate="touches")
        candidates = [
            int(right) for right in candidates
            if right != left and int(right) not in used
            and (not same_class_only or classes[int(right)] == classes[int(left)])
        ]
        if not candidates:
            continue
        right = max(
            candidates,
            key=lambda item: polygons[int(left)].boundary.intersection(polygons[item].boundary).length,
        )
        pairs.append((int(left), right))
        used.update((int(left), right))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def area_weighted_cell_target(
    polygons: list[Polygon], target: np.ndarray, transform: Any,
) -> np.ndarray:
    """Average the intended cell size over each cell's footprint.

    ``cell_target_width_m`` was sampled from the size raster at a single point --
    the cell centroid.  That is fine while cells are small relative to the size
    field, and wrong once they are not: a large background cell spanning the
    30/60/120/200 m ladder beside a river reports whichever single rung its centre
    happened to land on.  On the Pune graded mesh that mislabelled 66,519 of the
    92,187 joins the adjacency check rejected -- 72% of them, where the real cells
    were within the 2:1 rule all along.

    Averaging over the footprint keeps the meaning of the variable (the intended
    local resolution) and makes it describe the whole cell instead of one point.
    Cells smaller than a pixel keep the point sample, which is exact for them.
    """
    objects = np.asarray(polygons, dtype=object)
    centres = centroid(objects)
    rows, cols = rasterio.transform.rowcol(transform, get_x(centres), get_y(centres))
    rows = np.clip(np.asarray(rows, dtype=np.int64), 0, target.shape[0] - 1)
    cols = np.clip(np.asarray(cols, dtype=np.int64), 0, target.shape[1] - 1)
    result = target[rows, cols].astype(np.float64)
    if not len(polygons):
        return result
    # One rasterisation gives every pixel its owning cell; a bincount then averages
    # the size field per cell in a single pass over the grid.
    ownership = rasterize(
        ((polygon, index) for index, polygon in enumerate(polygons)),
        out_shape=target.shape, transform=transform, fill=-1, dtype="int32",
        all_touched=False,
    )
    covered = ownership >= 0
    if not covered.any():
        return result
    owner = ownership[covered].astype(np.int64)
    values = target[covered].astype(np.float64)
    count = np.bincount(owner, minlength=len(polygons))
    total = np.bincount(owner, weights=values, minlength=len(polygons))
    sampled = count > 0
    result[sampled] = total[sampled] / count[sampled]
    return result


def _write_ugrid(
    path: Path, faces: list[list[tuple[float, float]]], edges: dict[Any, list[int]], polygons: list[Polygon], crs: Any,
    target: np.ndarray, transform: Any, dem: np.ndarray, feature_classes: list[str],
    river_fraction: np.ndarray, river_channel_fraction: np.ndarray, floodplain_fraction: np.ndarray,
    waterbody_fraction: np.ndarray | None = None, river_strip: dict[str, np.ndarray] | None = None,
    cell_target_width: np.ndarray | None = None,
) -> dict[str, float]:
    export_report: dict[str, float] = {
        "merged_internal_faces": 0, "non_collinear_merged_faces": 0,
        "merged_face_length_deficit_m": 0.0,
    }
    nodes = sorted({point for face in faces for point in face})
    node_id = {point: index for index, point in enumerate(nodes)}
    connectivity = np.full((len(faces), max(map(len, faces))), -1, dtype=np.int32)
    for index, face in enumerate(faces):
        connectivity[index, : len(face)] = [node_id[point] for point in face]
    # Drop faces with no length.  Rounding rings onto the export grid can leave a
    # segment whose two endpoints coincide; it carries no flux because it has no
    # area, but a solver that divides by face length gets NaN from it.  Two such
    # faces killed the first Pune simulation at t=3 s.  ``_edge_quality`` cannot
    # warn about them because it discards sub-1e-8 segments before pairing, so
    # they are counted here and reported in QA instead.
    face_length_floor = 1e-6
    usable_edges = [
        (key, owners) for key, owners in edges.items()
        if len(owners) <= 2
        and np.hypot(key[0][0] - key[1][0], key[0][1] - key[1][1]) > face_length_floor
    ]
    zero_length_faces = sum(
        1 for key, owners in edges.items()
        if len(owners) <= 2
        and np.hypot(key[0][0] - key[1][0], key[0][1] - key[1][1]) <= face_length_floor
    )
    owner = np.asarray([owners[0] for _, owners in usable_edges], dtype=np.int32)
    neighbour = np.asarray([owners[1] if len(owners) == 2 else -1 for _, owners in usable_edges], dtype=np.int32)
    length = np.asarray([np.hypot(a[0] - b[0], a[1] - b[1]) for (a, b), _ in usable_edges])
    centres = np.asarray([(item.centroid.x, item.centroid.y) for item in polygons])
    areas = np.asarray([item.area for item in polygons])
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
        [{"rural": 0, "urban": 1, "floodplain": 2, "river": 3, "waterbody": 4}[item] for item in feature_classes],
        dtype=np.int16,
    )
    if waterbody_fraction is None:
        waterbody_fraction = np.zeros(len(faces), dtype=np.float64)
    # Orient each face normal from the owner's ring winding, never from its
    # centroid.  Clipping the river channel out of a background cell leaves
    # L-shaped and crescent cells whose centroid sits outside the cell -- often
    # inside the very neighbour across the face being oriented.  The old
    # centroid dot-product then reversed that normal, so the cell's outward
    # normals no longer summed to zero and a uniform flux had a non-zero
    # divergence.  Ring winding is exact for any simple polygon, convex or not.
    ring_normal: dict[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]] = {}
    for face in faces:
        ring = np.asarray(face, dtype=np.float64)
        following = np.roll(ring, -1, axis=0)
        signed_area = 0.5 * float(np.sum(
            ring[:, 0] * following[:, 1] - following[:, 0] * ring[:, 1]
        ))
        winding = 1.0 if signed_area > 0 else -1.0
        for start, end in zip(face, face[1:] + face[:1]):
            if start == end:
                continue
            key = tuple(sorted((start, end)))
            if key in ring_normal:
                continue  # the first face to register an edge is its ``owners[0]``
            dx, dy = end[0] - start[0], end[1] - start[1]
            span = max(np.hypot(dx, dy), 1e-12)
            ring_normal[key] = (winding * dy / span, -winding * dx / span)
    normal_x = np.empty(len(usable_edges)); normal_y = np.empty(len(usable_edges))
    for index, (key, owners) in enumerate(usable_edges):
        oriented = ring_normal.get(tuple(sorted(key)))
        if oriented is None:
            raise ValueError(
                f"Face {key} is in the edge table but is not a ring segment of any face. "
                "``edges`` must come from ``_topology(polygons)`` so every face has an "
                "owner ring to orient its normal from."
            )
        normal_x[index], normal_y[index] = oriented
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
        # The merged face must carry the vector sum of its segments, not their
        # length-weighted mean direction paired with their summed length.  With
        # the mean, a pair that meets along two segments pointing different ways
        # exports a face whose L*n does not equal sum(L_i * n_i), so the outward
        # normals of the cell no longer sum to zero and the divergence of a
        # uniform flux is non-zero -- the scheme stops being conservative.  For
        # genuinely collinear segments, which is the common case this merge was
        # written for, |sum(L_i * n_i)| == sum(L_i) and nothing changes.
        merged_normal_x = np.add.reduceat(normal_x[internal_index] * weights, starts)
        merged_normal_y = np.add.reduceat(normal_y[internal_index] * weights, starts)
        projected_length = np.hypot(merged_normal_x, merged_normal_y)
        normal_scale = np.maximum(projected_length, 1e-12)
        merged_normal_x /= normal_scale; merged_normal_y /= normal_scale
        merged_distance = center_distance[internal_index[starts]]
        # Where the segments are not collinear the conservative face is narrower
        # than the wetted interface.  Report it rather than let conveyance quietly
        # disappear: a large figure here means the mesh has cell pairs meeting on
        # two sides, which is a mesh-quality problem, not an export detail.
        non_collinear = projected_length < 0.999 * total_length
        export_report = {
            "merged_internal_faces": int(len(starts)),
            "non_collinear_merged_faces": int(non_collinear.sum()),
            "merged_face_length_deficit_m": float((total_length - projected_length)[non_collinear].sum()),
        }
        total_length = projected_length
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
        ds.schema_version = MESH_PRODUCT_SCHEMA
        ds.crs_wkt = crs.to_wkt() if crs else ""
        ds.edge_normal_convention = EDGE_NORMAL_CONVENTION
        apply_contract_attributes(ds, product="hydraulic_mesh")
        ds.createDimension("face", len(faces)); ds.createDimension("node", len(nodes)); ds.createDimension("edge", len(owner)); ds.createDimension("max_face_nodes", connectivity.shape[1])
        topology = ds.createVariable("mesh2d", "i4")
        topology.cf_role = "mesh_topology"
        topology.topology_dimension = 2
        topology.node_coordinates = "mesh2d_node_x mesh2d_node_y"
        topology.face_node_connectivity = "mesh2d_face_nodes"
        topology.face_coordinates = "mesh2d_face_x mesh2d_face_y"
        face_nodes = ds.createVariable(
            "mesh2d_face_nodes", "i4", ("face", "max_face_nodes"), fill_value=-1,
        )
        face_nodes.start_index = 0
        face_nodes[:] = connectivity
        node_x = ds.createVariable("mesh2d_node_x", "f8", ("node",)); node_x.units = "m"; node_x[:] = [item[0] for item in nodes]
        node_y = ds.createVariable("mesh2d_node_y", "f8", ("node",)); node_y.units = "m"; node_y[:] = [item[1] for item in nodes]
        for name, value in {"cell_area_m2": areas, "cell_bed_elevation_m": dem[rows, cols], "mesh2d_face_x": centres[:, 0], "mesh2d_face_y": centres[:, 1], "cell_target_width_m": target[rows, cols] if cell_target_width is None else cell_target_width, "cell_cfl_width_m": control_width, "cell_hydraulic_roughness": np.full(len(faces), np.nan)}.items():
            variable = ds.createVariable(name, "f8", ("face",))
            variable.units = "s m-1/3" if name == "cell_hydraulic_roughness" else ("m2" if name == "cell_area_m2" else "m")
            variable[:] = value
        ds.createVariable("cell_refinement_source", "i2", ("face",))[:] = refinement_codes
        ds.createVariable("edge_owner", "i4", ("edge",))[:] = owner
        ds.createVariable("edge_neighbor", "i4", ("edge",))[:] = neighbour
        for name, value in {
            "edge_length_m": length,
            "edge_center_distance_m": center_distance,
            "edge_midpoint_x": midpoint_x,
            "edge_midpoint_y": midpoint_y,
        }.items():
            variable = ds.createVariable(name, "f8", ("edge",)); variable.units = "m"; variable[:] = value
        ds.createVariable("edge_normal_x", "f8", ("edge",))[:] = normal_x
        ds.createVariable("edge_normal_y", "f8", ("edge",))[:] = normal_y
        ds.createVariable("edge_boundary_type", "i2", ("edge",))[:] = 0
        ds.createVariable("cell_river_fraction", "f8", ("face",))[:] = river_fraction
        ds.createVariable("cell_river_channel_fraction", "f8", ("face",))[:] = river_channel_fraction
        ds.createVariable("cell_floodplain_fraction", "f8", ("face",))[:] = floodplain_fraction
        ds.createVariable("cell_waterbody_fraction", "f8", ("face",))[:] = waterbody_fraction
        ds.createVariable("cell_feature_class", str, ("face",))[:] = np.asarray(feature_classes, dtype=object)
        ds.zero_length_faces_dropped = int(zero_length_faces)
        # Reach/station identity of a structured river cell.  ``-1``/NaN marks a
        # cell that is not part of the ribbon, so a model can select resolved 2-D
        # river cells without re-deriving them from geometry.
        for name, value in (river_strip or {}).items():
            ds.createVariable(name, "i4" if np.issubdtype(value.dtype, np.integer) else "f8", ("face",))[:] = value
    export_report["minimum_exported_cfl_width_m"] = float(np.min(control_width))
    export_report["zero_length_faces_dropped"] = int(zero_length_faces)
    return export_report


def _write_mesh_class_figure(
    path: Path, polygons: list[Polygon], feature_classes: list[str], crs: Any,
    river_geometry, floodplain_geometry, title: str, bounds: tuple[float, float, float, float] | None = None,
    connector_axes: list[LineString] | None = None, waterbody_geometry=None,
) -> None:
    """Write a compact mesh-stage diagnostic using the same class palette."""
    colours = {"rural": "#ECECEC", "urban": "#E6A0A1", "floodplain": "#8CC5E3", "river": "#0B81A2", "waterbody": "#2F80ED"}
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
    if waterbody_geometry is not None and not waterbody_geometry.is_empty:
        gpd.GeoSeries([waterbody_geometry.boundary], crs=crs).plot(ax=ax, color="#0B3D91", linewidth=0.85)
    if connector_axes:
        gpd.GeoSeries(connector_axes, crs=crs).plot(ax=ax, color="#9D2C00", linewidth=0.7, alpha=0.8)
    present = [item for item in ("waterbody", "river", "floodplain", "urban", "rural") if item in set(feature_classes)]
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
    stage_times: dict[str, float] = {}
    last_mark = started_at

    # A full-catchment build takes minutes.  ``HYDROBATHY_MESH_PROGRESS=1`` streams
    # the stage timings as they happen, which is the difference between watching a
    # build and guessing which stage it is stuck in.  Off by default.
    progress = bool(os.environ.get("HYDROBATHY_MESH_PROGRESS"))

    def mark(stage: str) -> None:
        nonlocal last_mark
        now = perf_counter()
        stage_times[stage] = stage_times.get(stage, 0.0) + now - last_mark
        if progress:
            print(
                f"[mesh] {stage}: {now - last_mark:.1f}s (elapsed {now - started_at:.1f}s)",
                flush=True,
            )
        last_mark = now

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
    waterbody_geometry = _vector_feature_geometry(
        config.waterbody_vector, crs, domain, config.waterbody_layer,
    )
    waterbody = (
        valid & geometry_mask([waterbody_geometry], out_shape=dem.shape, transform=transform, invert=True)
        if not waterbody_geometry.is_empty else np.zeros_like(valid, dtype=bool)
    )
    mark("inputs_and_domain")
    has_supplied_direction = np.isfinite(supplied_direction).any()
    if _needs_d4_routing(config, has_supplied_direction):
        accumulation, receiver = compute_d4_flow_accumulation(dem.astype("float32"), profile, nodata=np.nan)
    else:
        accumulation = None
        receiver = (
            receiver_from_d8_direction(supplied_direction)
            if config.river_source == "hydrobathydem_d8"
            else receiver_from_d4_direction(supplied_direction)
        )
    pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    if config.river_source in {"hydrobathydem_d4", "hydrobathydem_d8"}:
        river = valid & ~waterbody & ((supplied_mask > 0) if np.isfinite(supplied_mask).any() else (np.isfinite(width) & (width > 0)))
        if not river.any():
            raise ValueError(
                f"rivers.source={config.river_source!r} requires an aligned river_mask or positive river_width raster."
            )
        if np.isfinite(supplied_direction).any():
            receiver = (
                receiver_from_d8_direction(supplied_direction)
                if config.river_source == "hydrobathydem_d8"
                else receiver_from_d4_direction(supplied_direction)
            )
    else:
        if accumulation is None:
            raise AssertionError("D4 accumulation is required for derived river meshes.")
        river = valid & ~waterbody & (accumulation * pixel_area_km2 >= config.minimum_upstream_area_km2)
    river_rows, river_cols = np.where(river)
    river_x, river_y = xy(transform, river_rows, river_cols, offset="center")
    river_class_points = shapely_points(np.asarray(river_x), np.asarray(river_y))
    if _needs_reach_labels(config):
        hand, reach = connected_hand_with_reach(dem, receiver, river)
    else:
        hand = reach = None
    mark("routing_and_river_mask")
    # A HydroBathyDEM D4 case already carries LiN-derived width/depth.  Its
    # conditioned D4 geometry is the one routing, corridor, and mesh must
    # share; using a second vector centreline here creates visible detached
    # overbank corridors.
    if config.river_network is not None and config.river_source == "derive_d4":
        if reach is None:
            raise AssertionError("Reach labels are required for vector river orientation.")
        river_records = _vector_river_records(
            config.river_network, config.river_width_field, crs, domain, dem, transform,
        )
        reach_orientation = _vector_reach_orientations(reach, transform, river_records)
    else:
        river_records = _river_segment_records(
            river,
            receiver,
            transform,
            width,
            config.river_centerline_smoothing_iterations,
            config.river_width_smoothing_window_cells,
        )
        reach_orientation = {record[0]: record[3] for record in river_records}
    physical_river_geometry = _physical_river_geometry(river_records).difference(waterbody_geometry).buffer(0)
    # The structured ribbon defines its own computational channel: the polygon
    # is the union of the cells that will exist, so no cell can be classified
    # river without being one.  The old buffered-centreline channel remains for
    # the ``voronoi`` style, where classification still happens after the fact.
    strip_cells: list[Polygon] = []
    strip_attributes: list[dict[str, Any]] = []
    strip_banks: list[LineString] = []
    strip_discarded_area_m2 = 0.0
    graded_channel_seeds = np.empty((0, 2), dtype=np.float64)
    graded_bank_seeds = np.empty((0, 2), dtype=np.float64)
    graded_band_extent_m = 0.0
    if config.river_cell_style == "graded_rows":
        reach_records = _river_reach_records(
            river, receiver, transform, width,
            config.river_centerline_smoothing_iterations,
            config.river_width_smoothing_window_cells,
        )
        graded_channel_seeds, graded_bank_seeds, graded_band_extent_m = river_graded_seeds(
            reach_records,
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            config.minimum_hydraulic_width_m,
            max(config.background_width_m, config.urban_width_m, config.waterbody_target_width_m),
            config.maximum_adjacent_size_ratio,
            domain,
            waterbody_geometry,
            minimum_cell_scale_m=0.5 * config.minimum_hydraulic_width_m,
        )
        if not len(graded_channel_seeds):
            raise ValueError(
                "rivers.cell_style='graded_rows' produced no channel seeds. Check that the "
                "river mask, flow direction, and width rasters overlap the mesh domain."
            )
        # The channel polygon is now only used to classify finished cells; it is not
        # cut into anything, so it can keep the mapped-width definition.
        river_geometry = _physical_river_geometry(
            river_records, config.river_cross_river_target_width_m,
        ).difference(waterbody_geometry).buffer(0)
    elif config.river_cell_style == "structured_strip":
        if config.river_network is not None and config.river_source == "derive_d4":
            reach_records = [
                (reach_id, line, np.full(len(line.coords), reach_width, dtype=np.float64))
                for reach_id, line, reach_width, _ in river_records
            ]
        else:
            reach_records = _river_reach_records(
                river, receiver, transform, width,
                config.river_centerline_smoothing_iterations,
                config.river_width_smoothing_window_cells,
            )
        strip_cells, strip_attributes, strip_banks, strip_discarded_area_m2 = build_river_strip_mesh(
            reach_records,
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            config.minimum_hydraulic_width_m,
            domain,
            waterbody_geometry,
            minimum_cell_scale_m=0.5 * config.minimum_hydraulic_width_m,
        )
        if not strip_cells:
            raise ValueError(
                "rivers.cell_style='structured_strip' produced no river cells. Check that the "
                "river mask, flow direction, and width rasters overlap the mesh domain."
            )
        river_geometry = unary_union(strip_cells).buffer(0)
    elif config.river_cell_style == "voronoi":
        river_geometry = (
            _physical_river_geometry(river_records, config.river_cross_river_target_width_m)
            if config.unresolved_policy == "none"
            else physical_river_geometry
        ).difference(waterbody_geometry).buffer(0)
    # A supplied design-flow mask supersedes HAND.  The external mask is an
    # input artifact with its own provenance, not an interpolation of HAND.
    # HAND remains the explicit fallback for legacy cases.
    supplied_floodplain_geometry = None
    if not config.floodplain_enabled:
        floodplain = np.zeros_like(valid, dtype=bool)
    elif config.floodplain_vector is not None:
        vector = gpd.read_file(config.floodplain_vector).to_crs(crs)
        supplied_floodplain_geometry = vector.geometry.union_all().intersection(domain).buffer(0)
        supplied_floodplain_geometry = supplied_floodplain_geometry.difference(waterbody_geometry).buffer(0)
        floodplain = valid & ~waterbody & geometry_mask([supplied_floodplain_geometry], out_shape=dem.shape, transform=transform, invert=True) & ~river
    elif config.floodplain_mask is not None:
        floodplain = valid & ~waterbody & (supplied_floodplain > 0) & ~river
    else:
        if hand is None or reach is None:
            raise AssertionError("HAND and reach labels are required for fallback floodplain delineation.")
        floodplain = valid & ~waterbody & (reach >= 0) & np.isfinite(hand) & (hand <= config.floodplain_hand_stage_m) & ~river
    # The source grid is 30 m.  A topology-preserving half-pixel simplify is a
    # mesh-interface operation, not a change to the selected HAND cells: it
    # removes only sub-cell D4 stair steps that otherwise make zero-length
    # breakline fragments.  The physical width remains controlled by the
    # mapped-width buffered centreline above.
    interface_tolerance = 0.5 * min(abs(transform.a), abs(transform.e))
    physical_river_geometry = physical_river_geometry.simplify(interface_tolerance, preserve_topology=True)
    # A structured channel outline must keep every station vertex: those are the
    # exact points the background remnants have to meet for the interface to
    # pair up.  Only the buffered-centreline channel is simplified.
    if config.river_cell_style not in {"structured_strip"}:
        river_geometry = river_geometry.simplify(interface_tolerance, preserve_topology=True)
    waterbody_geometry = waterbody_geometry.simplify(2.0 * config.minimum_hydraulic_width_m, preserve_topology=True).buffer(0)
    floodplain_geometry = (
        Polygon()
        if not config.floodplain_enabled
        else exclusive_floodplain_geometry(
            supplied_floodplain_geometry, river_geometry, waterbody_geometry,
        )
        if supplied_floodplain_geometry is not None
        else exclusive_floodplain_geometry(
            _mask_geometry(floodplain, transform), river_geometry, waterbody_geometry,
        ).simplify(interface_tolerance, preserve_topology=True)
    )
    urban = valid & ((impervious >= config.impervious_threshold_percent) | (population >= config.population_threshold_per_km2))
    if config.urban_buffer_m > 0 and urban.any():
        urban |= distance_transform_edt(~urban, sampling=(abs(transform.e), abs(transform.a))) <= config.urban_buffer_m
    urban &= valid
    target = np.full(dem.shape, config.background_width_m, dtype=np.float64)
    target[floodplain] = np.minimum(target[floodplain], config.floodplain_target_width_m)
    target[urban] = np.minimum(target[urban], config.urban_width_m)
    target[waterbody] = config.waterbody_target_width_m
    # At least one 2-D strip is possible only at the DEM-scale floor.  Smaller
    # mapped channels keep their width/depth in Neal rather than being widened.
    resolved_river = (
        river
        if config.unresolved_policy == "none"
        else river & np.isfinite(width) & (width >= config.minimum_hydraulic_width_m)
    )
    target[resolved_river] = np.minimum(target[resolved_river], config.river_cross_river_target_width_m)
    if config.river_cell_style == "graded_rows" and resolved_river.any():
        # Record the graded band in the size field, not just in the seed layout.
        # The seeds already step 30 -> 60 -> 120 -> 200 m out from the bank, but
        # ``cell_target_width_m`` is what the adjacency check reads, and leaving it
        # at "30 m channel, 200 m rural" reported a 6.67 jump that the mesh no
        # longer has.  Distance from the channel gives the same ladder the rows use.
        spacing = (abs(transform.a), abs(transform.e))
        distance = distance_transform_edt(~resolved_river, sampling=spacing)
        edge = 0.5 * config.river_cross_river_target_width_m
        previous = config.river_cross_river_target_width_m
        for row_width in graded_row_plan(
            config.river_cross_river_target_width_m,
            max(config.background_width_m, config.urban_width_m, config.waterbody_target_width_m),
            config.maximum_adjacent_size_ratio,
        ):
            edge += 0.5 * (previous + row_width)
            band = valid & ~resolved_river & (distance <= edge)
            target[band] = np.minimum(target[band], row_width)
            previous = row_width
    target = np.maximum(target, config.minimum_hydraulic_width_m)
    target = _smooth_size_field(target, valid, config.maximum_adjacent_size_ratio)
    mark("feature_geometries_and_target")
    xmin, ymin, xmax, ymax = domain.bounds
    # 1. Build the physical river and its reach-specific HAND corridor first.
    # These sites are locked.  No urban/rural candidate may subsequently enter
    # either footprint.  2. Add paired support sites around the hard boundaries
    # so their later split does not create the tiny Voronoi faces that previously
    # controlled the timestep.
    # Under the structured style the channel is not seeded at all: the ribbon is
    # built explicitly and the background diagram only has to stop at its banks.
    # This is the change that stops a single river seed from claiming a large
    # off-channel Voronoi polygon.
    river_ribbon = (
        graded_channel_seeds
        if config.river_cell_style == "graded_rows"
        else np.empty((0, 2), dtype=np.float64)
        if config.river_cell_style == "structured_strip"
        else _river_core_seeds(
            river_records,
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            0.0 if config.unresolved_policy == "none" else config.minimum_hydraulic_width_m,
            config.minimum_hydraulic_width_m,
        )
    )
    river_bank_support = (
        graded_bank_seeds
        if config.river_cell_style == "graded_rows"
        else _river_bank_support_seeds(
            river_records,
            config.river_along_river_cell_length_m,
            config.river_cross_river_target_width_m,
            config.minimum_hydraulic_width_m,
        )
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
        if reach is None:
            raise AssertionError("Reach labels are required for flow-aligned floodplain seeds.")
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
    waterbody_ribbon = _lattice(
        domain.bounds, config.waterbody_target_width_m, waterbody, transform,
    )
    feature_geometry = unary_union([river_geometry, floodplain_geometry, waterbody_geometry])
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
    # The graded rows already fill the corridor out to ``graded_band_extent_m``, so
    # the background lattice must stay clear of all of it or the two interleave and
    # make exactly the short faces the grading exists to avoid.
    transition_buffer = max(0.5 * config.floodplain_target_width_m, graded_band_extent_m)
    transition_exclusion = valid & geometry_mask(
        [feature_geometry.buffer(transition_buffer)],
        out_shape=dem.shape, transform=transform, invert=True,
    ) if not feature_geometry.is_empty else reservation
    hard_boundary_parts = []
    if config.enforce_feature_boundaries or config.enforce_river_boundaries:
        hard_boundary_parts.append(river_geometry.boundary)
    if config.enforce_feature_boundaries:
        hard_boundary_parts.append(floodplain_geometry.boundary)
    if config.enforce_waterbody_boundaries:
        hard_boundary_parts.append(waterbody_geometry.boundary)
    hard_boundaries = unary_union(hard_boundary_parts) if hard_boundary_parts else LineString()
    support = (
        _boundary_support_seeds(
            hard_boundaries,
            spacing_m=min(config.floodplain_target_width_m, config.urban_width_m),
            offset_m=0.5 * config.floodplain_target_width_m,
            domain=domain,
        )
        if not hard_boundaries.is_empty else np.empty((0, 2), dtype=np.float64)
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
    river_seed_floor = config.minimum_hydraulic_width_m
    river_ribbon = _minimum_separated(river_ribbon, river_seed_floor)
    if len(river_bank_support):
        river_bank_support = np.asarray(
            [
                point for point in _minimum_separated(river_bank_support, river_seed_floor)
                if domain.covers(Point(point)) and not waterbody_geometry.covers(Point(point))
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        river_bank_support = _drop_near(river_bank_support, river_ribbon, river_seed_floor)
    river_support_blockers = (
        np.vstack([item for item in (river_ribbon, river_bank_support) if len(item)])
        if any(len(item) for item in (river_ribbon, river_bank_support)) else np.empty((0, 2), dtype=np.float64)
    )
    floodplain_ribbon = _drop_near(
        _minimum_separated(floodplain_ribbon, config.minimum_hydraulic_width_m),
        river_support_blockers, config.minimum_hydraulic_width_m,
    )
    waterbody_ribbon = _drop_near(
        _minimum_separated(waterbody_ribbon, config.minimum_hydraulic_width_m),
        np.vstack([item for item in (river_ribbon, river_bank_support, floodplain_ribbon) if len(item)])
        if any(len(item) for item in (river_ribbon, river_bank_support, floodplain_ribbon)) else np.empty((0, 2), dtype=np.float64),
        config.minimum_hydraulic_width_m,
    )
    support = _drop_near(
        _minimum_separated(support, config.minimum_hydraulic_width_m),
        np.vstack([item for item in (river_ribbon, river_bank_support, floodplain_ribbon, waterbody_ribbon) if len(item)])
        if any(len(item) for item in (river_ribbon, river_bank_support, floodplain_ribbon, waterbody_ribbon)) else np.empty((0, 2), dtype=np.float64),
        config.minimum_hydraulic_width_m,
    )
    feature_seeds = np.vstack([item for item in (river_ribbon, river_bank_support, floodplain_ribbon, waterbody_ribbon, support) if len(item)]) if any(len(item) for item in (river_ribbon, river_bank_support, floodplain_ribbon, waterbody_ribbon, support)) else np.empty((0, 2), dtype=np.float64)
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
    mark("seed_generation")
    points, polygons, quality_repair_iterations = _repair_short_cells(
        points, domain, config.minimum_hydraulic_width_m, max_iterations=config.quality_repair_max_iterations,
        protected_seed_count=0, target=target,
        transform=transform, minimum_face_length_factor=config.minimum_face_length_factor,
        minimum_center_distance_factor=config.minimum_center_distance_factor,
        protected_seed_indices=protected_core,
    )
    mark("initial_voronoi_and_seed_repair")
    step5_classes = _feature_classes(
        polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
        waterbody_geometry=waterbody_geometry,
    )
    step5_figure = config.out_dir / "diagnostics" / "mesh_step5_global_voronoi.png"
    if config.write_diagnostics:
        _write_mesh_class_figure(
            step5_figure,
            polygons, step5_classes, crs, river_geometry, floodplain_geometry,
            "Step 5 — river, urban, and rural Voronoi mesh"
            if not config.floodplain_enabled
            else "Step 5 — river-aligned and isotropic floodplain Voronoi mesh",
            connector_axes=connector_axes, waterbody_geometry=waterbody_geometry,
        )
        if config.diagnostic_window is not None:
            diagnostic = gpd.read_file(config.diagnostic_window).to_crs(crs).geometry.union_all()
            detail_indices = [index for index, polygon in enumerate(polygons) if polygon.intersects(diagnostic)]
            _write_mesh_class_figure(
                config.out_dir / "diagnostics" / "mesh_step5_detail.png",
                [polygons[index] for index in detail_indices], [step5_classes[index] for index in detail_indices], crs,
                river_geometry.intersection(diagnostic), floodplain_geometry.intersection(diagnostic),
                "Step 5 — detailed global Voronoi candidate", diagnostic.bounds, connector_axes,
                waterbody_geometry.intersection(diagnostic),
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
                    waterbody_geometry.intersection(geometry),
                )
    mark("step5_classification_and_diagnostics")
    if config.stop_after_step == 5:
        return {
            "status": "step5_diagnostic_only",
            "step5_figure": str(step5_figure) if config.write_diagnostics else None,
            "cells": len(polygons),
        }
    # 4. Convert both physical feature boundaries into common finite-volume
    # faces.  Because exactly the same lines split every affected polygon, the
    # result is conforming; cells are classified only after this operation.
    if not hard_boundaries.is_empty:
        polygons = _split_by_feature_boundaries(
            polygons, hard_boundaries, 2.0 * config.minimum_hydraulic_width_m ** 2,
        )
    if config.enforce_waterbody_boundaries and not waterbody_geometry.is_empty:
        # A reservoir shoreline is a hard constraint, not a preference.
        # _split_by_feature_boundaries refuses any cut that would leave a piece
        # under 2*minimum_hydraulic_width^2 (1,800 m2 here), so a cell clipping the
        # shore by less than that simply kept straddling: 566 cells did, one by 10%
        # of its area. Clipping with no area floor makes every cell wholly wet or
        # wholly dry. The slivers this creates are then merged, but the merge groups
        # below keep them on their own side of the shore, so the constraint holds.
        polygons = _clip_cells_to_feature_polygon(polygons, waterbody_geometry, 0.0)
    if config.unresolved_policy == "none" and config.river_cell_style != "structured_strip":
        polygons = _clip_cells_to_feature_polygon(
            polygons, river_geometry, 0.25 * config.minimum_hydraulic_width_m ** 2,
        )
    def classify_current_polygons(current: list[Polygon]) -> list[str]:
        current_tree = STRtree(current)
        return _fraction_refined_feature_classes(
            _feature_classes(
                current, river_geometry, floodplain_geometry, urban, transform, river_class_points,
                waterbody_geometry=waterbody_geometry,
            ),
            # The ribbon cells are the channel's own disjoint decomposition, so
            # passing them keeps both overlay operands small; querying the merged
            # 1,400 km channel instead made this call dominate the build.
            _geometry_area_fractions(current, river_geometry, current_tree, strip_cells),
            _geometry_area_fractions(current, floodplain_geometry, current_tree),
            _geometry_area_fractions(current, waterbody_geometry, current_tree),
        )

    feature_classes = classify_current_polygons(polygons)
    mark("feature_split_and_classification")
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
        merge_groups = _merge_group_keys(feature_classes)
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
        feature_classes = _feature_classes(
            polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
            waterbody_geometry=waterbody_geometry,
        )
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
        feature_classes = _feature_classes(
            polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
            waterbody_geometry=waterbody_geometry,
        )
    mark("post_split_quality_repair")
    # Always retain the post-split candidate figure.  When the quality gate
    # rejects it, this is the evidence needed to diagnose the problematic
    # transition geometry; it is not a model-ready mesh artifact.
    final_figure = config.out_dir / "diagnostics" / "mesh_step6_final.png"
    if config.write_diagnostics:
        _write_mesh_class_figure(
            final_figure,
            polygons, feature_classes, crs, river_geometry, floodplain_geometry,
            "Final Pune adaptive Voronoi mesh",
            waterbody_geometry=waterbody_geometry,
        )
        if config.diagnostic_window is not None:
            diagnostic = gpd.read_file(config.diagnostic_window).to_crs(crs).geometry.union_all()
            detail_indices = [index for index, polygon in enumerate(polygons) if polygon.intersects(diagnostic)]
            _write_mesh_class_figure(
                config.out_dir / "diagnostics" / "mesh_step6_final_detail.png",
                [polygons[index] for index in detail_indices], [feature_classes[index] for index in detail_indices], crs,
                river_geometry.intersection(diagnostic), floodplain_geometry.intersection(diagnostic),
                "Final adaptive mesh — river, urban, and rural cells"
                if not config.floodplain_enabled
                else "Final adaptive mesh — river, floodplain, urban, and rural cells",
                diagnostic.bounds,
                waterbody_geometry=waterbody_geometry.intersection(diagnostic),
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
                    waterbody_geometry=waterbody_geometry.intersection(geometry),
                )
    mark("step6_diagnostics")
    small_cell_agglomerations = 0
    for _ in range(config.quality_repair_max_iterations):
        polygon_objects = np.asarray(polygons, dtype=object)
        preflight_scale = 2.0 * np.asarray([item.area for item in polygon_objects]) / np.maximum(
            np.asarray([item.length for item in polygon_objects]), 1e-12,
        )
        bad = np.flatnonzero(preflight_scale < 0.5 * config.minimum_hydraulic_width_m)
        if not len(bad):
            break
        repair_pairs = _undersized_cell_repair_pairs(
            polygons, bad, feature_classes, same_class_only=False,
        )
        polygons, merged = _agglomerate_short_face_cells(
            polygons,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            0.5 * config.minimum_hydraulic_width_m,
            max_merges=len(repair_pairs),
            classes=None,
            candidate_pairs=repair_pairs,
        )
        if not merged:
            polygons, merged = _agglomerate_short_face_cells(
                polygons,
                config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
                config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
                0.5 * config.minimum_hydraulic_width_m,
                max_merges=max(len(bad), 1),
                classes=None,
            )
        if not merged:
            break
        small_cell_agglomerations += merged
        feature_classes = _feature_classes(
            polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
            waterbody_geometry=waterbody_geometry,
        )
    residual_face_agglomerations = 0
    for _ in range(config.quality_repair_max_iterations):
        preflight_edge_qa = _edge_quality(
            polygons, points, target, transform,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
        )
        if not (
            preflight_edge_qa["short_internal_face_count"]
            or preflight_edge_qa["close_internal_center_count"]
        ):
            break
        polygons, merged = _agglomerate_short_face_cells(
            polygons,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            0.5 * config.minimum_hydraulic_width_m,
            max_merges=len(preflight_edge_qa["bad_owners"]),
            classes=None,
            candidate_pairs=preflight_edge_qa["bad_owners"],
        )
        if not merged:
            break
        residual_face_agglomerations += merged
        feature_classes = _feature_classes(
            polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
            waterbody_geometry=waterbody_geometry,
        )
    # 5. Swap the structured ribbon in for the channel corridor of the
    # background diagram, then restore conformity.  Every polygon surgery in
    # this build -- feature splitting, channel clipping, and the agglomeration
    # passes above -- leaves T-junctions that ``_topology`` cannot pair, so the
    # insertion pass runs over the whole mesh rather than only the river.
    hanging_nodes_inserted = 0
    hanging_nodes_snapped = 0
    degenerate_cells_absorbed = 0
    short_faces_welded = 0
    pinched_cells_split = 0
    partition_voids_filled = 0
    partition_void_area_m2 = 0.0
    partition_void_searches_skipped = 0
    conformity_repair_rounds = 0
    cross_feature_residual_merges = 0
    background_sliver_merges = 0
    minimum_face_length_m = config.minimum_face_length_factor * config.minimum_hydraulic_width_m
    if config.river_cell_style == "structured_strip":
        polygons, _ = substitute_river_strip(polygons, river_geometry, strip_cells)
        polygons, repair = conform_and_repair_cells(
            polygons, minimum_face_length_m, 0.5 * config.minimum_hydraulic_width_m, domain,
        )
        if progress:
            print(f"[mesh]   repair {repair}", flush=True)
        hanging_nodes_inserted += repair["inserted"]
        hanging_nodes_snapped += repair["snapped"]
        short_faces_welded += repair["welded"]
        pinched_cells_split += repair["pinched"]
        degenerate_cells_absorbed += repair["absorbed"]
        partition_voids_filled += repair["voids_filled"]
        partition_void_area_m2 += repair["void_area_m2"]
        partition_void_searches_skipped += repair["void_search_skipped"]
        conformity_repair_rounds += repair["rounds"]
        feature_classes = classify_current_polygons(polygons)
        # Cutting the channel out leaves thin remnants against the bank.  Merge
        # them into their background neighbours only: the ribbon is immutable
        # feature geometry and must not be absorbed into a surface cell.
        previous_bad_pairs: int | None = None
        escalated = False
        # Give up after a few iterations that change nothing.  Without this the
        # loop runs its full budget -- forty iterations at four minutes each on
        # the Pune mesh -- to arrive at the same two cells it had at iteration
        # four.  The gate still reports them; it just does not cost three hours.
        stalled_iterations = 0
        previous_defect_count: tuple[int, int] | None = None
        for _ in range(config.quality_repair_max_iterations):
            polygon_objects = np.asarray(polygons, dtype=object)
            scale = 2.0 * np.asarray([item.area for item in polygon_objects]) / np.maximum(
                np.asarray([item.length for item in polygon_objects]), 1e-12,
            )
            repair_qa = _edge_quality(
                polygons, points, target, transform,
                config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
                config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            )
            bad = np.flatnonzero(scale < 0.5 * config.minimum_hydraulic_width_m)
            if not len(bad) and not len(repair_qa["bad_owners"]):
                break
            protected_groups = [
                item for item in _merge_group_keys(feature_classes)
            ]
            defect_count = (len(repair_qa["bad_owners"]), len(bad))
            stalled_iterations = (
                stalled_iterations + 1 if defect_count == previous_defect_count else 0
            )
            previous_defect_count = defect_count
            if stalled_iterations >= 3:
                break
            if progress:
                print(
                    f"[mesh]   bad pairs={len(repair_qa['bad_owners'])} "
                    f"undersized={len(bad)} "
                    f"short_faces={repair_qa['short_internal_face_count']} "
                    f"close_centres={repair_qa['close_internal_center_count']} "
                    f"self_paired={int((repair_qa['bad_owners'][:, 0] == repair_qa['bad_owners'][:, 1]).sum()) if len(repair_qa['bad_owners']) else 0} "
                    f"cells={len(polygons)}",
                    flush=True,
                )
            groups = [
                item for item in (
                    repair_qa["bad_owners"],
                    _undersized_cell_repair_pairs(polygons, bad, protected_groups, same_class_only=True)
                    if len(bad) else np.empty((0, 2), dtype=np.int64),
                    rerouted_repair_pairs(polygons, repair_qa["bad_owners"], protected_groups, "river"),
                ) if len(item)
            ]
            if not groups:
                break
            candidate_pairs = np.vstack(groups)
            # Protect the ribbon while the count is still coming down.  Once it
            # stops improving, the residue is pairs that no same-group merge can
            # reach -- a crescent remnant whose centre sits on top of a ribbon
            # cell's, which merging the two is the only local repair for.  The
            # escalation is counted so the cost to the ribbon is visible, and it
            # only ever touches the cells the gate would otherwise reject.
            # Sticky: once the count stops improving, stay escalated.  Toggling
            # it back off lets the next unescalated repair pass rebuild exactly
            # the pairs the escalated merge just removed, and the loop settles
            # into a cycle -- on the Pune mesh it repeated 163 bad pairs with
            # byte-identical weld and absorb counts for twenty iterations.
            escalated = escalated or (
                previous_bad_pairs is not None
                and len(repair_qa["bad_owners"]) >= previous_bad_pairs
            )
            previous_bad_pairs = len(repair_qa["bad_owners"])
            polygons, merged = _agglomerate_short_face_cells(
                polygons,
                config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
                config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
                0.5 * config.minimum_hydraulic_width_m,
                max_merges=len(candidate_pairs),
                classes=None if escalated else protected_groups,
                candidate_pairs=candidate_pairs,
            )
            if escalated:
                cross_feature_residual_merges += merged
            if not merged:
                # Nothing merged while the ribbon is protected.  That is exactly
                # what the escalation exists for -- a short face between a river
                # cell and a surface cell, which no same-group merge can reach --
                # so try once more without the protection before giving up.  The
                # count can fall to a single stubborn pair without ever
                # triggering the stall detector, which is how one face survived a
                # whole build.
                if escalated:
                    break
                escalated = True
                continue
            if progress:
                print(f"[mesh]   merged={merged} escalated={escalated}", flush=True)
            background_sliver_merges += merged
            mark("background_sliver_merge")
            polygons, repair = conform_and_repair_cells(
                polygons, minimum_face_length_m, 0.5 * config.minimum_hydraulic_width_m, domain,
                classes=list(feature_classes),
            )
            if progress:
                print(f"[mesh]   repair {repair}", flush=True)
            hanging_nodes_inserted += repair["inserted"]
            hanging_nodes_snapped += repair["snapped"]
            short_faces_welded += repair["welded"]
            pinched_cells_split += repair["pinched"]
            degenerate_cells_absorbed += repair["absorbed"]
            partition_voids_filled += repair["voids_filled"]
            partition_void_area_m2 += repair["void_area_m2"]
            partition_void_searches_skipped += repair["void_search_skipped"]
            conformity_repair_rounds += repair["rounds"]
            mark("background_sliver_repair")
            feature_classes = classify_current_polygons(polygons)
            mark("background_sliver_classify")
    else:
        polygons, repair = conform_and_repair_cells(
            polygons, minimum_face_length_m, 0.5 * config.minimum_hydraulic_width_m, domain,
        )
        hanging_nodes_inserted += repair["inserted"]
        hanging_nodes_snapped += repair["snapped"]
        short_faces_welded += repair["welded"]
        pinched_cells_split += repair["pinched"]
        degenerate_cells_absorbed += repair["absorbed"]
        partition_voids_filled += repair["voids_filled"]
        partition_void_area_m2 += repair["void_area_m2"]
        partition_void_searches_skipped += repair["void_search_skipped"]
        conformity_repair_rounds += repair["rounds"]
        feature_classes = classify_current_polygons(polygons)
    mark("river_strip_substitution")
    preflight_edge_qa = _edge_quality(
        polygons, points, target, transform,
        config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
        config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
    )
    polygon_objects = np.asarray(polygons, dtype=object)
    preflight_scale = 2.0 * np.asarray([item.area for item in polygon_objects]) / np.maximum(
        np.asarray([item.length for item in polygon_objects]), 1e-12,
    )
    mark("preflight_quality")
    if preflight_scale.min() < 0.5 * config.minimum_hydraulic_width_m:
        bad = np.flatnonzero(preflight_scale < 0.5 * config.minimum_hydraulic_width_m)
        bad_path = config.out_dir / "diagnostics" / "mesh_step6_bad_control_volumes.gpkg"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
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
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(
            {
                "face_id": bad,
                "feature_class": [feature_classes[index] for index in bad],
            },
            geometry=[polygons[index] for index in bad], crs=crs,
        ).to_file(bad_path, layer="bad_faces", driver="GPKG")
        # The owner list alone does not say which interface failed or why.  Write
        # the offending pairs with their face length and centre distance so the
        # cause is readable without re-running the build.
        # Write the owner geometry raw.  ``unary_union((cell, cell))`` silently
        # repairs a self-touching ring, so a union-based diagnostic reports a
        # clean polygon for exactly the defect being diagnosed -- which cost
        # real time here.  Same reason ``shared_face_length_m`` is left blank
        # for a self-paired face: intersecting a boundary with itself returns
        # the whole perimeter, not the offending segment.
        pair_geometry = []
        pair_rows: dict[str, list[Any]] = {
            "owner": [], "neighbour": [], "owner_class": [], "neighbour_class": [],
            "shared_face_length_m": [], "center_distance_m": [],
        }
        for left, right in np.unique(np.sort(preflight_edge_qa["bad_owners"], axis=1), axis=0):
            left, right = int(left), int(right)
            shared = polygons[left].intersection(polygons[right])
            pair_rows["owner"].append(left)
            pair_rows["neighbour"].append(right)
            pair_rows["owner_class"].append(feature_classes[left])
            pair_rows["neighbour_class"].append(feature_classes[right])
            pair_rows["shared_face_length_m"].append(
                float(getattr(shared, "length", 0.0)) if left != right else float("nan")
            )
            pair_rows["center_distance_m"].append(
                float(polygons[left].centroid.distance(polygons[right].centroid))
            )
            pair_geometry.append(polygons[left])
        gpd.GeoDataFrame(pair_rows, geometry=pair_geometry, crs=crs).to_file(
            bad_path, layer="bad_face_pairs", driver="GPKG",
        )
        rejected_path = config.out_dir / "reports" / "hybrid_mesh_qa_preflight.json"
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(json.dumps({
            "status": "step6_preflight_rejected",
            "cells": len(polygons),
            "generation_time_s": perf_counter() - started_at,
            "quality_repair_iterations": quality_repair_iterations,
            "short_face_agglomerations": short_face_merges,
            "cross_feature_short_face_agglomerations": cross_feature_short_face_merges,
            "small_cell_agglomerations": small_cell_agglomerations,
            "residual_face_agglomerations": residual_face_agglomerations,
            "timings_s": {key: round(value, 6) for key, value in stage_times.items()},
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
            "small_cell_agglomerations": small_cell_agglomerations,
            "residual_face_agglomerations": residual_face_agglomerations,
            "minimum_cfl_length_m": float(preflight_scale.min()),
            "p05_cfl_length_m": float(np.quantile(preflight_scale, 0.05)),
            "timings_s": {key: round(value, 6) for key, value in stage_times.items()},
            **{key: value for key, value in preflight_edge_qa.items() if key not in {"bad_owners", "short_owners", "face_seed", "face_target_m"}},
        }, indent=2), encoding="utf-8")
        return {
            "status": "step6_preflight_passed",
            "cells": len(polygons),
            "qa": str(qa_path),
            "step6_figure": str(final_figure) if config.write_diagnostics else None,
        }
    topology_normalization_merges = 0
    # ``exhausted`` exists so the gate below never judges a stale measurement.
    # ``final_edge_qa`` is taken at the top of an iteration, but the repair and
    # merge at the bottom replace ``polygons``; breaking straight out of the
    # bottom left the gate rejecting geometry that no longer existed, and the
    # failure diagnostic indexing the new cell list with the old indices.  Now a
    # stalled iteration loops once more to re-measure, then stops.
    exhausted = False
    normalization_escalated = False
    for iteration in range(config.quality_repair_max_iterations + 1):
        faces, edges = _topology(polygons)
        polygons = [Polygon(face) for face in faces]
        feature_classes = _feature_classes(
            polygons, river_geometry, floodplain_geometry, urban, transform, river_class_points,
            waterbody_geometry=waterbody_geometry,
        )
        final_edge_qa = _edge_quality(
            polygons, points, target, transform,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
        )
        if not final_edge_qa["short_internal_face_count"] and not final_edge_qa["close_internal_center_count"]:
            break
        if exhausted or iteration == config.quality_repair_max_iterations:
            break
        # Rounding the rings onto the 0.1 mm export grid can itself leave a
        # sub-floor segment.  Weld and absorb before merging: that repairs the
        # interface without consuming a well-formed cell, so only defects that
        # survive geometric repair cost a cell.
        # One round, not five.  The mesh has already cleared the preflight gate by
        # this point, so all this pass has to undo is what rounding the rings onto
        # the 0.1 mm export grid just introduced.  A full five-round repair here
        # costs ~250 s per iteration and finds almost nothing.
        polygons, repair = conform_and_repair_cells(
            polygons, minimum_face_length_m, 0.5 * config.minimum_hydraulic_width_m, domain,
            max_rounds=1,
        )
        hanging_nodes_inserted += repair["inserted"]
        hanging_nodes_snapped += repair["snapped"]
        short_faces_welded += repair["welded"]
        pinched_cells_split += repair["pinched"]
        degenerate_cells_absorbed += repair["absorbed"]
        partition_voids_filled += repair["voids_filled"]
        partition_void_area_m2 += repair["void_area_m2"]
        partition_void_searches_skipped += repair["void_search_skipped"]
        conformity_repair_rounds += repair["rounds"]
        # Deliberately no ``continue`` here.  Skipping the merge whenever the
        # repair did any work sounds conservative, but welding is almost always
        # non-zero, so the merge never ran and this loop just spun through five
        # repair rounds per iteration until it hit its cap -- 24 minutes with no
        # progress on the Pune build.  The loop re-measures at the top anyway, so
        # attempting the merge here costs nothing when there is nothing to merge.
        protection = (
            _merge_group_keys(feature_classes)
            if config.river_cell_style == "structured_strip" and not normalization_escalated
            else None
        )
        polygons, merged = _agglomerate_short_face_cells(
            polygons,
            config.minimum_face_length_factor * config.minimum_hydraulic_width_m,
            config.minimum_center_distance_factor * config.minimum_hydraulic_width_m,
            0.5 * config.minimum_hydraulic_width_m,
            max_merges=len(final_edge_qa["bad_owners"]),
            classes=protection,
            candidate_pairs=final_edge_qa["bad_owners"],
        )
        if merged and normalization_escalated:
            cross_feature_residual_merges += merged
        if not merged:
            # Same escalation the background loop uses, and for the same reason:
            # what survives here is a face between a ribbon cell and a surface
            # cell, which no same-group merge can reach.  On the Pune build this
            # was two pairs -- an urban and a rural cell each hugging the river
            # closely enough that their centres fell inside the floor -- and
            # without this the whole build was rejected for them.
            if normalization_escalated:
                exhausted = True
                continue
            normalization_escalated = True
            continue
        topology_normalization_merges += merged
        polygons, repair = conform_and_repair_cells(
            polygons, minimum_face_length_m, 0.5 * config.minimum_hydraulic_width_m, domain,
        )
        hanging_nodes_inserted += repair["inserted"]
        hanging_nodes_snapped += repair["snapped"]
        short_faces_welded += repair["welded"]
        pinched_cells_split += repair["pinched"]
        degenerate_cells_absorbed += repair["absorbed"]
        partition_voids_filled += repair["voids_filled"]
        partition_void_area_m2 += repair["void_area_m2"]
        partition_void_searches_skipped += repair["void_search_skipped"]
        conformity_repair_rounds += repair["rounds"]
    interior_unpaired_faces, interior_unpaired_length = internal_unpaired_edges(edges, domain)
    if final_edge_qa["short_internal_face_count"] or final_edge_qa["close_internal_center_count"]:
        # This gate previously raised with counts only.  Write the offending
        # pairs so a normalization failure is as diagnosable as a preflight one.
        bad_path = config.out_dir / "diagnostics" / "mesh_final_bad_faces.gpkg"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        pair_rows: dict[str, list[Any]] = {
            "owner": [], "neighbour": [], "owner_class": [], "neighbour_class": [],
            "shared_face_length_m": [], "center_distance_m": [], "owner_cfl_width_m": [],
            "owner_ring_vertices": [], "owner_is_valid": [],
        }
        pair_geometry = []
        for left, right in np.unique(np.sort(final_edge_qa["bad_owners"], axis=1), axis=0):
            left, right = int(left), int(right)
            pair_rows["owner"].append(left)
            pair_rows["neighbour"].append(right)
            pair_rows["owner_class"].append(feature_classes[left])
            pair_rows["neighbour_class"].append(feature_classes[right])
            pair_rows["shared_face_length_m"].append(
                float(polygons[left].boundary.intersection(polygons[right].boundary).length)
                if left != right else float("nan")
            )
            pair_rows["center_distance_m"].append(
                float(polygons[left].centroid.distance(polygons[right].centroid))
            )
            pair_rows["owner_cfl_width_m"].append(
                float(2.0 * polygons[left].area / max(polygons[left].length, 1e-12))
            )
            pair_rows["owner_ring_vertices"].append(
                int(len(polygons[left].exterior.coords) - 1)
            )
            pair_rows["owner_is_valid"].append(bool(polygons[left].is_valid))
            pair_geometry.append(polygons[left])
        gpd.GeoDataFrame(pair_rows, geometry=pair_geometry, crs=crs).to_file(
            bad_path, layer="final_bad_face_pairs", driver="GPKG",
        )
        raise ValueError(
            "Mesh quality failed after topology normalization: "
            f"{final_edge_qa['short_internal_face_count']} short internal faces and "
            f"{final_edge_qa['close_internal_center_count']} close centre pairs remain. "
            f"See {bad_path}."
        )
    mark("topology_normalization")
    gpkg = config.out_dir / "mesh" / "hydropol_hybrid_mesh.gpkg"
    ugrid = config.out_dir / "mesh" / "hydropol_hybrid_mesh.nc"
    gpkg.parent.mkdir(parents=True, exist_ok=True)
    polygon_tree = STRtree(polygons)
    river_fraction = _geometry_area_fractions(polygons, physical_river_geometry, polygon_tree)
    river_channel_fraction = _geometry_area_fractions(
        polygons, river_geometry, polygon_tree, strip_cells,
    )
    floodplain_fraction = _geometry_area_fractions(polygons, floodplain_geometry, polygon_tree)
    waterbody_fraction = _geometry_area_fractions(polygons, waterbody_geometry, polygon_tree)
    feature_class_fraction = 0.10
    river_channel_class_fraction = 0.5
    feature_classes = _fraction_refined_feature_classes(
        feature_classes, river_channel_fraction, floodplain_fraction, waterbody_fraction,
        feature_class_fraction, river_channel_class_fraction,
    )
    # Carry the ribbon's reach/station identity onto the exported cells so the
    # MATLAB and Python models can address a river cell by reach and station
    # rather than re-deriving it from geometry.  A repair pass may have merged
    # two strip cells, so the join is spatial rather than positional.
    reach_id = np.full(len(polygons), -1, dtype=np.int32)
    cross_band = np.full(len(polygons), -1, dtype=np.int32)
    station_start_m = np.full(len(polygons), np.nan)
    station_end_m = np.full(len(polygons), np.nan)
    channel_width_m = np.full(len(polygons), np.nan)
    river_index = np.flatnonzero(np.asarray(feature_classes, dtype=object) == "river")
    if strip_cells and len(river_index):
        representative = point_on_surface(np.asarray([polygons[item] for item in river_index], dtype=object))
        hits = STRtree(strip_cells).query(representative, predicate="within")
        matched: dict[int, int] = {}
        for point_index, strip_index in zip(hits[0], hits[1], strict=True):
            matched.setdefault(int(point_index), int(strip_index))
        for point_index, strip_index in matched.items():
            attribute = strip_attributes[strip_index]
            cell = int(river_index[point_index])
            reach_id[cell] = attribute["reach_id"]
            cross_band[cell] = attribute["cross_band"]
            station_start_m[cell] = attribute["station_start_m"]
            station_end_m[cell] = attribute["station_end_m"]
            channel_width_m[cell] = attribute["channel_width_m"]
    mark("feature_fractions")
    for sidecar in (gpkg, Path(f"{gpkg}-wal"), Path(f"{gpkg}-shm")):
        sidecar.unlink(missing_ok=True)
    gpd.GeoDataFrame(
        {"face_id": range(len(polygons)), "area_m2": [item.area for item in polygons], "feature_class": feature_classes,
         "river_fraction": river_fraction, "river_channel_fraction": river_channel_fraction,
         "floodplain_fraction": floodplain_fraction, "waterbody_fraction": waterbody_fraction,
         "reach_id": reach_id, "cross_band": cross_band, "station_start_m": station_start_m,
         "station_end_m": station_end_m, "channel_width_m": channel_width_m},
        geometry=polygons, crs=crs,
    ).to_file(gpkg, layer="mesh", driver="GPKG")
    if strip_banks:
        gpd.GeoDataFrame(
            {"bank_id": range(len(strip_banks))}, geometry=strip_banks, crs=crs,
        ).to_file(gpkg, layer="river_bank_breaklines", driver="GPKG")
    if not river_geometry.is_empty:
        gpd.GeoDataFrame(
            {"feature": ["computational_river_channel"]},
            geometry=[river_geometry],
            crs=crs,
        ).to_file(gpkg, layer="river_channel", driver="GPKG")
    if not physical_river_geometry.is_empty:
        gpd.GeoDataFrame(
            {"feature": ["physical_river_footprint"]},
            geometry=[physical_river_geometry],
            crs=crs,
        ).to_file(gpkg, layer="river_physical_footprint", driver="GPKG")
    mark("geopackage_write")
    cell_target_width = area_weighted_cell_target(polygons, target, transform)
    export_report = _write_ugrid(
        ugrid, faces, edges, polygons, crs, target, transform, dem, feature_classes,
        np.asarray(river_fraction), np.asarray(river_channel_fraction), np.asarray(floodplain_fraction),
        np.asarray(waterbody_fraction),
        cell_target_width=cell_target_width,
        river_strip={
            "cell_reach_id": reach_id, "cell_cross_band": cross_band,
            "cell_station_start_m": station_start_m, "cell_station_end_m": station_end_m,
            "cell_channel_width_m": channel_width_m,
        },
    )
    ugrid_control_width = np.asarray([export_report["minimum_exported_cfl_width_m"]])
    mark("ugrid_write")
    if config.write_diagnostics:
        _write_mesh_class_figure(
            config.out_dir / "diagnostics" / "hybrid_mesh_wireframe.png",
            polygons, feature_classes, crs, river_geometry, floodplain_geometry,
            "Step 6 — feature-boundary split and sliver repair",
            waterbody_geometry=waterbody_geometry,
        )
    mark("final_wireframe_diagnostic")
    degree = np.zeros(len(polygons), dtype=int)
    for owners in edges.values():
        for item in owners:
            degree[item] += 1
    perimeters = np.asarray([item.length for item in polygons])
    # 2A/P is a compactness measure, not the CFL length scale.  Both solvers use
    # ``cell_cfl_width_m`` -- the minimum face-normal centre separation -- which
    # ``_write_ugrid`` computes and reports back.  QA used to publish 2A/P under the
    # name ``minimum_cfl_length_m`` while the file carried the other number, so the
    # Pune report said 15.0 m for a mesh that actually shipped 7.39 m.  Both are
    # reported now, each under a name that says which one it is.
    cfl_width = 2.0 * np.asarray([item.area for item in polygons]) / np.maximum(perimeters, 1e-12)
    mesh_counts = Counter(feature_classes)
    unresolved_river_input_cells = int((river & ~resolved_river).sum())
    mark("summary_metrics")
    qa = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}, "cells": len(polygons), "edges": len(edges), "generation_time_s": perf_counter() - started_at, "timings_s": {key: round(value, 6) for key, value in stage_times.items()}, "mean_cell_degree": float(degree.mean()), "rural_quad_fraction": float(np.mean(degree == 4)), "minimum_cfl_length_m": float(np.min(ugrid_control_width)), "minimum_cell_compactness_2a_over_p_m": float(cfl_width.min()), "p05_cell_compactness_2a_over_p_m": float(np.quantile(cfl_width, 0.05)), "river_cells": int(mesh_counts["river"]), "urban_cells": int(mesh_counts["urban"]), "floodplain_cells": int(mesh_counts["floodplain"]), "waterbody_cells": int(mesh_counts["waterbody"]), "rural_cells": int(mesh_counts["rural"]), "river_input_raster_cells": int(river.sum()), "resolved_2d_river_input_cells": int(resolved_river.sum()), "unresolved_river_input_cells": unresolved_river_input_cells, "neal_subgrid_river_input_cells": unresolved_river_input_cells if config.unresolved_policy == "neal_subgrid" else 0, "urban_input_raster_cells": int(urban.sum()), "floodplain_input_raster_cells": int(floodplain.sum()), "waterbody_input_raster_cells": int(waterbody.sum()), "nonwater_cells_with_waterbody_fraction_gt_0p01": int(((np.asarray(feature_classes, dtype=object) != "waterbody") & (waterbody_fraction > 0.01)).sum()), "nonwater_cells_with_waterbody_fraction_gt_0p10": int(((np.asarray(feature_classes, dtype=object) != "waterbody") & (waterbody_fraction > 0.10)).sum()), "river_cells_with_waterbody_fraction_gt_0p01": int(((np.asarray(feature_classes, dtype=object) == "river") & (waterbody_fraction > 0.01)).sum()), "floodplain_cells_with_waterbody_fraction_gt_0p01": int(((np.asarray(feature_classes, dtype=object) == "floodplain") & (waterbody_fraction > 0.01)).sum()), "river_cells_with_channel_fraction_lt_0p5": int(((np.asarray(feature_classes, dtype=object) == "river") & (river_channel_fraction < river_channel_class_fraction)).sum()), "river_feature_seed_count": int(len(river_ribbon)), "river_bank_support_seed_count": int(len(river_bank_support)), "floodplain_feature_seed_count": int(len(floodplain_ribbon)), "waterbody_feature_seed_count": int(len(waterbody_ribbon)), "boundary_support_seed_count": int(len(support)), "reserved_feature_cells": int(reservation.sum()), "quality_repair_iterations": quality_repair_iterations, "short_face_agglomerations": short_face_merges, "cross_feature_short_face_agglomerations": cross_feature_short_face_merges, "small_cell_agglomerations": small_cell_agglomerations, "residual_face_agglomerations": residual_face_agglomerations, "topology_normalization_agglomerations": topology_normalization_merges, "river_cell_style": config.river_cell_style, "river_strip_cells_built": int(len(strip_cells)), "river_strip_reaches": int(len({item["reach_id"] for item in strip_attributes})) if strip_attributes else 0, "river_strip_channel_area_km2": float(river_geometry.area / 1e6), "river_strip_discarded_area_m2": float(strip_discarded_area_m2), "hanging_nodes_inserted": int(hanging_nodes_inserted), "hanging_nodes_snapped": int(hanging_nodes_snapped), "degenerate_cells_absorbed": int(degenerate_cells_absorbed), "short_faces_welded": int(short_faces_welded), "pinched_cells_split": int(pinched_cells_split), "partition_voids_filled": int(partition_voids_filled), "partition_void_area_m2": float(partition_void_area_m2), "partition_void_searches_skipped": int(partition_void_searches_skipped), "conformity_repair_rounds": int(conformity_repair_rounds), "cross_feature_residual_merges": int(cross_feature_residual_merges), "cells_with_exterior_centroid": int((~contains(np.asarray(polygons, dtype=object), centroid(np.asarray(polygons, dtype=object)))).sum()), "background_sliver_merges": int(background_sliver_merges), "interior_unpaired_face_count": int(interior_unpaired_faces), "interior_unpaired_face_length_m": float(interior_unpaired_length), "boundary_face_count": int(sum(1 for owners in edges.values() if len(owners) == 1)), "domain_perimeter_m": float(domain.length), **export_report, "cells_with_interior_ring": int(sum(1 for item in polygons if item.interiors)), "faces_with_more_than_two_owners": int(sum(1 for owners in edges.values() if len(owners) > 2)), "cell_area_sum_km2": float(sum(item.area for item in polygons) / 1e6), "domain_area_km2": float(domain.area / 1e6), **{key: value for key, value in final_edge_qa.items() if key not in {"bad_owners", "short_owners", "face_seed", "face_target_m"}}}
    qa_path = config.out_dir / "reports" / "hybrid_mesh_qa.json"; qa_path.parent.mkdir(parents=True, exist_ok=True); qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return {"ugrid": str(ugrid), "geopackage": str(gpkg), "qa": str(qa_path), **qa}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a parameterized hybrid quad/Voronoi HydroPol mesh.")
    parser.add_argument("--config", required=True, type=Path, help="JSON/TOML hybrid-mesh configuration.")
    args = parser.parse_args(argv)
    print(json.dumps(build_hybrid_mesh(HybridMeshConfig.from_mapping(load_config_file(args.config))), indent=2))


if __name__ == "__main__":
    main()
