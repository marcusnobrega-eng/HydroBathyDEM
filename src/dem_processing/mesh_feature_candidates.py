"""Build QA-only floodplain-refinement and breakline candidate layers."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from scipy.ndimage import binary_fill_holes, binary_opening, distance_transform_edt, label, maximum_filter, minimum_filter
from shapely.geometry import LineString, Polygon, shape
from shapely.ops import linemerge, polygonize, unary_union

from .config import load_config_file
from .hybrid_mesh import _line_parts, _river_segment_records, receiver_from_d4_direction, receiver_from_d8_direction


@dataclass(frozen=True)
class MeshFeatureCandidateConfig:
    dem: Path
    domain_vector: Path
    river_mask: Path
    river_direction: Path
    river_width: Path
    out_dir: Path
    river_upstream_area: Path | None = None
    routing_scheme: str = "d8"
    major_river_min_upstream_area_km2: float = 50.0
    major_river_min_width_m: float = 45.0
    floodplain_max_hand_m: float = 8.0
    floodplain_max_distance_m: float = 900.0
    floodplain_min_width_m: float = 0.0
    floodplain_min_area_m2: float = 100_000.0
    floodplain_max_hole_fill_area_m2: float = 0.0
    simplify_tolerance_m: float = 30.0
    waterbody_vector: Path | None = None
    fetch_osm_waterbodies: bool = False
    fetch_osm_barriers: bool = False
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    waterbody_min_area_m2: float = 20_000.0
    waterbody_max_distance_to_river_m: float = 300.0
    refine_waterbodies_from_dem: bool = False
    waterbody_refine_buffer_m: float = 300.0
    waterbody_refine_elevation_tolerance_m: float = 2.0
    waterbody_refine_relief_window_m: float = 90.0
    waterbody_refine_max_relief_m: float = 3.0
    waterbody_refine_min_area_ratio: float = 0.25
    waterbody_refine_max_area_ratio: float = 2.0
    barrier_min_length_m: float = 90.0
    barrier_filter_to_waterbody_edges: bool = True
    barrier_max_distance_to_waterbody_m: float = 120.0
    suppress_centerlines_inside_waterbodies: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "MeshFeatureCandidateConfig":
        values = dict(values)
        grouped = {
            name: values.pop(name, {})
            for name in ("inputs", "rivers", "floodplain", "breaklines", "osm")
        }
        for name, group in grouped.items():
            if group and not isinstance(group, dict):
                raise ValueError(f"{name} must be a configuration object.")
        values = {
            **grouped["inputs"],
            **grouped["rivers"],
            **grouped["floodplain"],
            **grouped["breaklines"],
            **grouped["osm"],
            **values,
        }
        aliases = {
            "output_dir": "out_dir",
            "river_upstream_area_path": "river_upstream_area",
            "waterbody_vector_path": "waterbody_vector",
            "minimum_upstream_area_km2": "major_river_min_upstream_area_km2",
        }
        values = {aliases.get(key, key): value for key, value in values.items()}
        for key in (
            "dem", "domain_vector", "river_mask", "river_direction", "river_width",
            "out_dir", "river_upstream_area", "waterbody_vector",
        ):
            if values.get(key) is not None:
                values[key] = Path(values[key])
        config = cls(**values)
        if config.routing_scheme not in {"d4", "d8"}:
            raise ValueError("routing_scheme must be 'd4' or 'd8'.")
        if config.floodplain_min_width_m < 0:
            raise ValueError("floodplain_min_width_m must be non-negative.")
        if config.floodplain_max_hole_fill_area_m2 < 0:
            raise ValueError("floodplain_max_hole_fill_area_m2 must be non-negative.")
        positive = (
            config.major_river_min_upstream_area_km2,
            config.major_river_min_width_m,
            config.floodplain_max_hand_m,
            config.floodplain_max_distance_m,
            config.floodplain_min_area_m2,
            config.waterbody_min_area_m2,
            config.waterbody_max_distance_to_river_m,
            config.waterbody_refine_buffer_m,
            config.waterbody_refine_elevation_tolerance_m,
            config.waterbody_refine_relief_window_m,
            config.waterbody_refine_max_relief_m,
            config.waterbody_refine_min_area_ratio,
            config.waterbody_refine_max_area_ratio,
            config.barrier_min_length_m,
            config.barrier_max_distance_to_waterbody_m,
        )
        if min(positive) <= 0 or config.simplify_tolerance_m < 0:
            raise ValueError("mesh feature candidate thresholds must be positive.")
        return config


def _aligned(path: Path | None, reference: rasterio.DatasetReader, default: float = np.nan) -> np.ndarray:
    if path is None:
        return np.full(reference.shape, default, dtype=np.float64)
    with rasterio.open(path) as source:
        if source.shape != reference.shape or source.transform != reference.transform or source.crs != reference.crs:
            raise ValueError(f"Input raster is not aligned with the DEM grid: {path}")
        return source.read(1, masked=True).astype(np.float64).filled(default)


def _polygon_parts(geometry) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    return [part for item in getattr(geometry, "geoms", []) for part in _polygon_parts(item)]


def _solid_polygon_parts(geometry) -> list[Polygon]:
    """Return polygons with interior rings removed for exclusive reservoir zones."""
    return [
        Polygon(part.exterior).buffer(0)
        for part in _polygon_parts(geometry)
        if part.area > 0
    ]


def _mask_polygons(
    mask: np.ndarray, transform: Any, simplify_m: float, minimum_area_m2: float,
) -> list[Polygon]:
    polygons: list[Polygon] = []
    for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if not value:
            continue
        cleaned = shape(geometry).buffer(0)
        if simplify_m:
            cleaned = cleaned.simplify(simplify_m, preserve_topology=True).buffer(0)
        polygons.extend(part for part in _polygon_parts(cleaned) if part.area >= minimum_area_m2)
    return polygons


def _filter_small_components(mask: np.ndarray, minimum_area_m2: float, pixel_area_m2: float) -> np.ndarray:
    groups, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    areas = np.bincount(groups.ravel()) * pixel_area_m2
    keep = areas >= minimum_area_m2
    keep[0] = False
    return keep[groups]


def _fill_small_mask_holes(
    mask: np.ndarray, blockers: np.ndarray, maximum_area_m2: float, pixel_area_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_area_m2 <= 0 or not mask.any():
        return mask, np.zeros_like(mask, dtype=bool)
    holes = binary_fill_holes(mask) & ~mask
    groups, count = label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return mask, holes
    areas = np.bincount(groups.ravel()) * pixel_area_m2
    blocked = np.zeros(count + 1, dtype=bool)
    blocked[np.unique(groups[blockers & (groups > 0)])] = True
    fill = (groups > 0) & ~blocked[groups] & (areas[groups] <= maximum_area_m2)
    return mask | fill, fill


def _disk(radius_cells: int) -> np.ndarray:
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return (x * x + y * y) <= radius_cells * radius_cells


def _remove_narrow_mask_parts(mask: np.ndarray, minimum_width_m: float, transform: Any) -> np.ndarray:
    if minimum_width_m <= 0:
        return mask
    radius = max(1, int(np.ceil(0.5 * minimum_width_m / min(abs(transform.a), abs(transform.e)))))
    return binary_opening(mask, structure=_disk(radius))


def _row_value(row, names: tuple[str, ...]):
    for name in names:
        value = row.get(name)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


def _waterbody_type(row) -> str | None:
    lake_type = _row_value(row, ("Lake_type", "lake_type", "LAKE_TYPE"))
    if lake_type is not None:
        try:
            return {1: "lake", 2: "reservoir", 3: "regulated_lake"}.get(int(lake_type), str(lake_type))
        except (TypeError, ValueError):
            return str(lake_type)
    return _row_value(row, ("water", "natural", "landuse", "waterway"))


def _major_river_mask(
    river: np.ndarray, width: np.ndarray, upstream_area: np.ndarray, config: MeshFeatureCandidateConfig,
) -> np.ndarray:
    major = river & (
        (np.isfinite(width) & (width >= config.major_river_min_width_m))
        | (np.isfinite(upstream_area) & (upstream_area >= config.major_river_min_upstream_area_km2))
    )
    return major if major.any() else river


def _nearest_river_hand(
    dem: np.ndarray, river: np.ndarray, transform: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not river.any():
        raise ValueError("At least one river cell is required to build feature candidates.")
    distance, nearest = distance_transform_edt(
        ~river,
        sampling=(abs(transform.e), abs(transform.a)),
        return_indices=True,
    )
    base = dem[nearest[0], nearest[1]]
    hand = dem - base
    hand[~np.isfinite(dem) | ~np.isfinite(base)] = np.nan
    return np.maximum(hand, 0.0), distance


def _receiver(direction: np.ndarray, routing_scheme: str) -> np.ndarray:
    return receiver_from_d4_direction(direction) if routing_scheme == "d4" else receiver_from_d8_direction(direction)


def _merged_record_lines(records: list[tuple[int, LineString, float, tuple[float, float]]], simplify_m: float) -> list[LineString]:
    if not records:
        return []
    merged = linemerge(unary_union([record[1] for record in records]))
    result: list[LineString] = []
    for line in _line_parts(merged):
        if simplify_m:
            line = line.simplify(simplify_m, preserve_topology=False)
        if line.length > 0:
            result.append(line)
    return result


def _domain_geometry(path: Path, crs: Any):
    domain = gpd.read_file(path)
    if domain.empty:
        raise ValueError(f"Domain vector is empty: {path}")
    return domain.to_crs(crs).geometry.union_all().buffer(0)


def _read_vector_near_domain(path: Path, domain, crs: Any) -> gpd.GeoDataFrame:
    sample = gpd.read_file(path, rows=1)
    if sample.crs is None:
        raise ValueError(f"Vector layer must define a CRS for bbox loading: {path}")
    bounds = tuple(gpd.GeoSeries([domain], crs=crs).to_crs(sample.crs).total_bounds)
    return gpd.read_file(path, bbox=bounds).to_crs(crs)


def _osm_query(url: str, query: str) -> dict[str, Any]:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "HydroBathyDEM mesh-feature-candidates"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _bbox_query(domain, crs: Any) -> tuple[float, float, float, float]:
    west, south, east, north = gpd.GeoSeries([domain], crs=crs).to_crs(4326).total_bounds
    return south, west, north, east


def _osm_way_line(element: dict[str, Any]) -> LineString | None:
    coords = [(point["lon"], point["lat"]) for point in element.get("geometry", [])]
    return LineString(coords) if len(coords) >= 2 else None


def _osm_water_polygons(elements: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    for element in elements:
        tags = element.get("tags", {})
        if element.get("type") == "way":
            line = _osm_way_line(element)
            if line is None or not line.is_ring:
                continue
            polygons = [Polygon(line.coords)]
        elif element.get("type") == "relation":
            lines = []
            for member in element.get("members", []):
                coords = [(point["lon"], point["lat"]) for point in member.get("geometry", [])]
                if len(coords) >= 2:
                    lines.append(LineString(coords))
            polygons = list(polygonize(lines))
        else:
            continue
        for polygon in polygons:
            if polygon.is_valid and polygon.area > 0:
                rows.append({
                    "source": "osm",
                    "osm_id": int(element.get("id", -1)),
                    "name": tags.get("name"),
                    "water": tags.get("water") or tags.get("natural") or tags.get("landuse") or tags.get("waterway"),
                    "geometry": polygon,
                })
    if not rows:
        return gpd.GeoDataFrame(
            {"source": [], "osm_id": [], "name": [], "water": []},
            geometry=[], crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _fetch_osm_waterbodies(config: MeshFeatureCandidateConfig, domain, crs: Any) -> gpd.GeoDataFrame:
    south, west, north, east = _bbox_query(domain, crs)
    bbox = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:90];
    (
      way["natural"="water"]({bbox});
      relation["natural"="water"]({bbox});
      way["landuse"="reservoir"]({bbox});
      relation["landuse"="reservoir"]({bbox});
      way["waterway"="riverbank"]({bbox});
      relation["waterway"="riverbank"]({bbox});
    );
    out geom tags;
    """
    return _osm_water_polygons(_osm_query(config.overpass_url, query).get("elements", []))


def _fetch_osm_barriers(config: MeshFeatureCandidateConfig, domain, crs: Any) -> gpd.GeoDataFrame:
    south, west, north, east = _bbox_query(domain, crs)
    bbox = f"{south},{west},{north},{east}"
    query = f"""
    [out:json][timeout:90];
    (
      way["waterway"="dam"]({bbox});
      way["man_made"~"^(dyke|embankment|dam)$"]({bbox});
      way["embankment"="yes"]({bbox});
    );
    out geom tags;
    """
    rows: list[dict[str, Any]] = []
    for element in _osm_query(config.overpass_url, query).get("elements", []):
        line = _osm_way_line(element)
        if line is None:
            continue
        tags = element.get("tags", {})
        rows.append({
            "source": "osm",
            "osm_id": int(element.get("id", -1)),
            "breakline_type": "barrier",
            "name": tags.get("name"),
            "osm_kind": tags.get("waterway") or tags.get("man_made") or tags.get("embankment"),
            "waterway": tags.get("waterway"),
            "man_made": tags.get("man_made"),
            "embankment": tags.get("embankment"),
            "highway": tags.get("highway"),
            "railway": tags.get("railway"),
            "geometry": line,
        })
    if not rows:
        return gpd.GeoDataFrame(
            {
                "source": [], "osm_id": [], "breakline_type": [], "name": [], "osm_kind": [],
                "waterway": [], "man_made": [], "embankment": [], "highway": [], "railway": [],
            },
            geometry=[], crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _load_waterbodies(
    config: MeshFeatureCandidateConfig, domain, crs: Any, river_lines,
    warnings: list[str],
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    if config.waterbody_vector is not None:
        frames.append(_read_vector_near_domain(config.waterbody_vector, domain, crs).assign(source=config.waterbody_vector.stem))
    if config.fetch_osm_waterbodies:
        try:
            fetched = _fetch_osm_waterbodies(config, domain, crs)
            if not fetched.empty:
                frames.append(fetched.to_crs(crs))
        except Exception as exc:  # pragma: no cover - network availability is external.
            warnings.append(f"OSM waterbody fetch failed: {exc}")
    if not frames:
        return gpd.GeoDataFrame(
            {"source": [], "waterbody_id": [], "waterbody_type": [], "name": [], "area_m2": []},
            geometry=[], crs=crs,
        )
    waterbodies = gpd.GeoDataFrame(pd_concat(frames), geometry="geometry", crs=crs)
    rows: list[dict[str, Any]] = []
    for _, row in waterbodies.iterrows():
        geometry = row.geometry.intersection(domain).buffer(0)
        if config.simplify_tolerance_m:
            geometry = geometry.simplify(config.simplify_tolerance_m, preserve_topology=True).buffer(0)
        for polygon in _solid_polygon_parts(geometry):
            if polygon.area < config.waterbody_min_area_m2:
                continue
            if not river_lines.is_empty and polygon.distance(river_lines) > config.waterbody_max_distance_to_river_m:
                continue
            rows.append({
                "source": row.get("source", "unknown"),
                "waterbody_id": _row_value(row, ("Hylak_id", "hylak_id", "HYLAK_ID", "osm_id", "id")),
                "waterbody_type": _waterbody_type(row),
                "name": row.get("name"),
                "area_m2": float(polygon.area),
                "geometry": polygon,
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _local_relief(dem: np.ndarray, valid: np.ndarray, window_m: float, transform: Any) -> np.ndarray:
    cell_m = min(abs(transform.a), abs(transform.e))
    size = max(1, int(round(window_m / cell_m)))
    if size % 2 == 0:
        size += 1
    high = maximum_filter(np.where(valid, dem, -np.inf), size=size, mode="nearest")
    low = minimum_filter(np.where(valid, dem, np.inf), size=size, mode="nearest")
    relief = high - low
    relief[~np.isfinite(relief)] = np.inf
    return relief


def _waterbody_mask(waterbodies: gpd.GeoDataFrame, shape_: tuple[int, int], transform: Any) -> np.ndarray:
    if waterbodies.empty:
        return np.zeros(shape_, dtype=bool)
    return geometry_mask(waterbodies.geometry, out_shape=shape_, transform=transform, invert=True)


def _refine_waterbodies_from_dem(
    waterbodies: gpd.GeoDataFrame, dem: np.ndarray, valid: np.ndarray,
    transform: Any, config: MeshFeatureCandidateConfig, warnings: list[str],
) -> gpd.GeoDataFrame:
    if not config.refine_waterbodies_from_dem or waterbodies.empty:
        return waterbodies
    relief = _local_relief(dem, valid, config.waterbody_refine_relief_window_m, transform)
    rows: list[dict[str, Any]] = []
    for _, row in waterbodies.iterrows():
        original = row.geometry
        original_area = float(original.area)
        core = original.buffer(-max(config.simplify_tolerance_m, 0.0))
        if core.is_empty:
            core = original
        core_mask = geometry_mask([core], out_shape=dem.shape, transform=transform, invert=True) & valid
        core_values = dem[core_mask & np.isfinite(dem)]
        if len(core_values) < 3:
            warnings.append(f"Waterbody {row.get('waterbody_id')} kept original geometry: too few DEM cells inside polygon.")
            for part in _solid_polygon_parts(original):
                rows.append({**row.drop(labels="geometry").to_dict(), "original_area_m2": original_area, "dem_refined": False, "geometry": part})
            continue
        water_level = float(np.nanmedian(core_values))
        search = original.buffer(config.waterbody_refine_buffer_m)
        search_mask = geometry_mask([search], out_shape=dem.shape, transform=transform, invert=True) & valid
        candidate = (
            search_mask
            & (np.abs(dem - water_level) <= config.waterbody_refine_elevation_tolerance_m)
            & (relief <= config.waterbody_refine_max_relief_m)
        )
        groups, _ = label(candidate, structure=np.ones((3, 3), dtype=np.uint8))
        connected_labels = np.unique(groups[core_mask & (groups > 0)])
        refined_mask = np.isin(groups, connected_labels) if len(connected_labels) else np.zeros_like(candidate)
        parts = _mask_polygons(refined_mask, transform, config.simplify_tolerance_m, config.waterbody_min_area_m2)
        refined = unary_union([original, *parts]).buffer(0) if parts else original
        refined = unary_union(_solid_polygon_parts(refined)).buffer(0)
        refined_area = float(refined.area) if not refined.is_empty else 0.0
        ratio = refined_area / original_area if original_area else 0.0
        if refined.is_empty or ratio < config.waterbody_refine_min_area_ratio or ratio > config.waterbody_refine_max_area_ratio:
            warnings.append(
                f"Waterbody {row.get('waterbody_id')} kept original geometry: DEM-refined area ratio {ratio:.2f} outside limits."
            )
            for part in _solid_polygon_parts(original):
                rows.append({
                    **row.drop(labels="geometry").to_dict(),
                    "original_area_m2": original_area,
                    "dem_water_level_m": water_level,
                    "dem_refined": False,
                    "dem_refine_area_ratio": ratio,
                    "geometry": part,
                })
            continue
        for part in _solid_polygon_parts(refined):
            rows.append({
                **row.drop(labels="geometry").to_dict(),
                "original_area_m2": original_area,
                "area_m2": float(part.area),
                "dem_water_level_m": water_level,
                "dem_refined": True,
                "dem_refine_area_ratio": ratio,
                "geometry": part,
            })
    if not rows:
        return gpd.GeoDataFrame(
            {"source": [], "waterbody_id": [], "waterbody_type": [], "name": [], "area_m2": []},
            geometry=[], crs=waterbodies.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=waterbodies.crs)


def _subtract_polygons(
    polygons: list[Polygon], blockers, minimum_area_m2: float,
) -> list[Polygon]:
    if blockers is None or blockers.is_empty:
        return polygons
    result: list[Polygon] = []
    for polygon in polygons:
        result.extend(
            part for part in _polygon_parts(polygon.difference(blockers).buffer(0))
            if part.area >= minimum_area_m2
        )
    return result


def _centerline_breaklines(
    lines: list[LineString], waterbodies: gpd.GeoDataFrame, config: MeshFeatureCandidateConfig,
) -> tuple[list[LineString], float]:
    if not config.suppress_centerlines_inside_waterbodies or waterbodies.empty:
        return lines, 0.0
    water = waterbodies.geometry.union_all()
    result: list[LineString] = []
    for line in lines:
        result.extend(
            part for part in _line_parts(line.difference(water))
            if part.length >= config.barrier_min_length_m
        )
    return result, max(sum(line.length for line in lines) - sum(line.length for line in result), 0.0)


def pd_concat(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def build_mesh_feature_candidates(config: MeshFeatureCandidateConfig) -> dict[str, Any]:
    warnings: list[str] = []
    with rasterio.open(config.dem) as dem_source:
        dem = dem_source.read(1, masked=True).astype(np.float64).filled(np.nan)
        crs = dem_source.crs
        transform = dem_source.transform
        domain = _domain_geometry(config.domain_vector, crs)
        valid = np.isfinite(dem) & geometry_mask([domain], out_shape=dem.shape, transform=transform, invert=True)
        river = valid & (_aligned(config.river_mask, dem_source, 0.0) > 0)
        direction = _aligned(config.river_direction, dem_source, 0.0)
        width = _aligned(config.river_width, dem_source, np.nan)
        upstream = _aligned(config.river_upstream_area, dem_source, np.nan)

    pixel_area_m2 = abs(transform.a * transform.e)
    receiver = _receiver(direction, config.routing_scheme)
    major_river = _major_river_mask(river, width, upstream, config)
    all_records = _river_segment_records(river, receiver, transform, width)
    major_records = [record for record in all_records if major_river.ravel()[record[0]]]
    major_lines = _merged_record_lines(major_records, config.simplify_tolerance_m)
    river_lines = unary_union(major_lines) if major_lines else LineString()
    waterbodies = _load_waterbodies(config, domain, crs, river_lines, warnings)
    waterbodies = _refine_waterbodies_from_dem(waterbodies, dem, valid, transform, config, warnings)
    water_mask = _waterbody_mask(waterbodies, dem.shape, transform)
    hand, distance = _nearest_river_hand(dem, major_river, transform)
    raw_floodplain = (
        valid & ~river
        & (distance <= config.floodplain_max_distance_m)
        & np.isfinite(hand)
        & (hand <= config.floodplain_max_hand_m)
    )
    floodplain_waterbody_overlap = raw_floodplain & water_mask
    floodplain = raw_floodplain & ~water_mask
    floodplain = _remove_narrow_mask_parts(floodplain, config.floodplain_min_width_m, transform)
    floodplain = _filter_small_components(floodplain, config.floodplain_min_area_m2, pixel_area_m2)
    floodplain, filled_holes = _fill_small_mask_holes(
        floodplain, water_mask, config.floodplain_max_hole_fill_area_m2, pixel_area_m2,
    )
    floodplain_polygons = _mask_polygons(
        floodplain, transform, config.simplify_tolerance_m, config.floodplain_min_area_m2,
    )
    filled_hole_polygons = _mask_polygons(
        filled_holes, transform, 0.0, 0.0,
    )
    water_geometry = waterbodies.geometry.union_all() if len(waterbodies) else Polygon()
    floodplain_polygons = _subtract_polygons(
        floodplain_polygons, water_geometry, config.floodplain_min_area_m2,
    )
    centerline_lines, suppressed_centerline_length_m = _centerline_breaklines(major_lines, waterbodies, config)
    breakline_rows = [
        {
            "source": "hydrobathydem",
            "breakline_type": "major_river_centerline",
            "length_m": float(line.length),
            "geometry": line,
        }
        for line in centerline_lines
    ]
    for _, row in waterbodies.iterrows():
        boundary = row.geometry.boundary
        for line in _line_parts(boundary):
            if config.simplify_tolerance_m:
                line = line.simplify(config.simplify_tolerance_m, preserve_topology=False)
            if line.length >= config.barrier_min_length_m:
                breakline_rows.append({
                    "source": row.get("source", "waterbody"),
                    "breakline_type": "waterbody_shoreline",
                    "name": row.get("name"),
                    "length_m": float(line.length),
                    "geometry": line,
                })
    if config.fetch_osm_barriers:
        try:
            barriers = _fetch_osm_barriers(config, domain, crs)
            if not barriers.empty:
                floodplain_geometry = unary_union(floodplain_polygons) if floodplain_polygons else Polygon()
                waterbody_edges = unary_union([geometry.boundary for geometry in waterbodies.geometry]) if len(waterbodies) else LineString()
                if config.barrier_filter_to_waterbody_edges and waterbody_edges.is_empty:
                    warnings.append("OSM barriers were skipped because no waterbody shoreline candidates were available.")
                for _, row in barriers.to_crs(crs).iterrows():
                    geometry = row.geometry.intersection(domain)
                    if not floodplain_geometry.is_empty:
                        geometry = geometry.intersection(floodplain_geometry)
                    for line in _line_parts(geometry):
                        if line.length >= config.barrier_min_length_m:
                            if config.barrier_filter_to_waterbody_edges and waterbody_edges.is_empty:
                                continue
                            distance_to_waterbody_m = (
                                float(line.distance(waterbody_edges))
                                if not waterbody_edges.is_empty else None
                            )
                            if (
                                config.barrier_filter_to_waterbody_edges
                                and distance_to_waterbody_m is not None
                                and distance_to_waterbody_m > config.barrier_max_distance_to_waterbody_m
                            ):
                                continue
                            is_dam = row.get("waterway") == "dam" or row.get("man_made") == "dam"
                            is_engineered_embankment = row.get("man_made") in {"dyke", "embankment"}
                            breakline_rows.append({
                                "source": row.get("source", "osm"),
                                "breakline_type": row.get("breakline_type", "barrier"),
                                "name": row.get("name"),
                                "osm_kind": row.get("osm_kind"),
                                "confidence": "high" if is_dam else "medium" if is_engineered_embankment else "low",
                                "distance_to_waterbody_m": distance_to_waterbody_m,
                                "length_m": float(line.length),
                                "geometry": line,
                            })
        except Exception as exc:  # pragma: no cover - network availability is external.
            warnings.append(f"OSM barrier fetch failed: {exc}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = config.out_dir / "mesh_feature_candidates.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    floodplain_gdf = gpd.GeoDataFrame(
        {
            "feature": ["floodplain_refinement"] * len(floodplain_polygons),
            "area_m2": [polygon.area for polygon in floodplain_polygons],
        },
        geometry=floodplain_polygons, crs=crs,
    )
    breaklines_gdf = (
        gpd.GeoDataFrame(breakline_rows, geometry="geometry", crs=crs)
        if breakline_rows else
        gpd.GeoDataFrame({"source": [], "breakline_type": [], "length_m": []}, geometry=[], crs=crs)
    )
    floodplain_gdf.to_file(gpkg, layer="floodplain_refinement", driver="GPKG")
    if filled_hole_polygons:
        gpd.GeoDataFrame(
            {
                "feature": ["filled_floodplain_hole"] * len(filled_hole_polygons),
                "area_m2": [polygon.area for polygon in filled_hole_polygons],
            },
            geometry=filled_hole_polygons, crs=crs,
        ).to_file(gpkg, layer="floodplain_filled_holes", driver="GPKG")
    waterbodies.to_file(gpkg, layer="waterbody_refinement", driver="GPKG")
    breaklines_gdf.to_file(gpkg, layer="breakline_candidates", driver="GPKG")

    mask_path = config.out_dir / "floodplain_refinement_candidate_mask.tif"
    with rasterio.open(config.dem) as source:
        profile = source.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        with rasterio.open(mask_path, "w", **profile) as target:
            target.write(floodplain.astype("uint8"), 1)

    breakline_by_type = {}
    for row in breakline_rows:
        key = str(row.get("breakline_type", "unknown"))
        item = breakline_by_type.setdefault(key, {"count": 0, "length_km": 0.0})
        item["count"] += 1
        item["length_km"] += float(row["length_m"]) / 1000.0
    report = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "method": "QA-only nearest-drainage floodplain refinement plus selective global-source breakline candidates.",
        "major_river_cells": int(major_river.sum()),
        "floodplain_refinement_cells": int(floodplain.sum()),
        "floodplain_refinement_area_km2": float(floodplain.sum() * pixel_area_m2 / 1e6),
        "floodplain_hole_fill_cells": int(filled_holes.sum()),
        "floodplain_hole_fill_area_km2": float(filled_holes.sum() * pixel_area_m2 / 1e6),
        "floodplain_candidate_area_removed_by_waterbodies_km2": float(floodplain_waterbody_overlap.sum() * pixel_area_m2 / 1e6),
        "floodplain_refinement_polygons": len(floodplain_polygons),
        "waterbody_candidates": int(len(waterbodies)),
        "waterbody_area_km2": float(waterbodies.geometry.area.sum() / 1e6) if not waterbodies.empty else 0.0,
        "dem_refined_waterbody_parts": int(waterbodies["dem_refined"].sum()) if "dem_refined" in waterbodies else 0,
        "breakline_candidates": int(len(breakline_rows)),
        "breakline_length_km": float(sum(row["length_m"] for row in breakline_rows) / 1000.0),
        "centerline_length_suppressed_inside_waterbodies_km": float(suppressed_centerline_length_m / 1000.0),
        "breakline_by_type": breakline_by_type,
        "warnings": warnings,
        "outputs": {
            "geopackage": str(gpkg),
            "floodplain_mask": str(mask_path),
        },
        "scope": "Candidate QA layers only; inspect before enforcing in a production mesh.",
    }
    report_path = config.out_dir / "mesh_feature_candidates_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {**report, "report": str(report_path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build QA-only mesh floodplain and breakline candidates.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_mesh_feature_candidates(MeshFeatureCandidateConfig.from_mapping(load_config_file(args.config))), indent=2))


if __name__ == "__main__":
    main()
