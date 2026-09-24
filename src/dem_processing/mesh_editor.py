"""Persistence for editable HydroBathyDEM computation-point sessions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import geopandas as gpd
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from shapely import covers, points as shapely_points
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .hybrid_mesh import ComputationPointSet, HybridMeshConfig, _domain_from_valid_mask


MESH_EDITOR_SCHEMA = "hydrobathydem-mesh-editor-1.0"
EDITOR_LAYERS = (
    "domain",
    "computation_points",
    "breaklines",
    "refinement_regions",
    "automatic_features",
)


@dataclass(frozen=True)
class MeshEditorSession:
    config: HybridMeshConfig
    points: ComputationPointSet
    domain: gpd.GeoDataFrame
    breaklines: gpd.GeoDataFrame
    refinement_regions: gpd.GeoDataFrame
    automatic_features: gpd.GeoDataFrame
    definition_sha256: str


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _config_payload(config: HybridMeshConfig) -> dict[str, Any]:
    return _json_value(asdict(config))


def _frame_payload(frame: gpd.GeoDataFrame) -> dict[str, Any]:
    payload = json.loads(frame.to_json(drop_id=True))
    features = payload.get("features", [])
    features.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    authority = frame.crs.to_authority() if frame.crs is not None else None
    return {
        "crs": ":".join(authority) if authority else frame.crs.to_wkt() if frame.crs else None,
        "features": features,
    }


def _definition_sha256(
    config: dict[str, Any], frames: dict[str, gpd.GeoDataFrame],
) -> str:
    stable_config = dict(config)
    stable_config.pop("out_dir", None)
    payload = {
        "schema_version": MESH_EDITOR_SCHEMA,
        "config": stable_config,
        "layers": {name: _frame_payload(frames[name]) for name in EDITOR_LAYERS},
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_domain(config: HybridMeshConfig) -> gpd.GeoDataFrame:
    with rasterio.open(config.dem) as source:
        if source.crs is None:
            raise ValueError("The mesh DEM must declare a CRS.")
        crs = source.crs
        elevation = source.read(1, masked=True).filled(np.nan)
        raster_domain = _domain_from_valid_mask(np.isfinite(elevation), source.transform)
        geometry = raster_domain
    if config.domain_vector is not None:
        boundary = gpd.read_file(config.domain_vector)
        if boundary.empty:
            raise ValueError(f"Mesh domain vector is empty: {config.domain_vector}")
        geometry = boundary.to_crs(crs).geometry.union_all().intersection(raster_domain)
        if geometry.is_empty:
            raise ValueError("The mesh domain does not intersect valid DEM cells.")
        edge_filter = 0.5 * config.minimum_hydraulic_width_m
        cleaned = geometry.buffer(-edge_filter, join_style="mitre").buffer(
            edge_filter, join_style="mitre",
        )
        if not cleaned.is_empty:
            geometry = cleaned.intersection(raster_domain)
    return gpd.GeoDataFrame({"domain_id": [0]}, geometry=[geometry], crs=crs)


def _empty_layer(crs: Any, column: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {column: np.asarray([], dtype=str)},
        geometry=gpd.GeoSeries([], crs=crs),
    )


def _normalise_layer(
    frame: gpd.GeoDataFrame | None,
    *,
    crs: Any,
    empty_column: str,
) -> gpd.GeoDataFrame:
    if frame is None:
        return _empty_layer(crs, empty_column)
    if frame.crs is None:
        raise ValueError("Mesh-editor layers must declare a CRS.")
    return frame.to_crs(crs).copy()


def _session_frames(
    config: HybridMeshConfig,
    points: ComputationPointSet,
    *,
    domain: gpd.GeoDataFrame | None,
    breaklines: gpd.GeoDataFrame | None,
    refinement_regions: gpd.GeoDataFrame | None,
    automatic_features: gpd.GeoDataFrame | None,
) -> dict[str, gpd.GeoDataFrame]:
    point_crs = rasterio.crs.CRS.from_wkt(points.crs_wkt)
    domain = _normalise_layer(
        domain if domain is not None else _effective_domain(config),
        crs=point_crs,
        empty_column="domain_id",
    )
    point_frame = gpd.GeoDataFrame(
        {
            "point_id": points.point_id,
            "origin": points.origin,
            "source_id": points.source_id,
            "active": points.active,
            "locked": points.locked,
            "manual": points.manually_modified,
        },
        geometry=gpd.points_from_xy(points.coordinates[:, 0], points.coordinates[:, 1]),
        crs=point_crs,
    ).sort_values("point_id", kind="stable").reset_index(drop=True)
    if automatic_features is None:
        automatic_features = point_frame.loc[
            point_frame["origin"] != "background",
            ["point_id", "origin", "source_id", "geometry"],
        ].copy()
    return {
        "domain": domain.reset_index(drop=True),
        "computation_points": point_frame,
        "breaklines": _normalise_layer(
            breaklines, crs=point_crs, empty_column="breakline_id",
        ),
        "refinement_regions": _normalise_layer(
            refinement_regions, crs=point_crs, empty_column="region_id",
        ),
        "automatic_features": _normalise_layer(
            automatic_features, crs=point_crs, empty_column="feature_id",
        ),
    }


def save_mesh_editor_session(
    project_dir: Path,
    config: HybridMeshConfig,
    points: ComputationPointSet,
    *,
    domain: gpd.GeoDataFrame | None = None,
    breaklines: gpd.GeoDataFrame | None = None,
    refinement_regions: gpd.GeoDataFrame | None = None,
    automatic_features: gpd.GeoDataFrame | None = None,
) -> MeshEditorSession:
    """Atomically write ``mesh_recipe.yaml`` and ``mesh_editor.gpkg``."""
    project_dir = Path(project_dir)
    mesh_dir = project_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    frames = _session_frames(
        config,
        points,
        domain=domain,
        breaklines=breaklines,
        refinement_regions=refinement_regions,
        automatic_features=automatic_features,
    )
    config_data = _config_payload(config)
    digest = _definition_sha256(config_data, frames)
    recipe = {
        "schema_version": MESH_EDITOR_SCHEMA,
        "definition_sha256": digest,
        "config": config_data,
        "geopackage": "mesh_editor.gpkg",
        "counts": {name: len(frame) for name, frame in frames.items()},
    }
    recipe_path = mesh_dir / "mesh_recipe.yaml"
    geopackage_path = mesh_dir / "mesh_editor.gpkg"
    with tempfile.TemporaryDirectory(dir=mesh_dir, prefix=".mesh-editor-") as temporary:
        temporary = Path(temporary)
        staged_geopackage = temporary / geopackage_path.name
        for index, name in enumerate(EDITOR_LAYERS):
            frames[name].to_file(
                staged_geopackage,
                layer=name,
                driver="GPKG",
                mode="w" if index == 0 else "a",
            )
        staged_recipe = temporary / recipe_path.name
        staged_recipe.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
        for sidecar in (
            Path(f"{geopackage_path}-wal"),
            Path(f"{geopackage_path}-shm"),
        ):
            sidecar.unlink(missing_ok=True)
        os.replace(staged_geopackage, geopackage_path)
        os.replace(staged_recipe, recipe_path)
    return MeshEditorSession(
        config=config,
        points=points,
        domain=frames["domain"],
        breaklines=frames["breaklines"],
        refinement_regions=frames["refinement_regions"],
        automatic_features=frames["automatic_features"],
        definition_sha256=digest,
    )


def load_mesh_editor_session(project_dir: Path) -> MeshEditorSession:
    """Load and verify a persisted mesh-editor definition."""
    mesh_dir = Path(project_dir) / "mesh"
    recipe_path = mesh_dir / "mesh_recipe.yaml"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("schema_version") != MESH_EDITOR_SCHEMA:
        raise ValueError(
            f"Unsupported mesh-editor schema {recipe.get('schema_version')!r}."
        )
    geopackage_path = mesh_dir / recipe.get("geopackage", "mesh_editor.gpkg")
    available = set(gpd.list_layers(geopackage_path)["name"])
    missing = set(EDITOR_LAYERS) - available
    if missing:
        raise ValueError(f"Mesh-editor GeoPackage is missing layers: {sorted(missing)}")
    frames = {name: gpd.read_file(geopackage_path, layer=name) for name in EDITOR_LAYERS}
    point_frame = frames["computation_points"].sort_values(
        "point_id", kind="stable",
    ).reset_index(drop=True)
    if point_frame.crs is None:
        raise ValueError("Computation points must declare a CRS.")
    if not point_frame.geometry.geom_type.eq("Point").all():
        raise ValueError("The computation_points layer may contain only Point geometry.")
    points = ComputationPointSet(
        point_id=point_frame["point_id"].to_numpy(),
        coordinates=np.column_stack((point_frame.geometry.x, point_frame.geometry.y)),
        origin=point_frame["origin"].to_numpy(),
        source_id=point_frame["source_id"].to_numpy(),
        active=point_frame["active"].to_numpy(dtype=bool),
        locked=point_frame["locked"].to_numpy(dtype=bool),
        manually_modified=point_frame["manual"].to_numpy(dtype=bool),
        crs_wkt=point_frame.crs.to_wkt(),
    )
    config_data = recipe.get("config")
    if not isinstance(config_data, dict):
        raise ValueError("mesh_recipe.yaml does not contain a valid config object.")
    digest = _definition_sha256(config_data, frames)
    if digest != recipe.get("definition_sha256"):
        raise ValueError("Mesh-editor definition hash does not match its saved files.")
    return MeshEditorSession(
        config=HybridMeshConfig.from_mapping(config_data),
        points=points,
        domain=frames["domain"],
        breaklines=frames["breaklines"],
        refinement_regions=frames["refinement_regions"],
        automatic_features=frames["automatic_features"],
        definition_sha256=digest,
    )


def apply_mesh_edits(
    points: ComputationPointSet,
    edits: list[dict[str, Any]],
    config: HybridMeshConfig,
) -> ComputationPointSet:
    """Apply validated point edits while preserving stable point identifiers."""
    point_id = points.point_id.copy()
    coordinates = points.coordinates.copy()
    origin = points.origin.astype(object).copy()
    source_id = points.source_id.astype(object).copy()
    active = points.active.copy()
    locked = points.locked.copy()
    manual = points.manually_modified.copy()

    def indexes(values: Any) -> np.ndarray:
        requested = np.atleast_1d(values).astype(np.int64)
        lookup = {int(value): index for index, value in enumerate(point_id)}
        missing = [int(value) for value in requested if int(value) not in lookup]
        if missing:
            raise ValueError(f"Unknown computation-point IDs: {missing}")
        return np.asarray([lookup[int(value)] for value in requested], dtype=np.int64)

    for edit in edits:
        operation = str(edit.get("operation", "")).lower()
        if operation == "add":
            coordinate = np.asarray(edit.get("coordinate"), dtype=np.float64)
            if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
                raise ValueError("An added computation point requires a finite x, y coordinate.")
            new_id = int(edit.get("point_id", int(point_id.max()) + 1 if len(point_id) else 0))
            if new_id in point_id:
                raise ValueError(f"Computation-point ID {new_id} already exists.")
            point_id = np.append(point_id, new_id)
            coordinates = np.vstack((coordinates, coordinate))
            origin = np.append(origin, "manual")
            source_id = np.append(source_id, f"manual:{new_id}")
            active = np.append(active, True)
            locked = np.append(locked, False)
            manual = np.append(manual, True)
            continue
        selected = indexes(edit.get("point_ids", edit.get("point_id", [])))
        if not len(selected):
            raise ValueError(f"Mesh edit {operation!r} requires at least one point ID.")
        if operation in {"move", "delete"} and np.any(locked[selected]):
            blocked = point_id[selected[locked[selected]]].tolist()
            raise ValueError(f"Locked computation points cannot be {operation}d: {blocked}")
        if operation == "move":
            if len(selected) != 1:
                raise ValueError("Move accepts exactly one computation point.")
            coordinate = np.asarray(edit.get("coordinate"), dtype=np.float64)
            if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
                raise ValueError("A moved computation point requires a finite x, y coordinate.")
            coordinates[selected[0]] = coordinate
            origin[selected] = "manual_modified"
            source_id[selected] = f"manual_move:{int(point_id[selected[0]])}"
            manual[selected] = True
        elif operation == "delete":
            active[selected] = False
            origin[selected] = "manual_modified"
            manual[selected] = True
        elif operation == "set_locked":
            locked[selected] = bool(edit.get("locked", True))
            manual[selected] = True
        else:
            raise ValueError(f"Unsupported computation-point edit: {operation!r}")

    active_coordinates = coordinates[active]
    if len(active_coordinates) < 4:
        raise ValueError("At least four active computation points are required.")
    if len(np.unique(active_coordinates, axis=0)) != len(active_coordinates):
        raise ValueError("Active computation-point coordinates must be unique.")
    domain = _effective_domain(config).geometry.union_all()
    if not np.all(covers(domain, shapely_points(active_coordinates))):
        raise ValueError("Active computation points must remain inside the mesh domain.")
    return ComputationPointSet(
        point_id=point_id,
        coordinates=coordinates,
        origin=origin,
        source_id=source_id,
        active=active,
        locked=locked,
        manually_modified=manual,
        crs_wkt=points.crs_wkt,
    )


def normalise_breaklines(
    breaklines: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> gpd.GeoDataFrame:
    """Validate breakline geometry and fill the public editor defaults."""
    with rasterio.open(config.dem) as source:
        point_crs = source.crs
    if point_crs is None:
        raise ValueError("The mesh DEM must declare a CRS.")
    if breaklines.crs is None:
        raise ValueError("Breaklines must declare a CRS.")
    frame = breaklines.to_crs(point_crs).copy().reset_index(drop=True)
    defaults: dict[str, Any] = {
        "name": "Breakline",
        "near_spacing_m": float(config.minimum_hydraulic_width_m),
        "far_spacing_m": float(config.background_width_m),
        "near_repeats": 2,
        "refinement_side": "both",
        "protect_adjacent": True,
        "enabled": True,
    }
    if "breakline_id" not in frame:
        frame["breakline_id"] = np.arange(len(frame), dtype=np.int64)
    for name, value in defaults.items():
        if name not in frame:
            frame[name] = value
        else:
            frame[name] = frame[name].fillna(value)
    frame["name"] = frame["name"].astype(str)
    frame["near_spacing_m"] = frame["near_spacing_m"].astype(float)
    frame["far_spacing_m"] = frame["far_spacing_m"].astype(float)
    frame["near_repeats"] = frame["near_repeats"].astype(int)
    frame["refinement_side"] = frame["refinement_side"].astype(str).str.lower()
    frame["protect_adjacent"] = frame["protect_adjacent"].astype(bool)
    frame["enabled"] = frame["enabled"].astype(bool)
    if frame["breakline_id"].duplicated().any():
        raise ValueError("Breakline IDs must be unique.")
    if not frame.geometry.geom_type.eq("LineString").all():
        raise ValueError("Breaklines may contain only LineString geometry.")
    if frame.geometry.is_empty.any() or (frame.geometry.length <= 0.0).any():
        raise ValueError("Breaklines must contain at least two distinct vertices.")
    if (frame["near_spacing_m"] <= 0.0).any() or (frame["far_spacing_m"] <= 0.0).any():
        raise ValueError("Breakline spacing must be positive.")
    if (frame["far_spacing_m"] < frame["near_spacing_m"]).any():
        raise ValueError("Far spacing cannot be smaller than near spacing.")
    if (frame["near_repeats"] < 1).any():
        raise ValueError("Breakline near repeats must be at least one.")
    allowed_sides = {"both", "left", "right"}
    invalid_sides = set(frame["refinement_side"]) - allowed_sides
    if invalid_sides:
        raise ValueError(f"Unsupported breakline refinement sides: {sorted(invalid_sides)}")
    domain = _effective_domain(config).geometry.union_all()
    if any(not domain.covers(geometry) for geometry in frame.geometry):
        raise ValueError("Breaklines must remain inside the mesh domain.")
    return frame


def breakline_influence_areas(
    breaklines: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> gpd.GeoDataFrame:
    """Return the graded bands affected when the breaklines are enforced."""
    from .hecras_mesh import Breakline, breakline_points

    frame = normalise_breaklines(breaklines, config)
    domain = _effective_domain(config).geometry.union_all()
    geometries = []
    reaches = []
    for row in frame.itertuples():
        grade_sign = {"both": 0.0, "left": 1.0, "right": -1.0}[row.refinement_side]
        _, reach = breakline_points(
            Breakline(
                geometry=row.geometry,
                near_spacing_m=row.near_spacing_m,
                near_repeats=row.near_repeats,
                far_spacing_m=row.far_spacing_m,
                grade_sign=grade_sign,
                name=row.name,
            ),
            domain,
            config.background_width_m,
        )
        geometries.append(row.geometry.buffer(reach).intersection(domain))
        reaches.append(float(reach))
    return gpd.GeoDataFrame(
        {"breakline_id": frame["breakline_id"], "reach_m": reaches},
        geometry=geometries,
        crs=frame.crs,
    )


def enforce_breaklines(
    points: ComputationPointSet,
    breaklines: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> ComputationPointSet:
    """Replace ordinary sites in graded bands with HEC-RAS-style point rows."""
    from .hecras_mesh import Breakline, breakline_points

    frame = normalise_breaklines(breaklines, config)
    domain = _effective_domain(config).geometry.union_all()
    point_crs = rasterio.crs.CRS.from_wkt(points.crs_wkt)
    if point_crs != frame.crs:
        frame = frame.to_crs(point_crs)
    point_id = points.point_id.copy()
    coordinates = points.coordinates.copy()
    origin = points.origin.astype(object).copy()
    source_id = points.source_id.astype(object).copy()
    active = points.active.copy()
    locked = points.locked.copy()
    manual = points.manually_modified.copy()
    user_breakline_points = np.asarray(
        [str(value).startswith("breakline:") for value in source_id], dtype=bool
    )
    active[user_breakline_points] = False
    masked = np.asarray(
        [str(value).startswith("breakline_mask:") for value in source_id], dtype=bool
    )
    source_id[masked] = np.asarray(
        [str(value).removeprefix("breakline_mask:") for value in source_id[masked]],
        dtype=object,
    )
    active[masked] = np.asarray(
        [not str(value).startswith("region_mask:") for value in source_id[masked]],
        dtype=bool,
    )

    for row in frame.loc[frame["enabled"]].itertuples():
        key = f"breakline:{row.breakline_id}:"
        previous = np.asarray([str(value).startswith(key) for value in source_id])
        active[previous] = False
        grade_sign = {"both": 0.0, "left": 1.0, "right": -1.0}[row.refinement_side]
        groups, reach = breakline_points(
            Breakline(
                geometry=row.geometry,
                near_spacing_m=row.near_spacing_m,
                near_repeats=row.near_repeats,
                far_spacing_m=row.far_spacing_m,
                grade_sign=grade_sign,
                name=row.name,
            ),
            domain,
            config.background_width_m,
        )
        band = row.geometry.buffer(reach)
        ordinary = active & ~locked & (origin != "breakline")
        ordinary_indexes = np.flatnonzero(ordinary)
        if len(ordinary_indexes):
            inside = covers(band, shapely_points(coordinates[ordinary_indexes]))
            selected = ordinary_indexes[np.asarray(inside, dtype=bool)]
            source_id[selected] = np.asarray(
                [f"breakline_mask:{value}" for value in source_id[selected]], dtype=object
            )
            active[selected] = False
        generated = (
            np.vstack([values for values, _spacing in groups if len(values)])
            if groups
            else np.empty((0, 2), dtype=np.float64)
        )
        if not len(generated):
            continue
        generated = np.unique(generated, axis=0)
        retained = coordinates[active]
        if len(retained):
            distance, _ = cKDTree(retained).query(generated)
            generated = generated[distance >= 0.25 * row.near_spacing_m]
        if not len(generated):
            continue
        first_id = int(point_id.max()) + 1 if len(point_id) else 0
        new_ids = np.arange(first_id, first_id + len(generated), dtype=np.int64)
        point_id = np.concatenate((point_id, new_ids))
        coordinates = np.vstack((coordinates, generated))
        origin = np.concatenate((origin, np.full(len(generated), "breakline", dtype=object)))
        source_id = np.concatenate(
            (source_id, np.asarray([f"{key}{index}" for index in range(len(generated))]))
        )
        active = np.concatenate((active, np.ones(len(generated), dtype=bool)))
        locked = np.concatenate(
            (locked, np.full(len(generated), bool(row.protect_adjacent), dtype=bool))
        )
        manual = np.concatenate((manual, np.ones(len(generated), dtype=bool)))

    return ComputationPointSet(
        point_id=point_id,
        coordinates=coordinates,
        origin=origin,
        source_id=source_id,
        active=active,
        locked=locked,
        manually_modified=manual,
        crs_wkt=points.crs_wkt,
    )


def normalise_refinement_regions(
    regions: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> gpd.GeoDataFrame:
    """Validate editable refinement/coarsening regions and fill defaults."""
    with rasterio.open(config.dem) as source:
        point_crs = source.crs
    if point_crs is None:
        raise ValueError("The mesh DEM must declare a CRS.")
    if regions.crs is None:
        raise ValueError("Refinement regions must declare a CRS.")
    frame = regions.to_crs(point_crs).copy().reset_index(drop=True)
    default_spacing = float(config.minimum_hydraulic_width_m)
    defaults: dict[str, Any] = {
        "name": "Refinement region",
        "mode": "refine",
        "spacing_x_m": default_spacing,
        "spacing_y_m": default_spacing,
        "perimeter_spacing_m": default_spacing,
        "transition_spacing_m": float(config.background_width_m),
        "near_repeats": 1,
        "priority": np.arange(1, len(frame) + 1, dtype=np.int64),
        "enabled": True,
    }
    if "region_id" not in frame:
        frame["region_id"] = [f"region_{index + 1}" for index in range(len(frame))]
    for name, value in defaults.items():
        if name not in frame:
            frame[name] = value
        elif name == "priority":
            missing = frame[name].isna()
            frame.loc[missing, name] = np.arange(1, len(frame) + 1)[missing]
        else:
            frame[name] = frame[name].fillna(value)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["name"] = frame["name"].astype(str)
    frame["mode"] = frame["mode"].astype(str).str.lower()
    for name in (
        "spacing_x_m",
        "spacing_y_m",
        "perimeter_spacing_m",
        "transition_spacing_m",
    ):
        frame[name] = frame[name].astype(float)
    frame["near_repeats"] = frame["near_repeats"].astype(int)
    frame["priority"] = frame["priority"].astype(int)
    frame["enabled"] = frame["enabled"].astype(bool)
    if frame["region_id"].duplicated().any():
        raise ValueError("Refinement-region IDs must be unique.")
    if not frame.geometry.geom_type.eq("Polygon").all():
        raise ValueError("Refinement regions may contain only Polygon geometry.")
    if frame.geometry.is_empty.any() or (~frame.geometry.is_valid).any():
        raise ValueError("Refinement-region polygons must be non-empty and valid.")
    spacing_columns = [
        "spacing_x_m",
        "spacing_y_m",
        "perimeter_spacing_m",
        "transition_spacing_m",
    ]
    if (frame[spacing_columns] <= 0.0).any().any():
        raise ValueError("Refinement-region spacing must be positive.")
    if (frame["transition_spacing_m"] < frame["perimeter_spacing_m"]).any():
        raise ValueError("External transition spacing cannot be below perimeter spacing.")
    if (frame["near_repeats"] < 1).any():
        raise ValueError("Refinement-region near repeats must be at least one.")
    invalid_modes = set(frame["mode"]) - {"refine", "coarsen"}
    if invalid_modes:
        raise ValueError(f"Unsupported refinement-region modes: {sorted(invalid_modes)}")
    background = float(config.background_width_m)
    effective = frame[["spacing_x_m", "spacing_y_m"]].min(axis=1)
    if ((frame["mode"] == "refine") & (effective > background)).any():
        raise ValueError("Refinement spacing cannot exceed the background spacing.")
    if ((frame["mode"] == "coarsen") & (effective < background)).any():
        raise ValueError("Coarsening spacing cannot be below the background spacing.")
    domain = _effective_domain(config).geometry.union_all()
    if any(not domain.covers(geometry) for geometry in frame.geometry):
        raise ValueError("Refinement regions must remain inside the mesh domain.")
    return frame.sort_values("priority", kind="stable").reset_index(drop=True)


def _region_spec(row: Any) -> Any:
    from .hecras_mesh import RefinementRegion

    return RefinementRegion(
        geometry=row.geometry,
        cell_spacing_x_m=row.spacing_x_m,
        cell_spacing_y_m=row.spacing_y_m,
        perimeter_spacing_m=row.perimeter_spacing_m,
        near_repeats=row.near_repeats,
        far_spacing_m=row.transition_spacing_m,
        boundary_is_breakline=True,
        boundary_near_repeats=row.near_repeats,
        name=row.name,
    )


def refinement_region_influence_areas(
    regions: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> gpd.GeoDataFrame:
    """Return each region plus its externally graded boundary band."""
    from .hecras_mesh import breakline_points, region_boundary_breaklines

    frame = normalise_refinement_regions(regions, config)
    domain = _effective_domain(config).geometry.union_all()
    geometries: list[Any] = []
    reaches: list[float] = []
    for row in frame.itertuples():
        reach = 0.0
        for boundary in region_boundary_breaklines(
            _region_spec(row), config.background_width_m
        ):
            _groups, boundary_reach = breakline_points(
                boundary, domain, config.background_width_m
            )
            reach = max(reach, float(boundary_reach))
        geometries.append(row.geometry.buffer(reach).intersection(domain))
        reaches.append(reach)
    return gpd.GeoDataFrame(
        {"region_id": frame["region_id"], "reach_m": reaches},
        geometry=geometries,
        crs=frame.crs,
    )


def enforce_refinement_regions(
    points: ComputationPointSet,
    regions: gpd.GeoDataFrame,
    config: HybridMeshConfig,
) -> ComputationPointSet:
    """Replace ordinary sites with prioritized region lattices and boundary rows."""
    from .hecras_mesh import breakline_points, region_boundary_breaklines, region_points

    frame = normalise_refinement_regions(regions, config)
    domain = _effective_domain(config).geometry.union_all()
    point_crs = rasterio.crs.CRS.from_wkt(points.crs_wkt)
    if point_crs != frame.crs:
        frame = frame.to_crs(point_crs)
    point_id = points.point_id.copy()
    coordinates = points.coordinates.copy()
    origin = points.origin.astype(object).copy()
    source_id = points.source_id.astype(object).copy()
    active = points.active.copy()
    locked = points.locked.copy()
    manual = points.manually_modified.copy()

    generated = np.asarray(
        [str(value).startswith("region:") for value in source_id], dtype=bool
    )
    active[generated] = False
    masked = np.asarray(
        [str(value).startswith("region_mask:") for value in source_id], dtype=bool
    )
    source_id[masked] = np.asarray(
        [str(value).removeprefix("region_mask:") for value in source_id[masked]],
        dtype=object,
    )
    active[masked] = np.asarray(
        [not str(value).startswith("breakline_mask:") for value in source_id[masked]],
        dtype=bool,
    )

    enabled = frame.loc[frame["enabled"]].copy()
    enabled["effective_spacing_m"] = enabled[["spacing_x_m", "spacing_y_m"]].min(axis=1)
    enabled = enabled.sort_values(
        ["effective_spacing_m", "priority"], kind="stable"
    ).reset_index(drop=True)
    if len(enabled):
        areas = refinement_region_influence_areas(enabled, config).to_crs(point_crs)
        affected = unary_union(areas.geometry)
        ordinary = active & ~locked & ~np.isin(origin, ["breakline", "region"])
        indexes = np.flatnonzero(ordinary)
        if len(indexes):
            inside = np.asarray(
                covers(affected, shapely_points(coordinates[indexes])), dtype=bool
            )
            selected = indexes[inside]
            source_id[selected] = np.asarray(
                [f"region_mask:{value}" for value in source_id[selected]], dtype=object
            )
            active[selected] = False

    claimed: list[Polygon] = []

    def append_group(values: np.ndarray, key: str, spacing: float, protect: bool) -> None:
        nonlocal point_id, coordinates, origin, source_id, active, locked, manual
        if not len(values):
            return
        values = np.unique(np.asarray(values, dtype=np.float64), axis=0)
        retained = coordinates[active]
        if len(retained):
            distance, _ = cKDTree(retained).query(values)
            values = values[distance >= 0.25 * spacing]
        if not len(values):
            return
        first_id = int(point_id.max()) + 1 if len(point_id) else 0
        new_ids = np.arange(first_id, first_id + len(values), dtype=np.int64)
        point_id = np.concatenate((point_id, new_ids))
        coordinates = np.vstack((coordinates, values))
        origin = np.concatenate((origin, np.full(len(values), "region", dtype=object)))
        source_id = np.concatenate(
            (source_id, np.asarray([f"{key}{index}" for index in range(len(values))]))
        )
        active = np.concatenate((active, np.ones(len(values), dtype=bool)))
        locked = np.concatenate((locked, np.full(len(values), protect, dtype=bool)))
        manual = np.concatenate((manual, np.ones(len(values), dtype=bool)))

    # Hard boundaries are placed first, then interior lattices. This mirrors the
    # production HEC-RAS-style generator and keeps region outlines on cell faces.
    for row in enabled.itertuples():
        spec = _region_spec(row)
        for boundary_index, boundary in enumerate(
            region_boundary_breaklines(spec, config.background_width_m)
        ):
            groups, _reach = breakline_points(
                boundary, domain, config.background_width_m
            )
            for group_index, (values, spacing) in enumerate(groups):
                append_group(
                    values,
                    f"region:{row.region_id}:boundary:{boundary_index}:{group_index}:",
                    spacing,
                    True,
                )

    for row in enabled.itertuples():
        spec = _region_spec(row)
        for group_index, (values, spacing) in enumerate(region_points(spec, domain)):
            if claimed and len(values):
                blocked = unary_union(claimed)
                values = values[
                    ~np.asarray(covers(blocked, shapely_points(values)), dtype=bool)
                ]
            append_group(
                values,
                f"region:{row.region_id}:interior:{group_index}:",
                spacing,
                False,
            )
        claimed.append(row.geometry)

    if active.sum() < 4:
        raise ValueError("Refinement regions left fewer than four active computation points.")
    return ComputationPointSet(
        point_id=point_id,
        coordinates=coordinates,
        origin=origin,
        source_id=source_id,
        active=active,
        locked=locked,
        manually_modified=manual,
        crs_wkt=points.crs_wkt,
    )


def promote_mesh_revision(
    project_dir: Path, staged_dir: Path, definition_sha256: str,
) -> Path:
    """Promote a staged product while retaining only current and previous."""
    staged_dir = Path(staged_dir)
    if not staged_dir.is_dir() or not any(staged_dir.iterdir()):
        raise ValueError("A mesh revision must be a non-empty directory.")
    revisions = Path(project_dir) / "derived" / "mesh"
    revisions.mkdir(parents=True, exist_ok=True)
    current = revisions / "current"
    previous = revisions / "previous"
    incoming = revisions / f".incoming-{uuid4().hex}"
    shutil.copytree(staged_dir, incoming)
    (incoming / "mesh_revision.json").write_text(
        json.dumps({"definition_sha256": definition_sha256}, indent=2),
        encoding="utf-8",
    )
    previous_backup = revisions / f".previous-{uuid4().hex}"
    try:
        if previous.exists():
            os.replace(previous, previous_backup)
        if current.exists():
            os.replace(current, previous)
        os.replace(incoming, current)
    except Exception:
        if not current.exists() and previous.exists():
            os.replace(previous, current)
        if previous_backup.exists():
            os.replace(previous_backup, previous)
        raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    shutil.rmtree(previous_backup, ignore_errors=True)
    return current
