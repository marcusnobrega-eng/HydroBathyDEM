from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from dem_processing.hybrid_mesh import (
    ComputationPointSet,
    HybridMeshConfig,
    compile_mesh_from_points,
    generate_computation_points,
)
from dem_processing.mesh_editor import (
    EDITOR_LAYERS,
    apply_mesh_edits,
    breakline_influence_areas,
    enforce_breaklines,
    enforce_refinement_regions,
    load_mesh_editor_session,
    normalise_breaklines,
    normalise_refinement_regions,
    promote_mesh_revision,
    refinement_region_influence_areas,
    save_mesh_editor_session,
)


def _editor_definition(tmp_path: Path) -> tuple[HybridMeshConfig, ComputationPointSet]:
    dem = tmp_path / "dem.tif"
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 2000000.0, 10.0, 10.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.arange(16, dtype=np.float32).reshape(4, 4), 1)
    config = HybridMeshConfig(
        dem=dem,
        out_dir=tmp_path / "compiled",
        background_width_m=10.0,
        minimum_hydraulic_width_m=5.0,
        floodplain_enabled=False,
        river_cell_style="voronoi",
        write_diagnostics=False,
    )
    coordinates = np.asarray(
        [
            [500005.0, 1999995.0],
            [500025.0, 1999995.0],
            [500005.0, 1999975.0],
            [500025.0, 1999975.0],
        ]
    )
    points = ComputationPointSet(
        point_id=np.arange(4),
        coordinates=coordinates,
        origin=np.asarray(["background", "river", "background", "background"]),
        source_id=np.asarray(["background:0", "river_core:0", "background:2", "background:3"]),
        active=np.asarray([True, True, True, False]),
        locked=np.asarray([False, True, False, False]),
        manually_modified=np.asarray([False, False, True, False]),
        crs_wkt=rasterio.crs.CRS.from_epsg(32643).to_wkt(),
    )
    return config, points


def test_mesh_editor_session_round_trip_preserves_definition(tmp_path: Path) -> None:
    config, points = _editor_definition(tmp_path)
    project = tmp_path / "project"

    saved = save_mesh_editor_session(project, config, points)
    loaded = load_mesh_editor_session(project)
    saved_again = save_mesh_editor_session(
        project,
        loaded.config,
        loaded.points,
        domain=loaded.domain,
        breaklines=loaded.breaklines,
        refinement_regions=loaded.refinement_regions,
        automatic_features=loaded.automatic_features,
    )

    assert (project / "mesh" / "mesh_recipe.yaml").is_file()
    geopackage = project / "mesh" / "mesh_editor.gpkg"
    assert set(gpd.list_layers(geopackage)["name"]) == set(EDITOR_LAYERS)
    assert saved.definition_sha256 == loaded.definition_sha256 == saved_again.definition_sha256
    assert loaded.config == config
    for field in (
        "point_id",
        "coordinates",
        "origin",
        "source_id",
        "active",
        "locked",
        "manually_modified",
    ):
        assert np.array_equal(getattr(loaded.points, field), getattr(points, field))
    assert len(loaded.domain) == 1
    assert len(loaded.automatic_features) == 1


def test_default_mesh_domain_excludes_dem_nodata_cells(tmp_path: Path) -> None:
    config, points = _editor_definition(tmp_path)
    with rasterio.open(config.dem, "r+") as dataset:
        elevation = dataset.read(1)
        elevation[0, 0] = dataset.nodata
        dataset.write(elevation, 1)

    session = save_mesh_editor_session(tmp_path / "project", config, points)

    assert session.domain.geometry.area.sum() == pytest.approx(1_500.0)


def test_mesh_editor_session_rejects_modified_recipe(tmp_path: Path) -> None:
    config, points = _editor_definition(tmp_path)
    project = tmp_path / "project"
    save_mesh_editor_session(project, config, points)
    recipe_path = project / "mesh" / "mesh_recipe.yaml"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["config"]["background_width_m"] = 99.0
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")

    try:
        load_mesh_editor_session(project)
    except ValueError as error:
        assert "hash does not match" in str(error)
    else:
        raise AssertionError("A modified mesh recipe was accepted.")


def test_point_edits_add_move_delete_lock_and_preserve_ids(tmp_path: Path) -> None:
    config, original = _editor_definition(tmp_path)
    points = ComputationPointSet(
        point_id=np.arange(5),
        coordinates=np.asarray(
            [
                [500005.0, 1999995.0],
                [500015.0, 1999995.0],
                [500025.0, 1999995.0],
                [500005.0, 1999975.0],
                [500025.0, 1999975.0],
            ]
        ),
        origin=np.full(5, "background"),
        source_id=np.asarray([f"background:{index}" for index in range(5)]),
        active=np.ones(5, dtype=bool),
        locked=np.zeros(5, dtype=bool),
        manually_modified=np.zeros(5, dtype=bool),
        crs_wkt=original.crs_wkt,
    )
    edited = apply_mesh_edits(
        points,
        [
            {"operation": "move", "point_id": 0, "coordinate": [500006.0, 1999994.0]},
            {"operation": "delete", "point_id": 1},
            {"operation": "set_locked", "point_id": 2, "locked": True},
            {"operation": "add", "coordinate": [500015.0, 1999975.0]},
        ],
        config,
    )

    assert edited.point_id.tolist() == [0, 1, 2, 3, 4, 5]
    assert edited.origin.tolist()[:2] == ["manual_modified", "manual_modified"]
    assert not edited.active[1]
    assert edited.locked[2]
    assert edited.origin[5] == "manual"
    assert edited.manually_modified[[0, 1, 2, 5]].all()


def test_point_edits_reject_locked_duplicate_and_outside_points(tmp_path: Path) -> None:
    config, points = _editor_definition(tmp_path)
    with pytest.raises(ValueError, match="Locked computation points"):
        apply_mesh_edits(
            points,
            [{"operation": "move", "point_id": 1, "coordinate": [500020.0, 1999990.0]}],
            config,
        )
    with pytest.raises(ValueError, match="must be unique"):
        apply_mesh_edits(
            points,
            [{"operation": "add", "coordinate": points.coordinates[0].tolist()}],
            config,
        )
    with pytest.raises(ValueError, match="inside the mesh domain"):
        apply_mesh_edits(
            points,
            [{"operation": "add", "coordinate": [0.0, 0.0]}],
            config,
        )


def test_session_hash_ignores_transient_compilation_directory(tmp_path: Path) -> None:
    config, points = _editor_definition(tmp_path)
    first = save_mesh_editor_session(tmp_path / "project", config, points)
    second = save_mesh_editor_session(
        tmp_path / "project",
        replace(config, out_dir=tmp_path / "another-build-directory"),
        points,
    )
    assert first.definition_sha256 == second.definition_sha256


def test_breakline_enforcement_persists_and_aligns_compiled_faces(tmp_path: Path) -> None:
    config, original = _editor_definition(tmp_path)
    coordinates = np.asarray(
        [
            [500000.0 + x, 2000000.0 - y]
            for y in (5.0, 15.0, 25.0, 35.0)
            for x in (5.0, 15.0, 25.0, 35.0)
        ]
    )
    points = replace(
        original,
        point_id=np.arange(len(coordinates)),
        coordinates=coordinates,
        origin=np.full(len(coordinates), "background"),
        source_id=np.asarray([f"background:{index}" for index in range(len(coordinates))]),
        active=np.ones(len(coordinates), dtype=bool),
        locked=np.asarray([True, *([False] * (len(coordinates) - 1))]),
        manually_modified=np.zeros(len(coordinates), dtype=bool),
    )
    line = LineString([(500020.0, 1999962.0), (500020.0, 1999998.0)])
    breaklines = gpd.GeoDataFrame(
        {
            "breakline_id": ["levee"],
            "name": ["Levee"],
            "near_spacing_m": [5.0],
            "far_spacing_m": [10.0],
            "near_repeats": [1],
            "refinement_side": ["both"],
            "protect_adjacent": [True],
            "enabled": [True],
        },
        geometry=[line],
        crs="EPSG:32643",
    )
    normalised = normalise_breaklines(breaklines, config)
    influence = breakline_influence_areas(normalised, config)
    edited = enforce_breaklines(points, normalised, config)
    generated = edited.active & (edited.origin == "breakline")
    assert generated.any()
    assert edited.locked[generated].all()
    assert edited.active[0]
    assert float(influence.iloc[0]["reach_m"]) > 0.0

    project = tmp_path / "project"
    saved = save_mesh_editor_session(project, config, edited, breaklines=normalised)
    loaded = load_mesh_editor_session(project)
    assert saved.definition_sha256 == loaded.definition_sha256
    assert loaded.breaklines.iloc[0]["name"] == "Levee"

    built = compile_mesh_from_points(replace(config, out_dir=tmp_path / "compiled"), edited)
    cells = gpd.read_file(Path(built["geopackage"]), layer="mesh")
    boundaries = unary_union(cells.geometry.boundary)
    samples = [line.interpolate(value, normalized=True) for value in np.linspace(0.1, 0.9, 9)]
    assert max(point.distance(boundaries) for point in samples) < 1.0

    cleared = enforce_breaklines(edited, normalised.iloc[0:0], config)
    assert np.array_equal(cleared.active[: len(points.active)], points.active)
    assert not any(
        str(value).startswith("breakline:")
        for value in cleared.source_id[cleared.active]
    )


def test_bent_breakline_aligns_compiled_faces(tmp_path: Path) -> None:
    config, original = _editor_definition(tmp_path)
    line = LineString(
        [(500006.0, 1999965.0), (500020.0, 1999975.0), (500034.0, 1999995.0)]
    )
    breaklines = gpd.GeoDataFrame(
        {
            "breakline_id": ["bent"],
            "near_spacing_m": [4.0],
            "far_spacing_m": [10.0],
            "near_repeats": [1],
            "refinement_side": ["both"],
            "protect_adjacent": [True],
            "enabled": [True],
        },
        geometry=[line],
        crs="EPSG:32643",
    )
    edited = enforce_breaklines(original, breaklines, config)
    built = compile_mesh_from_points(replace(config, out_dir=tmp_path / "bent"), edited)
    cells = gpd.read_file(Path(built["geopackage"]), layer="mesh")
    boundaries = unary_union(cells.geometry.boundary)
    samples = [line.interpolate(value, normalized=True) for value in np.linspace(0.1, 0.9, 17)]

    assert max(point.distance(boundaries) for point in samples) < 1.0


def test_generated_computation_points_survive_reopen(tmp_path: Path) -> None:
    config, _ = _editor_definition(tmp_path)
    points = generate_computation_points(config)
    project = tmp_path / "generated-project"

    saved = save_mesh_editor_session(project, config, points)
    loaded = load_mesh_editor_session(project)

    assert saved.definition_sha256 == loaded.definition_sha256
    assert np.array_equal(points.point_id, loaded.points.point_id)
    assert np.array_equal(points.coordinates, loaded.points.coordinates)
    assert np.array_equal(points.origin, loaded.points.origin)


def test_refinement_regions_prioritize_spacing_persist_and_compile(tmp_path: Path) -> None:
    config, original = _editor_definition(tmp_path)
    coordinates = np.asarray(
        [
            [500000.0 + x, 2000000.0 - y]
            for y in (5.0, 15.0, 25.0, 35.0)
            for x in (5.0, 15.0, 25.0, 35.0)
        ]
    )
    points = replace(
        original,
        point_id=np.arange(len(coordinates)),
        coordinates=coordinates,
        origin=np.full(len(coordinates), "background"),
        source_id=np.asarray([f"background:{index}" for index in range(len(coordinates))]),
        active=np.ones(len(coordinates), dtype=bool),
        locked=np.zeros(len(coordinates), dtype=bool),
        manually_modified=np.zeros(len(coordinates), dtype=bool),
    )
    fine = Polygon(
        [(500010.0, 1999965.0), (500032.0, 1999965.0),
         (500032.0, 1999988.0), (500010.0, 1999988.0)]
    )
    overlap = Polygon(
        [(500001.0, 1999961.0), (500039.0, 1999961.0),
         (500039.0, 1999979.0), (500001.0, 1999979.0)]
    )
    regions = gpd.GeoDataFrame(
        {
            "region_id": ["fine", "coarse"],
            "name": ["Fine", "Coarse"],
            "mode": ["refine", "coarsen"],
            "spacing_x_m": [5.0, 12.0],
            "spacing_y_m": [5.0, 12.0],
            "perimeter_spacing_m": [5.0, 8.0],
            "transition_spacing_m": [10.0, 10.0],
            "near_repeats": [1, 1],
            "priority": [2, 1],
            "enabled": [True, True],
        },
        geometry=[fine, overlap],
        crs="EPSG:32643",
    )
    normalised = normalise_refinement_regions(regions, config)
    tied = regions.copy()
    tied["priority"] = 1
    assert normalise_refinement_regions(tied, config)["region_id"].tolist() == [
        "fine",
        "coarse",
    ]
    influence = refinement_region_influence_areas(normalised, config)
    edited = enforce_refinement_regions(points, normalised, config)
    active_source = edited.source_id[edited.active].astype(str)
    fine_points = edited.coordinates[
        edited.active & np.char.startswith(edited.source_id.astype(str), "region:fine:interior")
    ]
    coarse_points = edited.coordinates[
        edited.active & np.char.startswith(edited.source_id.astype(str), "region:coarse:interior")
    ]

    assert len(fine_points) >= 8
    assert np.isclose(np.diff(np.unique(fine_points[:, 0])), 5.0).all()
    assert not any(fine.covers(Point(value)) for value in coarse_points)
    influence_by_id = influence.set_index("region_id")
    assert influence_by_id.loc["fine"].geometry.contains(fine)
    assert influence_by_id.loc["coarse"].geometry.contains(overlap)
    assert any(value.startswith("region:fine:boundary") for value in active_source)

    project = tmp_path / "region-project"
    saved = save_mesh_editor_session(
        project, config, edited, refinement_regions=normalised
    )
    loaded = load_mesh_editor_session(project)
    assert saved.definition_sha256 == loaded.definition_sha256
    assert set(loaded.refinement_regions["region_id"]) == {"fine", "coarse"}

    built = compile_mesh_from_points(replace(config, out_dir=tmp_path / "region-compiled"), edited)
    cells = gpd.read_file(Path(built["geopackage"]), layer="mesh")
    assert len(cells) >= 4
    assert cells.geometry.is_valid.all()
    assert (cells.geometry.area > 0.0).all()


def test_mesh_revision_promotion_keeps_only_current_and_previous(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for revision in range(3):
        staged = tmp_path / f"staged-{revision}"
        staged.mkdir()
        (staged / "mesh.nc").write_text(str(revision), encoding="utf-8")
        current = promote_mesh_revision(project, staged, f"hash-{revision}")
        assert (current / "mesh.nc").read_text(encoding="utf-8") == str(revision)

    revisions = project / "derived" / "mesh"
    assert (revisions / "current" / "mesh.nc").read_text(encoding="utf-8") == "2"
    assert (revisions / "previous" / "mesh.nc").read_text(encoding="utf-8") == "1"
    assert json.loads(
        (revisions / "current" / "mesh_revision.json").read_text(encoding="utf-8")
    ) == {"definition_sha256": "hash-2"}
    assert sorted(path.name for path in revisions.iterdir()) == ["current", "previous"]
