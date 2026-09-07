from pathlib import Path

import numpy as np
from affine import Affine
from pytest import approx
from shapely import unary_union
from shapely.geometry import LineString, Point, Polygon, box

from dem_processing.config import load_config_file
from dem_processing.hybrid_mesh import (
    HybridMeshConfig,
    absorb_degenerate_cells,
    build_river_strip_cells,
    build_river_strip_mesh,
    collapse_short_faces,
    conform_and_repair_cells,
    exclusive_floodplain_geometry,
    fill_partition_voids,
    hole_free_parts,
    insert_hanging_nodes,
    internal_unpaired_edges,
    split_cells_without_interior_centroid,
    split_self_touching_cells,
    substitute_river_strip,
    _fold_limited_half_width,
    _river_reach_paths,
    _river_reach_records,
    _station_normals,
    _agglomerate_short_face_cells,
    _bad_face_seed_removals,
    connected_hand_with_reach,
    _clip_cells_to_feature_polygon,
    _edge_quality,
    _polygons_from_seeds,
    _physical_river_geometry,
    _complete_floodplain_connector_axis,
    _flow_aligned_ribbon_seeds,
    _feature_classes,
    _fraction_refined_feature_classes,
    graded_row_plan,
    _geometry_area_fractions,
    _needs_d4_routing,
    _needs_reach_labels,
    _repair_short_cells,
    _split_by_feature_boundaries,
    _river_bank_support_seeds,
    _river_core_seeds,
    _river_seeds,
    _smooth_width_values,
    _floodplain_connector_ribbon_seeds,
    _minimum_separated,
    _undersized_cell_repair_pairs,
    _smoothed_floodplain_connector_tangents,
    _topology,
    _write_ugrid,
    receiver_from_d4_direction,
    receiver_from_d8_direction,
)
from dem_processing.mesh_product import _selected_mesh


def test_floodplain_defaults_to_isotropic_refinement() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "dem": "dem.tif",
        "out_dir": "mesh",
        "floodplain_target_width_m": 60,
    })
    assert config.floodplain_target_width_m == 60
    assert config.floodplain_align_to_flow is False


def test_floodplain_refinement_can_be_disabled() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "dem": "dem.tif",
        "out_dir": "mesh",
        "floodplain": {"enabled": False},
    })
    assert config.floodplain_enabled is False


def test_waterbody_config_is_grouped_mesh_input() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "inputs": {"dem": "dem.tif", "out_dir": "mesh", "waterbody_vector": "water.gpkg"},
        "resolution": {"waterbody_target_width_m": 200},
    })
    assert config.waterbody_vector == Path("water.gpkg")
    assert config.waterbody_target_width_m == 200


def test_bad_face_repair_prefers_one_shared_unprotected_seed() -> None:
    pairs = np.array([[0, 1], [0, 2], [0, 3]])
    target = np.array([60.0, 45.0, 45.0, 45.0])
    assert _bad_face_seed_removals(pairs, np.zeros(4, dtype=bool), target).tolist() == [0]
    assert set(_bad_face_seed_removals(pairs, np.array([True, False, False, False]), target)) == {1, 2, 3}


def test_bad_face_repair_thins_a_protected_protected_conflict() -> None:
    removed = _bad_face_seed_removals(
        np.array([[0, 1]]), np.array([True, True]), np.array([30.0, 30.0]),
    )
    assert len(removed) == 1


def test_grouped_mesh_inputs_are_case_parameters() -> None:
    mode, values = _selected_mesh(
        load_config_file(Path("examples/mesh_voronoi_config_template.json"))
    )
    assert mode == "voronoi_fv"
    config = HybridMeshConfig.from_mapping(values)
    assert config.river_source == "hydrobathydem_d4"
    assert config.unresolved_policy == "none"
    assert config.background_width_m == 120.0
    assert config.urban_width_m == 45.0
    assert config.minimum_hydraulic_width_m == 30.0


def test_reach_specific_hand_keeps_the_first_downstream_mapped_river() -> None:
    dem = np.array([[5.0, 4.0, 3.0, 2.0]], dtype=float)
    receiver = np.array([[1, 2, 3, -1]], dtype=np.int64)
    river = np.array([[False, False, True, True]])
    hand, reach = connected_hand_with_reach(dem, receiver, river)
    assert np.allclose(hand, [[2.0, 1.0, 0.0, 0.0]])
    assert reach.tolist() == [[2, 2, 2, 3]]


def test_hydrobathy_d4_direction_codes_become_receivers() -> None:
    direction = np.array([[2, 3], [1, 4]], dtype=float)
    assert receiver_from_d4_direction(direction).tolist() == [[1, 3], [0, 2]]


def test_hydrobathy_d8_direction_codes_become_receivers() -> None:
    direction = np.array([[4, 5, 6], [3, 0, 7], [2, 1, 8]], dtype=float)
    receiver = receiver_from_d8_direction(direction)
    assert np.all(receiver[direction > 0] == 4)
    assert receiver[1, 1] == -1


def test_hydrobathy_d8_is_an_explicit_river_source() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "dem": "dem.tif",
        "out_dir": "mesh",
        "rivers": {
            "source": "hydrobathydem_d8",
            "centerline_smoothing_iterations": 2,
            "width_smoothing_window_cells": 5,
            "unresolved_policy": "none",
        },
    })
    assert config.river_source == "hydrobathydem_d8"
    assert config.river_centerline_smoothing_iterations == 2
    assert config.river_width_smoothing_window_cells == 5


def test_supplied_d8_without_floodplain_skips_d4_and_reach_labels() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "dem": "dem.tif",
        "out_dir": "mesh",
        "rivers": {"source": "hydrobathydem_d8", "unresolved_policy": "none"},
        "floodplain": {"enabled": False},
    })
    assert _needs_d4_routing(config, has_supplied_direction=True) is False
    assert _needs_reach_labels(config) is False


def test_mesh_diagnostics_can_be_disabled_from_quality_group() -> None:
    config = HybridMeshConfig.from_mapping({
        "mesh_mode": "voronoi_fv",
        "dem": "dem.tif",
        "out_dir": "mesh",
        "quality": {"write_diagnostics": False},
    })
    assert config.write_diagnostics is False


def test_undersized_cell_repairs_to_longest_same_class_neighbour() -> None:
    polygons = [box(0, 0, 1, 1), box(1, 0, 3, 1), box(0, 1, 1, 3)]
    pairs = _undersized_cell_repair_pairs(
        polygons, np.array([0]), ["river", "river", "urban"],
    )
    assert pairs.tolist() == [[0, 1]]
    assert _undersized_cell_repair_pairs(
        polygons[:2], np.array([0]), ["urban", "rural"], same_class_only=False,
    ).tolist() == [[0, 1]]


def test_river_ribbon_seeds_are_separated_and_flow_aligned() -> None:
    river = np.ones((1, 12), dtype=bool)
    receiver = np.arange(12, dtype=np.int64).reshape(1, 12) + 1
    receiver[0, -1] = -1
    seeds = _river_seeds(
        river, receiver, Affine.scale(30, -30), cross_width_m=np.full(river.shape, 45.0), minimum_distance_m=30,
    )
    distances = np.hypot(seeds[:, None, 0] - seeds[None, :, 0], seeds[:, None, 1] - seeds[None, :, 1])
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 30
    assert len(np.unique(seeds[:, 1])) == 3  # cross-stream rows, not a single D4-centreline row


def test_no_neal_policy_can_seed_a_river_narrower_than_the_mesh_floor() -> None:
    records = [(0, LineString([(0, 0), (90, 0)]), 5.0, (1.0, 0.0))]
    assert len(_river_core_seeds(records, along_m=30, cross_target_m=90, minimum_width_m=30)) == 0
    seeds = _river_core_seeds(records, along_m=30, cross_target_m=90, minimum_width_m=0)
    assert len(seeds) == 3
    assert np.allclose(seeds[:, 1], 0.0)


def test_no_neal_policy_does_not_widen_a_narrow_channel_to_preferred_rows() -> None:
    records = [(0, LineString([(0, 0), (90, 0)]), 5.0, (1.0, 0.0))]
    seeds = _river_core_seeds(
        records,
        along_m=30,
        cross_target_m=30,
        minimum_width_m=0,
        minimum_seed_distance_m=30,
    )
    assert len(seeds) == 3
    assert np.allclose(np.unique(seeds[:, 1]), [0.0])


def test_bank_support_seeds_bound_a_narrow_channel_without_becoming_channel_seeds() -> None:
    records = [(0, LineString([(0, 0), (90, 0)]), 5.0, (1.0, 0.0))]
    support = _river_bank_support_seeds(records, along_m=30, cross_target_m=30, minimum_seed_distance_m=30)
    assert len(support) == 6
    assert np.allclose(np.unique(support[:, 1]), [-32.5, 32.5])


def test_d8_reach_widths_are_smoothed_before_bank_buffering() -> None:
    smoothed = _smooth_width_values(np.asarray([30.0, 30.0, 300.0, 30.0, 30.0]), 3)
    assert smoothed.max() < 300.0
    assert np.isclose(smoothed[2], 120.0)


def test_mesh_host_width_does_not_change_physical_river_width() -> None:
    records = [(0, LineString([(0, 0), (90, 0)]), 5.0, (1.0, 0.0))]
    physical = _physical_river_geometry(records)
    host = _physical_river_geometry(records, minimum_width_m=90)
    assert np.isclose(physical.bounds[3] - physical.bounds[1], 5.0)
    assert np.isclose(host.bounds[3] - host.bounds[1], 90.0)


def test_source_river_point_does_not_claim_a_whole_polygon() -> None:
    polygons = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    classes = _feature_classes(
        polygons, Polygon(), Polygon(), np.zeros((1, 2), dtype=bool),
        Affine.translation(0, 10) * Affine.scale(10, -10),
        np.asarray([Point(2, 5)], dtype=object),
    )
    assert classes == ["rural", "rural"]


def test_waterbody_class_overrides_river_and_floodplain() -> None:
    polygons = [box(0, 0, 10, 10)]
    transform = Affine.translation(0, 10) * Affine.scale(10, -10)
    classes = _feature_classes(
        polygons, box(0, 0, 10, 10), box(0, 0, 10, 10),
        np.zeros((1, 1), dtype=bool), transform,
        np.asarray([Point(5, 5)], dtype=object),
        waterbody_geometry=box(0, 0, 10, 10),
    )
    assert classes == ["waterbody"]


def test_hand_ribbon_is_created_before_background_sites_and_keeps_flow_axis() -> None:
    hand = np.ones((1, 12), dtype=bool)
    receiver = np.arange(12, dtype=np.int64).reshape(1, 12) + 1
    receiver[0, -1] = -1
    seeds = _flow_aligned_ribbon_seeds(
        hand, receiver, Affine.scale(30, -30),
        cross_width_m=np.full(hand.shape, 45.0),
        along_width_m=np.full(hand.shape, 90.0),
        minimum_distance_m=30.0,
    )
    # The core has three flow-normal rows at 45 m spacing and no pair violates
    # the hydraulic floor before transition/rural sites are introduced.
    assert len(np.unique(seeds[:, 1])) == 3
    distances = np.hypot(seeds[:, None, 0] - seeds[None, :, 0], seeds[:, None, 1] - seeds[None, :, 1])
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 30.0


def test_floodplain_connector_ribbon_uses_cross_stream_rows() -> None:
    mask = np.ones((8, 12), dtype=bool)
    geometry = box(0, -240, 360, 0)
    seeds = _floodplain_connector_ribbon_seeds(
        [LineString(((15, -105), (345, -105)))], geometry, mask,
        Affine.scale(30, -30), along_m=60.0, cross_m=60.0, minimum_distance_m=30.0,
    )
    assert len(np.unique(seeds[:, 1])) >= 3


def test_complete_floodplain_axis_removes_the_unoriented_fallback_gap() -> None:
    floodplain = np.ones((7, 12), dtype=bool)
    receiver = np.full(floodplain.shape, -1, dtype=np.int64)
    for row in range(floodplain.shape[0]):
        receiver[row, :-1] = np.arange(row * floodplain.shape[1] + 1, (row + 1) * floodplain.shape[1])
    initial_axis = np.zeros_like(floodplain)
    completed = _complete_floodplain_connector_axis(
        initial_axis, receiver, floodplain, Affine.scale(30, -30), cross_m=60.0,
    )
    from scipy.ndimage import distance_transform_edt
    assert completed.sum() > initial_axis.sum()
    assert distance_transform_edt(~completed, sampling=(30.0, 30.0))[floodplain].max() <= 60.0


def test_floodplain_connector_tangent_smooths_a_d4_staircase_from_cell_size() -> None:
    axis = np.zeros((3, 3), dtype=bool)
    axis[(0, 0, 1, 1, 2), (0, 1, 1, 2, 2)] = True
    receiver = np.full(axis.shape, -1, dtype=np.int64)
    receiver[0, 0] = 1
    receiver[0, 1] = 4
    receiver[1, 1] = 5
    receiver[1, 2] = 8
    tangent_x, tangent_y, lines = _smoothed_floodplain_connector_tangents(
        axis, receiver, Affine.scale(30, -30), along_cell_length_m=40.0,
    )
    assert len(lines) == 1
    assert tangent_x[0, 1] > 0
    assert tangent_y[0, 1] < 0


def test_short_voronoi_cell_repair_removes_near_duplicate_generator() -> None:
    seeds = np.array([[30, 30], [30, 31], [0, 0], [60, 0], [0, 60], [60, 60]], dtype=float)
    repaired, polygons, iterations = _repair_short_cells(
        seeds, box(0, 0, 60, 60), minimum_cell_width_m=40, protected_seed_count=4,
    )
    scale = np.array([2 * item.area / item.length for item in polygons])
    assert iterations > 0
    assert len(repaired) < len(seeds)
    assert scale.min() >= 15  # protected envelope anchors bound the remaining 30 m squares


def test_short_cell_repair_never_moves_a_protected_feature_seed() -> None:
    seeds = np.array([[30, 30], [30, 31], [0, 0], [60, 0], [0, 60], [60, 60]], dtype=float)
    repaired, _, _ = _repair_short_cells(
        seeds, box(0, 0, 60, 60), minimum_cell_width_m=40,
        protected_seed_count=4, protected_seed_indices=np.array([0]),
    )
    assert any(np.array_equal(seed, [30, 30]) for seed in repaired)


def test_face_quality_flags_short_internal_face() -> None:
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
        Polygon([(10, 0), (20, 0), (20, 1), (10, 1)]),
    ]
    qa = _edge_quality(
        polygons, np.array([[5.0, 0.5], [15.0, 0.5]]), np.full((2, 2), 30.0),
        Affine.translation(0, 30) * Affine.scale(30, -30), minimum_face_length_m=7.5, minimum_center_distance_m=15.0,
    )
    assert qa["min_internal_face_length_m"] == 1.0
    assert qa["short_internal_face_count"] == 1


def test_face_quality_ignores_repeated_zero_length_ring_segments() -> None:
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
    ]
    qa = _edge_quality(
        polygons, np.array([[5.0, 5.0], [15.0, 5.0]]), np.full((2, 2), 30.0),
        Affine.translation(0, 30) * Affine.scale(30, -30), minimum_face_length_m=7.5, minimum_center_distance_m=15.0,
    )
    assert qa["min_internal_face_length_m"] == 10.0


def test_topology_collapses_machine_scale_false_face() -> None:
    polygons = [
        Polygon([(0, 0), (10, 0), (10.00000001, 0), (10, 10), (0, 10)]),
        Polygon([(10, 0), (20, 0), (20, 10), (10, 10), (10.00000001, 0)]),
    ]
    faces, edges = _topology(polygons)
    rebuilt = [Polygon(face) for face in faces]
    assert all(edge[0] != edge[1] for edge in edges)
    assert min(
        np.hypot(a[0] - b[0], a[1] - b[1])
        for face in faces for a, b in zip(face, face[1:] + face[:1], strict=True)
    ) > 0
    assert all(item.is_valid for item in rebuilt)


def test_polygon_feature_clipping_makes_the_channel_a_real_cell_boundary() -> None:
    pieces = _clip_cells_to_feature_polygon([box(0, 0, 100, 100)], box(40, 0, 70, 100), 1.0)
    assert len(pieces) == 3
    assert np.isclose(sum(piece.area for piece in pieces), 10000.0)
    assert sorted(round(piece.bounds[2] - piece.bounds[0], 1) for piece in pieces) == [30.0, 30.0, 40.0]


def test_polygon_feature_clipping_preserves_cell_when_it_would_make_a_sliver() -> None:
    pieces = _clip_cells_to_feature_polygon([box(0, 0, 100, 100)], box(98, 0, 100, 100), 500.0)
    assert len(pieces) == 1
    assert np.isclose(pieces[0].area, 10000.0)


def test_exact_feature_fraction_queries_only_intersecting_polygons() -> None:
    polygons = [box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)]
    fractions = _geometry_area_fractions(polygons, box(5, 0, 15, 10))
    assert np.allclose(fractions, [0.5, 0.5, 0.0])


def test_fraction_refined_classes_do_not_paint_partial_channel_cells_as_river() -> None:
    classes = _fraction_refined_feature_classes(
        ["river", "river", "floodplain", "river"],
        np.array([0.8, 0.2, 0.6, 0.9]),
        np.array([0.0, 0.6, 0.8, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.2]),
    )
    assert classes == ["river", "floodplain", "river", "waterbody"]


def test_short_face_agglomeration_removes_only_the_pathological_interface() -> None:
    polygons, count = _agglomerate_short_face_cells(
        [
            Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
            Polygon([(10, 0), (20, 0), (20, 1), (10, 1)]),
        ],
        minimum_face_length_m=7.5,
    )
    assert count == 1
    assert len(polygons) == 1
    assert polygons[0].area == 20


def test_batched_agglomeration_never_merges_across_feature_classes() -> None:
    polygons, count = _agglomerate_short_face_cells(
        [
            Polygon([(0, 0), (10, 0), (10, 1), (0, 1)]),
            Polygon([(10, 0), (20, 0), (20, 1), (10, 1)]),
        ],
        minimum_face_length_m=7.5,
        classes=["river", "rural"],
        candidate_pairs=np.array([[0, 1]]),
    )
    assert count == 0
    assert len(polygons) == 2


def test_feature_boundary_split_queries_only_intersecting_cells() -> None:
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
    ]
    split_polygons = _split_by_feature_boundaries(
        polygons, LineString([(5, -5), (5, 15)]), minimum_piece_area_m2=1.0,
    )
    assert len(split_polygons) == 3
    assert sum(item.area for item in split_polygons) == 200


def test_voronoi_boundary_clip_keeps_all_cells_in_a_nonrectangular_domain() -> None:
    domain = Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)])
    polygons = _polygons_from_seeds(
        np.array([[5, 5], [20, 5], [5, 20], [8, 8]], dtype=float), domain,
    )
    assert polygons
    assert all(domain.covers(polygon) for polygon in polygons)


def test_feature_seed_thinning_enforces_the_hydraulic_floor_at_a_confluence() -> None:
    seeds = _minimum_separated(
        np.array([[0.0, 0.0], [0.01, 0.01], [30.0, 0.0], [60.0, 0.0]]), 30.0,
    )
    distance = np.hypot(seeds[:, None, 0] - seeds[None, :, 0], seeds[:, None, 1] - seeds[None, :, 1])
    np.fill_diagonal(distance, np.inf)
    assert distance.min() >= 30.0


def test_ugrid_export_matches_hydropol_boundary_contract(tmp_path: Path) -> None:
    # Both cells carry the intermediate vertex, which is what a conforming mesh
    # looks like: the shared interface is two segments owned by both cells, so
    # the export has to merge them into one finite-volume face.
    polygons = [
        Polygon(((0, 0), (10, 0), (10, 5), (10, 10), (0, 10))),
        Polygon(((10, 0), (20, 0), (20, 10), (10, 10), (10, 5))),
    ]
    faces, edges = _topology(polygons)
    polygons = [Polygon(face) for face in faces]
    assert len([key for key, owners in edges.items() if len(owners) == 2]) == 2
    output = tmp_path / "mesh.nc"
    _write_ugrid(
        output, faces, edges, polygons, None,
        np.full((1, 2), 10.0), Affine.translation(0, 10) * Affine.scale(10, -10),
        np.ones((1, 2)), ["rural", "urban"], np.zeros(2), np.zeros(2), np.zeros(2),
    )
    import netCDF4

    with netCDF4.Dataset(output) as dataset:
        assert "cell_refinement_source" in dataset.variables
        assert "cell_river_channel_fraction" in dataset.variables
        assert "cell_waterbody_fraction" in dataset.variables
        assert np.all(dataset["edge_center_distance_m"][:] > 0)
        assert np.allclose(dataset["cell_cfl_width_m"][:], 10.0)
        owner = np.asarray(dataset["edge_owner"][:])
        neighbor = np.asarray(dataset["edge_neighbor"][:])
        pairs = np.column_stack((owner[neighbor >= 0], neighbor[neighbor >= 0]))
        assert len(pairs) == len(np.unique(np.sort(pairs, axis=1), axis=0))


def _unpaired_ring_segments(polygons: list[Polygon]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Ring segments no second cell matches -- the mesh-conformity invariant."""
    owners: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for index, polygon in enumerate(polygons):
        ring = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        for left, right in zip(ring, ring[1:] + ring[:1]):
            if left != right:
                owners.setdefault(tuple(sorted((left, right))), []).append(index)
    return [key for key, value in owners.items() if len(value) == 1]


def test_graded_rows_is_the_default_river_cell_style() -> None:
    """The channel is seeded into the one diagram, not cut into it afterwards."""
    config = HybridMeshConfig.from_mapping({"dem": "dem.tif", "out_dir": "out"})
    assert config.river_cell_style == "graded_rows"


def test_graded_row_plan_never_breaks_the_size_ratio() -> None:
    """Each row may be at most ``size_ratio`` times the one inside it.

    Without these rows a 30 m channel cell touches a 200 m rural cell directly, a
    jump of 6.67 that both solvers reject.
    """
    plan = graded_row_plan(30.0, 200.0, 2.0)
    assert plan == [60.0, 120.0, 200.0]
    for inner, outer in zip([30.0] + plan[:-1], plan, strict=True):
        assert outer / inner <= 2.0 + 1e-9
    assert graded_row_plan(30.0, 30.0, 2.0) == []          # nothing to grade
    assert graded_row_plan(200.0, 30.0, 2.0) == []         # channel already coarser


def test_river_cell_style_is_a_grouped_river_parameter() -> None:
    config = HybridMeshConfig.from_mapping(
        {"inputs": {"dem": "dem.tif", "out_dir": "out"}, "rivers": {"cell_style": "voronoi"}}
    )
    assert config.river_cell_style == "voronoi"


def test_unknown_river_cell_style_is_rejected() -> None:
    try:
        HybridMeshConfig.from_mapping(
            {"inputs": {"dem": "dem.tif", "out_dir": "out"}, "rivers": {"cell_style": "quadtree"}}
        )
    except ValueError as error:
        assert "cell_style" in str(error)
    else:
        raise AssertionError("an unknown river cell style must be rejected")


def test_narrow_reach_strip_is_one_row_at_the_hydraulic_floor() -> None:
    """A 8 m mapped channel is meshed at the 30 m floor, not widened further."""
    cells, attributes, _, _ = build_river_strip_cells(
        LineString(((0.0, 0.0), (300.0, 0.0))), np.full(2, 8.0),
        along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0,
    )
    assert len(cells) == 10
    assert all(attribute["cross_band_count"] == 1 for attribute in attributes)
    assert np.allclose([cell.area for cell in cells], 900.0)
    bounds = np.asarray([cell.bounds for cell in cells])
    assert np.allclose(bounds[:, 3] - bounds[:, 1], 30.0)  # cross-flow
    assert np.allclose(bounds[:, 2] - bounds[:, 0], 30.0)  # along-flow


def test_wide_reach_strip_adds_rows_without_changing_channel_width() -> None:
    cells, attributes, _, _ = build_river_strip_cells(
        LineString(((0.0, 0.0), (300.0, 0.0))), np.full(2, 95.0),
        along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0,
    )
    assert attributes[0]["cross_band_count"] == 3
    bounds = np.asarray([cell.bounds for cell in cells])
    assert np.isclose(bounds[:, 3].max() - bounds[:, 1].min(), 95.0)


def test_strip_cell_size_is_prescribed_not_emergent() -> None:
    """No strip cell may exceed its own station box, unlike a Voronoi cell."""
    stations = np.linspace(0.0, 2.0 * np.pi, 40)
    centerline = LineString(np.column_stack((stations * 120.0, 200.0 * np.sin(stations))))
    cells, _, _, _ = build_river_strip_cells(
        centerline, np.full(40, 60.0), along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0,
    )
    assert cells
    ceiling = 30.0 * 60.0
    assert max(cell.area for cell in cells) <= ceiling * 1.05


def test_strip_cells_do_not_fold_through_a_sharp_bend() -> None:
    """A normal-offset ribbon must stay single valued where the bend is tight.

    The bend here is deliberately unphysical -- a 90 m channel meandering with a
    60 m amplitude, so its radius of curvature is smaller than its own width.
    The per-reach builder guarantees valid cells and reports the width it had to
    clamp; it does not guarantee non-overlap on input like this, because a
    centreline plus a width cannot describe that channel.  Non-overlap is the
    assembly's contract, asserted below and in the confluence test.
    """
    stations = np.linspace(0.0, 4.0 * np.pi, 60)
    centerline = LineString(np.column_stack((stations * 25.0, 60.0 * np.sin(stations))))
    cells, attributes, _, _ = build_river_strip_cells(
        centerline, np.full(60, 90.0), along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0,
    )
    assert all(cell.is_valid and cell.area > 0 for cell in cells)
    assert min(item["channel_width_m"] for item in attributes) < 90.0  # clamped, and reported
    assembled, _, _, _ = build_river_strip_mesh(
        [(0, centerline, np.full(60, 90.0))], 30.0, 30.0, 30.0,
        box(-500.0, -500.0, 3000.0, 500.0),
    )
    assert sum(cell.area for cell in assembled) == approx(
        unary_union(assembled).area, rel=1e-9,
    )


def test_fold_limit_leaves_a_straight_reach_untouched() -> None:
    stations = np.asarray([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    normals = _station_normals(stations)
    half = np.full(3, 45.0)
    assert np.allclose(_fold_limited_half_width(stations, normals, half), half)


def test_reach_paths_split_at_a_confluence() -> None:
    """Each path is single thread, so one cross-section series fits each reach."""
    river = np.zeros((4, 3), dtype=bool)
    river[:, 1] = True
    river[1, 0] = True
    receiver = np.full(12, -1, dtype=np.int64)
    for row in range(3):
        receiver[row * 3 + 1] = (row + 1) * 3 + 1
    receiver[3] = 7  # tributary cell (1, 0) joins the stem at (2, 1)
    paths, leftover = _river_reach_paths(river, receiver.reshape(4, 3))
    assert sorted(len(path) for path in paths) == [2, 2, 3]
    assert not leftover


def test_reach_records_carry_a_width_per_smoothed_vertex() -> None:
    river = np.zeros((5, 3), dtype=bool)
    river[:, 1] = True
    receiver = np.full((5, 3), -1, dtype=np.int64)
    for row in range(4):
        receiver[row, 1] = (row + 1) * 3 + 1
    width = np.where(river, 40.0, np.nan)
    records = _river_reach_records(
        river, receiver, Affine.translation(0, 150) * Affine.scale(30, -30), width,
        centerline_smoothing_iterations=2, width_smoothing_window_cells=3,
    )
    assert len(records) == 1
    _, line, widths = records[0]
    assert len(widths) == len(line.coords)
    assert np.allclose(widths, 40.0)


def test_strip_cells_never_enter_a_waterbody() -> None:
    reaches = [(0, LineString(((0.0, 0.0), (600.0, 0.0))), np.full(2, 40.0))]
    lake = box(200.0, -100.0, 400.0, 100.0)
    cells, _, _, _ = build_river_strip_mesh(
        reaches, 30.0, 30.0, 30.0, box(-500.0, -500.0, 1100.0, 500.0), lake,
    )
    assert cells
    assert unary_union(cells).intersection(lake).area == 0.0


def test_confluence_overlap_is_trimmed_from_the_narrower_reach() -> None:
    """The wider reach is the main stem and keeps its cross-section intact."""
    reaches = [
        (0, LineString(((0.0, 0.0), (600.0, 0.0))), np.full(2, 90.0)),
        (1, LineString(((300.0, 300.0), (300.0, 0.0))), np.full(2, 30.0)),
    ]
    cells, attributes, _, discarded = build_river_strip_mesh(
        reaches, 30.0, 30.0, 30.0, box(-500.0, -500.0, 1100.0, 800.0),
    )
    assert discarded >= 0.0
    assert sum(cell.area for cell in cells) == approx(unary_union(cells).area, abs=1e-6)
    stem = [cell for cell, item in zip(cells, attributes) if item["reach_id"] == 0]
    assert all(len(cell.exterior.coords) - 1 == 4 for cell in stem)


def test_hanging_node_insertion_pairs_a_t_junction() -> None:
    polygons = [box(0, 0, 30, 30), box(30, 0, 60, 15), box(30, 15, 60, 30)]
    assert len(_unpaired_ring_segments(polygons)) == 10
    repaired, inserted, snapped = insert_hanging_nodes(polygons)
    assert inserted == 1 and snapped == 0
    assert len(_unpaired_ring_segments(repaired)) == 7  # outer perimeter only
    assert sum(item.area for item in repaired) == approx(sum(item.area for item in polygons))


def test_hanging_node_insertion_snaps_rather_than_cutting_a_short_face() -> None:
    """Inserting a node 2 m from a station would make a 2 m finite-volume face."""
    polygons = [box(0, 0, 30, 30), box(30, 0, 60, 2), box(30, 2, 60, 30)]
    repaired, inserted, snapped = insert_hanging_nodes(polygons, minimum_face_length_m=7.5)
    assert snapped >= 1
    assert not _interior_short_faces(repaired, 7.5)


def _interior_short_faces(polygons: list[Polygon], minimum_length_m: float) -> list[float]:
    owners: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = {}
    for index, polygon in enumerate(polygons):
        ring = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        for left, right in zip(ring, ring[1:] + ring[:1]):
            if left != right:
                owners.setdefault(tuple(sorted((left, right))), []).append(index)
    return [
        float(np.hypot(key[0][0] - key[1][0], key[0][1] - key[1][1]))
        for key, value in owners.items()
        if len(value) == 2
        and np.hypot(key[0][0] - key[1][0], key[0][1] - key[1][1]) < minimum_length_m
    ]


def test_short_face_weld_removes_a_sub_floor_interface() -> None:
    """A 1 m interface controls the timestep while carrying no flow."""
    polygons = [
        Polygon(((0, 0), (30, 0), (30, 15), (30, 16), (0, 30))),
        Polygon(((0, 30), (30, 16), (30, 15), (30, 30))),
    ]
    assert _interior_short_faces(polygons, 7.5)
    welded, count = collapse_short_faces(polygons, 7.5)
    assert count >= 1
    assert not _interior_short_faces(welded, 7.5)


def test_weld_never_collapses_a_cell_into_a_spike() -> None:
    """GEOS calls an out-and-back ring valid, so area alone cannot detect it."""
    polygons = [Polygon(((0, 0), (30, 0), (30, 1), (0, 1)))]
    welded, _ = collapse_short_faces(polygons, 7.5)
    assert all(2.0 * item.area / item.length > 1e-3 for item in welded)


def _self_paired_cells(polygons: list[Polygon]) -> list[int]:
    """Cells whose own ring walks a segment twice, making them their own neighbour."""
    result: list[int] = []
    for index, polygon in enumerate(polygons):
        ring = [(round(x, 4), round(y, 4)) for x, y in list(polygon.exterior.coords)[:-1]]
        keys = [tuple(sorted((left, right))) for left, right in zip(ring, ring[1:] + ring[:1])]
        keys = [key for key in keys if key[0] != key[1]]
        if len(set(keys)) != len(keys):
            result.append(index)
    return result


def test_self_touching_cell_stops_being_its_own_neighbour() -> None:
    """A repeated ring segment makes _edge_quality pair a cell with itself,
    which no merge can repair because there is only one cell."""
    pinched = Polygon(((0, 0), (30, 0), (30, 30), (15, 15), (0, 30), (15, 15)))
    assert _self_paired_cells([pinched]) == [0]
    result, _ = split_self_touching_cells([pinched])
    assert not _self_paired_cells(result)
    assert all(item.is_valid and item.area > 0 for item in result)


def test_degenerate_fragment_is_absorbed_by_its_largest_neighbour() -> None:
    sliver = Polygon(((60, 0), (60.2, 0), (60.2, 60), (60, 60)))
    polygons = [box(0, 0, 60, 60), sliver, box(60.2, 0, 120, 60)]
    assert 2.0 * sliver.area / sliver.length < 15.0
    assert all(2.0 * polygons[index].area / polygons[index].length > 15.0 for index in (0, 2))
    result, absorbed = absorb_degenerate_cells(polygons, 15.0)
    assert absorbed == 1
    assert len(result) == 2
    assert sum(item.area for item in result) == approx(sum(item.area for item in polygons))


def test_partition_void_is_returned_to_a_neighbour() -> None:
    """A void is a hole in the partition: mass entering it would be lost."""
    domain = box(0, 0, 60, 30)
    polygons = [box(0, 0, 30, 30), Polygon(((30, 0), (60, 0), (60, 30), (30, 30), (40, 15)))]
    void = domain.difference(unary_union(polygons))
    assert void.area > 0
    filled, count, area, searched = fill_partition_voids(polygons, domain, 15.0)
    assert searched
    assert count == 1
    assert area == approx(void.area)
    assert domain.difference(unary_union(filled)).area == approx(0.0, abs=1e-9)


def test_hole_free_parts_splits_a_cell_that_encloses_the_ribbon() -> None:
    """_topology reads only the exterior ring, so an enclosed interface is lost."""
    enclosing = Polygon(
        ((0, 0), (90, 0), (90, 90), (0, 90)),
        holes=[((30, 30), (60, 30), (60, 60), (30, 60))],
    )
    parts = hole_free_parts(enclosing)
    assert len(parts) > 1
    assert not any(part.interiors for part in parts)
    assert sum(part.area for part in parts) == approx(enclosing.area)


def test_substitute_river_strip_cuts_the_channel_and_drops_the_ribbon_in() -> None:
    background = [box(0, 0, 90, 90)]
    strip = [box(30, index * 30.0, 60, index * 30.0 + 30.0) for index in range(3)]
    channel = unary_union(strip)
    result, is_strip = substitute_river_strip(background, channel, strip)
    assert is_strip.sum() == 3
    assert sum(item.area for item in result) == approx(90.0 * 90.0)
    assert not any(item.interiors for item in result)
    assert unary_union([
        item for item, strip_cell in zip(result, is_strip) if not strip_cell
    ]).intersection(channel).area == approx(0.0, abs=1e-9)


def test_conform_and_repair_preserves_the_cell_partition() -> None:
    domain = box(0, 0, 90, 90)
    background = [box(0, 0, 90, 90)]
    strip = [box(30, index * 30.0, 60, index * 30.0 + 30.0) for index in range(3)]
    polygons, _ = substitute_river_strip(background, unary_union(strip), strip)
    repaired, report = conform_and_repair_cells(polygons, 7.5, 15.0, domain)
    assert sum(item.area for item in repaired) == approx(domain.area)
    assert domain.difference(unary_union(repaired)).area == approx(0.0, abs=1e-6)
    assert not _interior_short_faces(repaired, 7.5)
    assert report["inserted"] >= 0


def test_internal_unpaired_edges_ignores_the_true_domain_boundary() -> None:
    domain = box(0, 0, 60, 30)
    faces, edges = _topology([box(0, 0, 30, 30), box(30, 0, 60, 30)])
    count, length = internal_unpaired_edges(edges, domain)
    assert count == 0 and length == 0.0
    hanging = {((10.0, 10.0), (20.0, 10.0)): [0]}
    count, length = internal_unpaired_edges({**edges, **hanging}, domain)
    assert count == 1 and length == approx(10.0)


def test_floodplain_is_the_candidate_minus_channel_and_waterbody() -> None:
    candidate = box(0, 0, 90, 90)
    river = box(30, 0, 60, 90)
    lake = box(0, 0, 30, 30)
    result = exclusive_floodplain_geometry(candidate, river, lake)
    assert result.intersection(river).area == approx(0.0, abs=1e-9)
    assert result.intersection(lake).area == approx(0.0, abs=1e-9)
    assert result.area == approx(candidate.area - river.area - lake.area)


def test_floodplain_without_blockers_is_the_candidate() -> None:
    candidate = box(0, 0, 90, 90)
    assert exclusive_floodplain_geometry(candidate, Polygon(), Polygon()).area == approx(candidate.area)
    assert exclusive_floodplain_geometry(Polygon(), Polygon(), Polygon()).is_empty


def _normal_closure_error(dataset, cells: int) -> np.ndarray:
    """Relative |sum(L * n)| per cell: zero for a closed finite-volume cell."""
    owner = np.asarray(dataset["edge_owner"][:], dtype=np.int64)
    neighbour = np.asarray(dataset["edge_neighbor"][:], dtype=np.int64)
    length = np.asarray(dataset["edge_length_m"][:], dtype=np.float64)
    normal_x = np.asarray(dataset["edge_normal_x"][:], dtype=np.float64)
    normal_y = np.asarray(dataset["edge_normal_y"][:], dtype=np.float64)
    internal = neighbour >= 0
    sum_x = np.zeros(cells)
    sum_y = np.zeros(cells)
    perimeter = np.zeros(cells)
    np.add.at(sum_x, owner, length * normal_x)
    np.add.at(sum_y, owner, length * normal_y)
    np.add.at(perimeter, owner, length)
    np.subtract.at(sum_x, neighbour[internal], length[internal] * normal_x[internal])
    np.subtract.at(sum_y, neighbour[internal], length[internal] * normal_y[internal])
    np.add.at(perimeter, neighbour[internal], length[internal])
    return np.hypot(sum_x, sum_y) / np.maximum(perimeter, 1e-12)


def test_merged_face_keeps_the_cell_closed(tmp_path: Path) -> None:
    """Two cells meeting along two non-collinear segments export as one face.

    Pairing the summed segment length with the mean segment direction leaves the
    cell's outward normals not summing to zero, so a uniform flux has non-zero
    divergence and the scheme is no longer conservative.
    """
    import netCDF4

    wrapping = Polygon(((0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)))
    enclosed = box(10, 10, 30, 30)
    polygons = [wrapping, enclosed]
    faces, edges = _topology(polygons)
    polygons = [Polygon(face) for face in faces]
    shared = [key for key, owners in edges.items() if len(owners) == 2]
    assert len(shared) == 2  # the merge this test is about actually happens
    output = tmp_path / "mesh.nc"
    report = _write_ugrid(
        output, faces, edges, polygons, None,
        np.full((1, 1), 30.0), Affine.translation(0, 30) * Affine.scale(30, -30),
        np.ones((1, 1)), ["rural", "river"], np.zeros(2), np.zeros(2), np.zeros(2),
    )
    assert report["merged_internal_faces"] == 1
    assert report["non_collinear_merged_faces"] == 1
    assert report["merged_face_length_deficit_m"] > 0.0
    with netCDF4.Dataset(output) as dataset:
        owner = np.asarray(dataset["edge_owner"][:])
        neighbour = np.asarray(dataset["edge_neighbor"][:])
        internal = neighbour >= 0
        assert internal.sum() == 1
        assert np.max(_normal_closure_error(dataset, len(polygons))) == approx(0.0, abs=1e-12)
        pairs = np.column_stack((owner[internal], neighbour[internal]))
        assert len(pairs) == len(np.unique(np.sort(pairs, axis=1), axis=0))


def test_collinear_merge_keeps_the_full_interface_length(tmp_path: Path) -> None:
    """The common case this merge exists for must lose no conveyance width."""
    import netCDF4

    left = Polygon(((0, 0), (30, 0), (30, 15), (30, 30), (0, 30)))
    right = Polygon(((30, 0), (60, 0), (60, 30), (30, 30), (30, 15)))
    faces, edges = _topology([left, right])
    assert len([key for key, owners in edges.items() if len(owners) == 2]) == 2
    polygons = [Polygon(face) for face in faces]
    output = tmp_path / "mesh.nc"
    report = _write_ugrid(
        output, faces, edges, polygons, None,
        np.full((1, 2), 30.0), Affine.translation(0, 30) * Affine.scale(30, -30),
        np.ones((1, 2)), ["rural", "rural"], np.zeros(2), np.zeros(2), np.zeros(2),
    )
    assert report["non_collinear_merged_faces"] == 0
    assert report["merged_face_length_deficit_m"] == approx(0.0)
    with netCDF4.Dataset(output) as dataset:
        neighbour = np.asarray(dataset["edge_neighbor"][:])
        length = np.asarray(dataset["edge_length_m"][:])
        assert length[neighbour >= 0].sum() == approx(30.0)
        assert np.max(_normal_closure_error(dataset, len(polygons))) == approx(0.0, abs=1e-12)


def test_ribbon_width_tracks_mapped_width_above_the_floor() -> None:
    """The promise of the structured style: cross-stream size is prescribed.

    Below the hydraulic floor the channel is meshed at the floor and not widened
    further; above it the ribbon reproduces the mapped width, adding cross bands
    rather than stretching cells.
    """
    reach = LineString(((0.0, 0.0), (600.0, 0.0)))
    expected_bands = {5: 1, 15: 1, 29: 1, 31: 1, 45: 2, 60: 2, 90: 3, 120: 4}
    for mapped, bands in expected_bands.items():
        cells, attributes, _, _ = build_river_strip_cells(
            reach, np.full(2, float(mapped)), along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0,
        )
        ribbon_width = sum(cell.area for cell in cells) / reach.length
        assert ribbon_width == approx(max(float(mapped), 30.0)), mapped
        assert attributes[0]["cross_band_count"] == bands, mapped
        # Every face of every cell stays above the quality floor by construction.
        assert min(cell.area for cell in cells) / 30.0 >= 7.5


def test_station_attributes_tile_the_reach_without_gaps() -> None:
    """Station bounds are the model's handle on a river cell, so they must be
    contiguous and cover the reach exactly."""
    reach = LineString(((0.0, 0.0), (300.0, 0.0), (300.0, 300.0)))
    cells, attributes, _, _ = build_river_strip_cells(
        reach, np.full(3, 45.0), along_m=30.0, cross_target_m=30.0, minimum_width_m=30.0, reach_id=7,
    )
    assert all(item["reach_id"] == 7 for item in attributes)
    bounds = sorted({(item["station_start_m"], item["station_end_m"]) for item in attributes})
    assert bounds[0][0] == approx(0.0)
    assert bounds[-1][1] == approx(reach.length)
    for (_, end), (following, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert end == approx(following)
    # Every station interval carries a full set of cross bands.
    per_interval = {}
    for item in attributes:
        per_interval.setdefault((item["station_start_m"], item["station_end_m"]), set()).add(item["cross_band"])
    for interval, bands in per_interval.items():
        assert bands == set(range(len(bands))), interval


def test_crescent_cell_is_split_until_it_holds_its_own_centroid() -> None:
    """A cell-centred scheme measures face distances from the cell centre, so a
    cell whose centroid lies outside it exports meaningless distances."""
    crescent = Polygon(((0, 0), (90, 0), (90, 30), (30, 30), (30, 60), (90, 60), (90, 90), (0, 90)))
    assert not crescent.contains(crescent.centroid)
    result, count = split_cells_without_interior_centroid([crescent])
    assert count >= 1
    assert sum(item.area for item in result) == approx(crescent.area)
    assert all(item.contains(item.centroid) for item in result)


def test_convex_cells_are_left_alone_by_the_centroid_split() -> None:
    polygons = [box(0, 0, 30, 30), box(30, 0, 60, 30)]
    result, count = split_cells_without_interior_centroid(polygons)
    assert count == 0
    assert len(result) == 2
