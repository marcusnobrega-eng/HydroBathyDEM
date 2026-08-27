from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import GeometryCollection, LineString, Polygon, box

from dem_processing.config import load_config_file
from dem_processing.hybrid_mesh import (
    HybridMeshConfig,
    _agglomerate_short_face_cells,
    _bad_face_seed_removals,
    connected_hand_with_reach,
    _edge_quality,
    _polygons_from_seeds,
    _complete_floodplain_connector_axis,
    _flow_aligned_ribbon_seeds,
    _repair_short_cells,
    _split_by_feature_boundaries,
    _river_seeds,
    _floodplain_connector_ribbon_seeds,
    _minimum_separated,
    _smoothed_floodplain_connector_tangents,
    _write_ugrid,
    receiver_from_d4_direction,
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
    polygons = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    faces = [list(polygon.exterior.coords)[:-1] for polygon in polygons]
    edges = {
        ((0.0, 0.0), (0.0, 10.0)): [0],
        ((0.0, 0.0), (10.0, 0.0)): [0],
        ((0.0, 10.0), (10.0, 10.0)): [0],
        ((10.0, 0.0), (10.0, 5.0)): [0, 1],
        ((10.0, 5.0), (10.0, 10.0)): [0, 1],
        ((10.0, 0.0), (20.0, 0.0)): [1],
        ((10.0, 10.0), (20.0, 10.0)): [1],
        ((20.0, 0.0), (20.0, 10.0)): [1],
    }
    output = tmp_path / "mesh.nc"
    _write_ugrid(
        output, faces, edges, polygons, None,
        np.full((1, 2), 10.0), Affine.translation(0, 10) * Affine.scale(10, -10),
        np.ones((1, 2)), ["rural", "urban"], GeometryCollection(), GeometryCollection(),
    )
    import netCDF4

    with netCDF4.Dataset(output) as dataset:
        assert "cell_refinement_source" in dataset.variables
        assert np.all(dataset["edge_center_distance_m"][:] > 0)
        assert np.allclose(dataset["cell_cfl_width_m"][:], 10.0)
        owner = np.asarray(dataset["edge_owner"][:])
        neighbor = np.asarray(dataset["edge_neighbor"][:])
        pairs = np.column_stack((owner[neighbor >= 0], neighbor[neighbor >= 0]))
        assert len(pairs) == len(np.unique(np.sort(pairs, axis=1), axis=0))
