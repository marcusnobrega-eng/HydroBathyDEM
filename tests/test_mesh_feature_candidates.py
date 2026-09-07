import numpy as np
import geopandas as gpd
from affine import Affine
from shapely.geometry import box

from dem_processing.mesh_feature_candidates import (
    MeshFeatureCandidateConfig,
    _filter_small_components,
    _fill_small_mask_holes,
    _major_river_mask,
    _nearest_river_hand,
    _refine_waterbodies_from_dem,
)


def test_mesh_feature_candidate_grouped_config() -> None:
    config = MeshFeatureCandidateConfig.from_mapping({
        "inputs": {
            "dem": "dem.tif",
            "domain_vector": "domain.geojson",
            "river_mask": "river.tif",
            "river_direction": "direction.tif",
            "river_width": "width.tif",
            "out_dir": "out",
        },
        "rivers": {"minimum_upstream_area_km2": 25, "major_river_min_width_m": 35},
        "floodplain": {"floodplain_max_hand_m": 6, "floodplain_max_distance_m": 500},
        "breaklines": {"fetch_osm_waterbodies": True},
    })
    assert config.major_river_min_upstream_area_km2 == 25
    assert config.major_river_min_width_m == 35
    assert config.floodplain_max_hand_m == 6
    assert config.fetch_osm_waterbodies is True


def test_fill_small_mask_holes_protects_blockers() -> None:
    mask = np.ones((7, 7), dtype=bool)
    mask[2, 2] = False
    mask[4, 4] = False
    blockers = np.zeros_like(mask)
    blockers[4, 4] = True
    filled, holes = _fill_small_mask_holes(mask, blockers, maximum_area_m2=1.5, pixel_area_m2=1.0)
    assert filled[2, 2]
    assert holes[2, 2]
    assert not filled[4, 4]
    assert not holes[4, 4]


def test_dem_waterbody_refinement_expands_connected_flat_surface() -> None:
    dem = np.full((5, 5), 20.0)
    dem[1:4, 1:4] = 10.0
    transform = Affine.translation(0, 0) * Affine.scale(30, -30)
    waterbodies = gpd.GeoDataFrame(
        {"source": ["test"], "waterbody_id": [1], "waterbody_type": ["reservoir"], "name": [None], "area_m2": [3600.0]},
        geometry=[box(30, -90, 90, -30)],
        crs="EPSG:32643",
    )
    config = MeshFeatureCandidateConfig.from_mapping({
        "dem": "dem.tif",
        "domain_vector": "domain.geojson",
        "river_mask": "river.tif",
        "river_direction": "direction.tif",
        "river_width": "width.tif",
        "out_dir": "out",
        "refine_waterbodies_from_dem": True,
        "waterbody_min_area_m2": 1,
        "waterbody_refine_buffer_m": 60,
        "waterbody_refine_elevation_tolerance_m": 0.1,
        "waterbody_refine_relief_window_m": 30,
        "waterbody_refine_max_relief_m": 0.1,
        "waterbody_refine_max_area_ratio": 20,
    })
    refined = _refine_waterbodies_from_dem(
        waterbodies, dem, np.ones_like(dem, dtype=bool), transform, config, [],
    )
    assert bool(refined.iloc[0].dem_refined)
    assert refined.geometry.area.sum() > waterbodies.geometry.area.sum()


def test_major_river_mask_uses_width_or_upstream_area() -> None:
    river = np.array([[True, True, True]])
    width = np.array([[10.0, 50.0, 10.0]])
    upstream = np.array([[10.0, 10.0, 100.0]])
    config = MeshFeatureCandidateConfig.from_mapping({
        "dem": "dem.tif",
        "domain_vector": "domain.geojson",
        "river_mask": "river.tif",
        "river_direction": "direction.tif",
        "river_width": "width.tif",
        "out_dir": "out",
    })
    assert _major_river_mask(river, width, upstream, config).tolist() == [[False, True, True]]


def test_nearest_river_hand_uses_nearest_drainage_elevation() -> None:
    dem = np.array([[12.0, 11.0, 10.0], [14.0, 13.0, 10.0]])
    river = np.array([[False, False, True], [False, False, True]])
    hand, distance = _nearest_river_hand(dem, river, Affine.translation(0, 0) * Affine.scale(30, -30))
    assert hand.tolist() == [[2.0, 1.0, 0.0], [4.0, 3.0, 0.0]]
    assert distance[0, 0] == 60


def test_filter_small_components_keeps_only_large_groups() -> None:
    mask = np.array([
        [True, True, False],
        [False, False, False],
        [True, False, False],
    ])
    result = _filter_small_components(mask, minimum_area_m2=2.0, pixel_area_m2=1.0)
    assert result.tolist() == [
        [True, True, False],
        [False, False, False],
        [False, False, False],
    ]
