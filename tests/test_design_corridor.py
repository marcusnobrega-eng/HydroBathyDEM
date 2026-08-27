import numpy as np

from dem_processing.design_corridor import _bank_from_lateral_terrain, _compound_discharge, _conditioned_floodplain_receivers, _connected_side_depths, _fill_terminal_river_directions, _floodplain_flow_accumulation, _receiver_to_d4_direction, _refinement_flow_envelope, _smoothed_profile


def test_terminal_river_direction_inherits_widest_upstream_axis() -> None:
    direction = np.array([[2.0, np.nan, 4.0]])
    river = np.ones_like(direction, dtype=bool)
    width = np.array([[20.0, 30.0, 50.0]])
    filled, count = _fill_terminal_river_directions(direction, river, width)
    assert count == 1
    assert filled.tolist() == [[2.0, 4.0, 4.0]]


def test_connected_side_wetting_stops_at_the_first_high_bank() -> None:
    depths = _connected_side_depths(np.array([99.0, 101.0, 98.0]), 100.0)
    assert np.allclose(depths, [1.0, 0.0, 0.0])


def test_missing_lateral_terrain_stops_connected_wetting() -> None:
    depths = _connected_side_depths(np.array([99.0, np.nan, 98.0]), 100.0)
    assert np.allclose(depths, [1.0, 0.0, 0.0])


def test_compound_conveyance_increases_with_stage() -> None:
    args = (0.0, 2.0, 30.0, 0.001, np.array([2.5, 3.0]), np.array([2.5, 3.0]), 30.0, 0.035, 0.05)
    assert _compound_discharge(2.5, *args) > _compound_discharge(1.5, *args)


def test_refinement_envelope_does_not_drop_after_a_confluence() -> None:
    peak = np.array([[10.0, 4.0, 6.0]])
    receiver = np.array([[1, 2, -1]], dtype=np.int64)
    active = np.ones_like(peak, dtype=bool)
    envelope = _refinement_flow_envelope(peak, receiver, active)
    assert envelope.tolist() == [[10.0, 10.0, 10.0]]


def test_robust_bank_ignores_one_low_lateral_pixel_and_smooths_between_stations() -> None:
    bank = _bank_from_lateral_terrain(np.array([90.0, 100.0, 100.0]), np.array([101.0, 101.0, 101.0]), 3)
    assert bank == 100.0
    profile = _smoothed_profile(np.array([100.0, 80.0, 100.0]), np.array([0.0, 300.0, 600.0]), 600.0)
    assert profile[1] == 100.0


def test_floodplain_drainage_is_conditioned_to_the_river_sink() -> None:
    terrain = np.array([[3.0, 0.0, 1.0]])
    active = np.ones_like(terrain, dtype=bool)
    river = np.array([[False, False, True]])
    potential, receiver = _conditioned_floodplain_receivers(terrain, active, river, terrain)
    assert potential[0, 1] > potential[0, 2]
    assert _receiver_to_d4_direction(receiver).tolist() == [[2, 2, 0]]
    assert _floodplain_flow_accumulation(receiver, active).tolist() == [[1.0, 2.0, 3.0]]
