"""Unit tests for the pure, CPU-only helper functions in
distributed_inference.py -- calibrate() and majority_filter(). The GPU
actor (TileClassifier) itself needs a real model checkpoint and CUDA, so
it's out of scope for CI; these two functions are the parts that are both
deterministic and fast enough to test on every commit.
"""

import numpy as np

from distributed_inference import REFLECTANCE_MAX, calibrate, majority_filter


def test_calibrate_maps_zero_to_zero():
    tile = np.zeros((2, 2, 3), dtype=np.uint16)
    assert np.all(calibrate(tile) == 0.0)


def test_calibrate_scales_linearly_below_max():
    tile = np.array([[[REFLECTANCE_MAX // 2] * 3]], dtype=np.uint16)
    result = calibrate(tile)
    assert np.allclose(result, 0.5, atol=1e-3)


def test_calibrate_clips_above_max():
    # Values well past REFLECTANCE_MAX (e.g. a bright rooftop or cloud edge)
    # must saturate at 1.0, not overflow or wrap.
    tile = np.array([[[REFLECTANCE_MAX * 3] * 3]], dtype=np.uint32)
    result = calibrate(tile)
    assert np.all(result == 1.0)


def test_calibrate_output_always_in_unit_range():
    rng = np.random.default_rng(42)
    tile = rng.integers(0, 65535, size=(64, 64, 3)).astype(np.uint16)
    result = calibrate(tile)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_majority_filter_smooths_isolated_noise():
    # A single "River" tile surrounded on all sides by "Forest" should get
    # relabeled to match its neighborhood.
    grid = np.array(
        [
            ["Forest", "Forest", "Forest"],
            ["Forest", "River", "Forest"],
            ["Forest", "Forest", "Forest"],
        ],
        dtype=object,
    )
    smoothed = majority_filter(grid, size=3, center_weight=1)
    assert smoothed[1, 1] == "Forest"
    # Untouched tiles stay untouched.
    assert smoothed[0, 0] == "Forest"


def test_majority_filter_center_weight_resists_narrow_majority():
    # Center tile is "Forest"; among its 8 neighbors, 5 say "River" and 3
    # say "Forest" -- a narrow neighbor majority.
    grid = np.array(
        [
            ["Forest", "River", "River"],
            ["River", "Forest", "River"],
            ["Forest", "Forest", "River"],
        ],
        dtype=object,
    )

    # center_weight=1: the center's single vote isn't enough to overcome
    # the 5-4 neighbor majority -- it gets overridden.
    assert majority_filter(grid, size=3, center_weight=1)[1, 1] == "River"

    # center_weight=3 (the actual default): the center's own vote counts
    # 3x, tipping the same neighborhood back in its favor -- this is the
    # documented "less aggressive smoothing" behavior.
    assert majority_filter(grid, size=3, center_weight=3)[1, 1] == "Forest"


def test_majority_filter_preserves_grid_shape():
    rng = np.random.default_rng(0)
    classes = np.array(["Forest", "River", "AnnualCrop"], dtype=object)
    grid = rng.choice(classes, size=(10, 12))
    smoothed = majority_filter(grid)
    assert smoothed.shape == grid.shape
