"""Unit tests cho `PerspectiveTransform` (Plan.md §2.0)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.tracking.motion.perspective import PerspectiveTransform


def test_pixel_to_bev_with_identity_homography() -> None:
    transform = PerspectiveTransform(np.eye(3, dtype=np.float64))
    assert transform.pixel_to_bev((120, 340)) == (120.0, 340.0)


def test_bev_to_pixel_uses_inverse_homography() -> None:
    transform = PerspectiveTransform(
        np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    # pixel -> bev scale x2, inverse phải chia 2
    assert transform.bev_to_pixel((240.0, 100.0)) == (120, 50)


def test_estimate_distance_matches_pinhole_formula() -> None:
    distance = PerspectiveTransform.estimate_distance(
        bbox_height_px=100.0,
        focal_length=800.0,
        real_height_m=1.7,
    )
    assert math.isclose(distance, 13.6, abs_tol=1e-6)


def test_invalid_homography_shape_raises() -> None:
    with pytest.raises(ValueError, match="3x3"):
        PerspectiveTransform(np.ones((2, 2), dtype=np.float64))


def test_estimate_distance_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="bbox_height_px"):
        PerspectiveTransform.estimate_distance(
            bbox_height_px=0.0,
            focal_length=800.0,
            real_height_m=1.7,
        )
