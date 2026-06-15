"""Unit tests cho `TrajectoryExtrapolator` (Plan.md §2.2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.common.models import Track
from src.tracking.motion.extrapolator import TrajectoryExtrapolator
from src.tracking.motion.perspective import PerspectiveTransform


def _make_track(hits: int, misses: int) -> Track:
    return Track(
        track_id=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
        hits=hits,
        misses=misses,
    )


def test_extrapolate_zero_velocity_returns_input() -> None:
    ex = TrajectoryExtrapolator()
    p = ex.extrapolate(
        (100, 200),
        velocity=(0.0, 0.0),
        acceleration=(0.0, 0.0),
        dt_seconds=1.5,
    )
    assert p == (100, 200)


def test_extrapolate_constant_velocity() -> None:
    ex = TrajectoryExtrapolator()
    p = ex.extrapolate(
        (0, 0),
        velocity=(10.0, 5.0),
        acceleration=(0.0, 0.0),
        dt_seconds=2.0,
    )
    assert p == (20, 10)


def test_extrapolate_with_acceleration() -> None:
    ex = TrajectoryExtrapolator()
    # x = 0 + 10*2 + 0.5*4*4 = 28
    # y = 0 + 0*2  + 0.5*-2*4 = -4
    p = ex.extrapolate(
        (0, 0),
        velocity=(10.0, 0.0),
        acceleration=(4.0, -2.0),
        dt_seconds=2.0,
    )
    assert p == (28, -4)


def test_extrapolate_negative_dt_raises() -> None:
    ex = TrajectoryExtrapolator()
    with pytest.raises(ValueError, match="dt_seconds"):
        ex.extrapolate(
            (0, 0),
            velocity=(1.0, 1.0),
            acceleration=(0.0, 0.0),
            dt_seconds=-0.1,
        )


def test_extrapolate_with_perspective_transform() -> None:
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
    ex = TrajectoryExtrapolator(perspective_transform=transform)
    p = ex.extrapolate(
        (10, 20),
        velocity=(4.0, 0.0),      # BEV units / second
        acceleration=(0.0, 0.0),  # BEV units / second^2
        dt_seconds=1.0,
    )
    # pixel(10,20) -> bev(20,40) -> bev'(24,40) -> pixel(12,20)
    assert p == (12, 20)


def test_confidence_perfect_track() -> None:
    ex = TrajectoryExtrapolator(min_hits_threshold=3, decay_lambda=0.0)
    track = _make_track(hits=20, misses=0)
    c = ex.compute_confidence(
        track, prediction_horizon_s=0.0, velocity_variance=0.0
    )
    # tracking_quality = 1, detection_consistency = 1, motion_smoothness = 1
    assert math.isclose(c, 1.0, abs_tol=1e-6)


def test_confidence_decays_with_horizon() -> None:
    ex = TrajectoryExtrapolator(decay_lambda=1.0)
    track = _make_track(hits=20, misses=0)
    c_short = ex.compute_confidence(track, prediction_horizon_s=0.5)
    c_long = ex.compute_confidence(track, prediction_horizon_s=2.0)
    assert c_short > c_long > 0.0


def test_confidence_tracks_misses() -> None:
    ex = TrajectoryExtrapolator(decay_lambda=0.0)
    track_good = _make_track(hits=10, misses=0)
    track_bad = _make_track(hits=10, misses=10)
    c_good = ex.compute_confidence(track_good, prediction_horizon_s=0.0)
    c_bad = ex.compute_confidence(track_bad, prediction_horizon_s=0.0)
    assert c_good > c_bad


def test_confidence_empty_track_stats() -> None:
    ex = TrajectoryExtrapolator(decay_lambda=0.0)
    track = _make_track(hits=0, misses=0)
    c = ex.compute_confidence(track, prediction_horizon_s=0.0)
    # total_frames = 0 → không crash; confidence vẫn trong [0, 1]
    assert 0.0 <= c <= 1.0


def test_confidence_motion_variance_reduces_smoothness() -> None:
    ex = TrajectoryExtrapolator(decay_lambda=0.0)
    track = _make_track(hits=20, misses=0)
    c_smooth = ex.compute_confidence(
        track, prediction_horizon_s=0.0, velocity_variance=0.0
    )
    c_noisy = ex.compute_confidence(
        track, prediction_horizon_s=0.0, velocity_variance=100.0
    )
    assert c_smooth > c_noisy


def test_confidence_negative_inputs_raise() -> None:
    ex = TrajectoryExtrapolator()
    track = _make_track(hits=5, misses=0)
    with pytest.raises(ValueError):
        ex.compute_confidence(track, prediction_horizon_s=-1.0)
    with pytest.raises(ValueError):
        ex.compute_confidence(
            track, prediction_horizon_s=0.5, velocity_variance=-1.0
        )


def test_bad_weights_raise() -> None:
    with pytest.raises(ValueError, match="weight"):
        TrajectoryExtrapolator(
            tracking_weight=0.5,
            consistency_weight=0.5,
            smoothness_weight=0.5,
        )


def test_bad_min_hits_raises() -> None:
    with pytest.raises(ValueError, match="min_hits_threshold"):
        TrajectoryExtrapolator(min_hits_threshold=0)


def test_bad_decay_lambda_raises() -> None:
    with pytest.raises(ValueError, match="decay_lambda"):
        TrajectoryExtrapolator(decay_lambda=-0.1)
