from __future__ import annotations

import pytest

from src.tracking.kalman_filter import BoundingBoxKalmanFilter


def test_get_velocity_requires_initialization() -> None:
    kf = BoundingBoxKalmanFilter()

    with pytest.raises(RuntimeError):
        kf.get_velocity()


def test_get_velocity_returns_state_velocity_components() -> None:
    kf = BoundingBoxKalmanFilter()
    kf.initiate((100.0, 100.0, 200.0, 200.0))
    kf.predict()
    kf.update((110.0, 100.0, 210.0, 200.0))

    expected_velocity = (float(kf.state[4, 0]), float(kf.state[5, 0]))

    assert kf.get_velocity() == pytest.approx(expected_velocity)
