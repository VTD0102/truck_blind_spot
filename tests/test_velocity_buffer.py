"""Unit tests cho `VelocityBuffer` (Plan.md §2.1)."""
from __future__ import annotations

import math

import pytest

from src.tracking.motion.velocity_buffer import VelocityBuffer


def test_happy_constant_velocity() -> None:
    buf = VelocityBuffer(max_size=10)
    # Vận tốc không đổi: vx = 10 px/s, vy = 5 px/s
    for i in range(5):
        buf.push((100 + 10 * i, 200 + 5 * i), timestamp=float(i))

    velocity = buf.get_velocity()
    assert velocity is not None
    vx, vy = velocity
    assert math.isclose(vx, 10.0, abs_tol=1e-6)
    assert math.isclose(vy, 5.0, abs_tol=1e-6)


def test_empty_buffer_returns_none() -> None:
    buf = VelocityBuffer()
    assert buf.get_velocity() is None
    assert buf.get_acceleration() is None
    assert len(buf) == 0


def test_single_point_returns_none() -> None:
    buf = VelocityBuffer()
    buf.push((100, 200), 0.0)
    assert buf.get_velocity() is None
    assert buf.get_acceleration() is None


def test_two_points_enough_for_velocity_not_acceleration() -> None:
    buf = VelocityBuffer()
    buf.push((0, 0), 0.0)
    buf.push((20, 10), 1.0)
    velocity = buf.get_velocity()
    assert velocity is not None
    assert math.isclose(velocity[0], 20.0, abs_tol=1e-6)
    assert buf.get_acceleration() is None


def test_non_monotonic_timestamp_raises() -> None:
    buf = VelocityBuffer()
    buf.push((0, 0), 1.0)
    with pytest.raises(ValueError, match="monotonic"):
        buf.push((10, 10), 0.5)


def test_buffer_respects_max_size() -> None:
    buf = VelocityBuffer(max_size=3)
    for i in range(5):
        buf.push((i, i), float(i))
    assert len(buf) == 3
    # Entries cũ nhất bị loại bỏ: timestamps = [2.0, 3.0, 4.0]
    assert list(buf.timestamps) == [2.0, 3.0, 4.0]


def test_acceleration_constant_uniform() -> None:
    buf = VelocityBuffer(max_size=10)
    # x(t) = 0.5 * 4 * t^2             → ax = 4
    # y(t) = 2*t + 0.5 * 2 * t^2       → ay = 2
    for i in range(6):
        t = float(i)
        x = 0.5 * 4.0 * t * t
        y = 2.0 * t + 0.5 * 2.0 * t * t
        buf.push((int(round(x)), int(round(y))), t)

    acc = buf.get_acceleration()
    assert acc is not None
    ax, ay = acc
    assert math.isclose(ax, 4.0, abs_tol=0.5)
    assert math.isclose(ay, 2.0, abs_tol=0.5)


def test_smoothed_velocity_uses_recent_window() -> None:
    buf = VelocityBuffer(max_size=10)
    # 5 frames đầu: đứng yên
    for i in range(5):
        buf.push((0, 0), float(i))
    # 3 frames cuối: chuyển động với vx ~ 10 px/s
    for i in range(5, 8):
        buf.push((10 * (i - 4), 0), float(i))

    smoothed = buf.get_smoothed_velocity(window=3)
    assert smoothed is not None
    vx, _vy = smoothed
    assert vx > 5.0  # đã bắt được chuyển động gần đây

    overall = buf.get_velocity()
    assert overall is not None
    assert overall[0] < vx  # ước lượng toàn buffer bị kéo xuống


def test_smoothed_window_below_minimum_raises() -> None:
    buf = VelocityBuffer()
    with pytest.raises(ValueError, match="window"):
        buf.get_smoothed_velocity(window=1)


def test_duplicate_timestamps_return_none() -> None:
    buf = VelocityBuffer()
    buf.push((0, 0), 1.0)
    buf.push((10, 10), 1.0)
    # ts[0] == ts[-1] → không thể tính slope
    assert buf.get_velocity() is None


def test_max_size_validation() -> None:
    # max_size=2 < min_points_for_acceleration (default 3)
    with pytest.raises(ValueError, match="max_size"):
        VelocityBuffer(max_size=2)


def test_clear_resets_state() -> None:
    buf = VelocityBuffer()
    buf.push((0, 0), 0.0)
    buf.push((10, 10), 1.0)
    buf.clear()
    assert len(buf) == 0
    assert buf.get_velocity() is None
