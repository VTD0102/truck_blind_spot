"""Unit tests for MotionPredictor (Plan.md §2.4)."""
from __future__ import annotations

import copy

import pytest

from src.common.enums import TrackStatus
from src.common.models import MotionPrediction, PredictedPoint, Track
from src.tracking.motion.predictor import MotionPredictor
from src.tracking.motion.velocity_buffer import VelocityBuffer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FPS = 30.0
FRAME_DT = 1.0 / FPS


def make_track(
    track_id: int = 1,
    hits: int = 5,
    misses: int = 0,
    bbox: tuple = (100, 100, 200, 200),
    in_roi: bool = True,
    zone_name: str = "danger",
) -> Track:
    return Track(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name="person",
        hits=hits,
        misses=misses,
        is_confirmed=True,
        status=TrackStatus.CONFIRMED,
        in_roi=in_roi,
        zone_name=zone_name,
        risk_level="high",
    )


def _fill_buffer_constant_velocity(
    track: Track,
    start_pos: tuple,
    dx: float,
    dy: float,
    n_frames: int = 10,
) -> None:
    """Đổ n_frames điểm với vận tốc hằng (dx, dy) mỗi frame vào track."""
    buf = VelocityBuffer()
    for i in range(n_frames):
        x = int(start_pos[0] + i * dx)
        y = int(start_pos[1] + i * dy)
        buf.push((x, y), i * FRAME_DT)
    track.velocity_buffer = buf
    # anchor_point tại frame cuối
    track.anchor_point = (
        int(start_pos[0] + (n_frames - 1) * dx),
        int(start_pos[1] + (n_frames - 1) * dy),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_constant_velocity_linear_extrapolation() -> None:
    """0.5s tại 30fps = 15 frames, dx=5 px/frame → ~75 px tiến."""
    predictor = MotionPredictor(prediction_horizons_s=[0.5, 1.0, 2.0], fps=FPS)
    track = make_track()
    # dx = 5 px/frame = 150 px/s; vận tốc hằng
    _fill_buffer_constant_velocity(track, start_pos=(100, 200), dx=5, dy=0)

    result = predictor.update(track, timestamp=10 * FRAME_DT)

    assert len(result.trajectory) == 3
    # trajectory[0] @ t=0.5s: anchor_x + 150*0.5 ≈ 145 + 75 = 220
    pred_x = result.trajectory[0].position[0]
    anchor_x = track.anchor_point[0]  # type: ignore[index]
    expected_x = anchor_x + 75
    assert abs(pred_x - expected_x) <= 10, (
        f"expected ~{expected_x}, got {pred_x}"
    )


def test_accelerating_motion_quadratic_extrapolation() -> None:
    """Chuyển động gia tốc phải cho x tại t=2s lớn hơn ước tính tuyến tính."""
    predictor = MotionPredictor(prediction_horizons_s=[0.5, 1.0, 2.0], fps=FPS)
    track = make_track()

    # x(t) = x0 + v0*t + 0.5*a*t^2, a=100 px/s^2, v0=50 px/s
    a = 100.0  # px/s^2
    v0 = 50.0  # px/s
    x0 = 100
    n = 10
    buf = VelocityBuffer()
    for i in range(n):
        t = i * FRAME_DT
        x = x0 + v0 * t + 0.5 * a * t * t
        buf.push((int(x), 200), t)
    track.velocity_buffer = buf
    t_last = (n - 1) * FRAME_DT
    anchor_x = int(x0 + v0 * t_last + 0.5 * a * t_last * t_last)
    track.anchor_point = (anchor_x, 200)

    result = predictor.update(track, timestamp=t_last)

    # trajectory[-1] @ t=2s
    pred_x_2s = result.trajectory[-1].position[0]
    # ước tính tuyến tính (chỉ vận tốc, không gia tốc) tại t=2s
    vel = track.velocity_buffer.get_velocity()
    assert vel is not None
    linear_x = anchor_x + vel[0] * 2.0
    # dự đoán bậc 2 phải vượt trội ít nhất 200 px so với tuyến tính
    assert pred_x_2s >= linear_x + 200, (
        f"quadratic pred_x={pred_x_2s}, linear={linear_x}"
    )


def test_confidence_decay_with_horizon() -> None:
    """Confidence tại horizon gần phải cao hơn horizon xa."""
    predictor = MotionPredictor(prediction_horizons_s=[0.5, 1.0, 2.0], fps=FPS)
    track = make_track(hits=10, misses=0)
    _fill_buffer_constant_velocity(track, (100, 100), dx=3, dy=0)

    result = predictor.update(track, timestamp=9 * FRAME_DT)

    c0 = result.trajectory[0].confidence  # 0.5s
    c1 = result.trajectory[1].confidence  # 1.0s
    c2 = result.trajectory[2].confidence  # 2.0s
    assert c0 > c1 > c2, f"expected c0 > c1 > c2, got {c0:.3f} {c1:.3f} {c2:.3f}"


def test_alert_level_none_when_not_in_roi() -> None:
    """Track ngoài ROI phải có alert_level='none'."""
    predictor = MotionPredictor(fps=FPS)
    track = make_track(in_roi=False)
    _fill_buffer_constant_velocity(track, (50, 50), dx=2, dy=0)

    result = predictor.update(track, timestamp=9 * FRAME_DT)
    assert result.alert_level == "none"


def test_alert_level_high_for_confirmed_track_in_roi() -> None:
    """Track xác nhận tốt trong ROI phải có alert_level medium hoặc high."""
    predictor = MotionPredictor(fps=FPS)
    track = make_track(hits=10, misses=0, in_roi=True)
    _fill_buffer_constant_velocity(track, (100, 100), dx=2, dy=0, n_frames=8)

    result = predictor.update(track, timestamp=7 * FRAME_DT)
    assert result.alert_level in ("medium", "high"), (
        f"expected medium or high, got '{result.alert_level}'"
    )


def test_should_alert_returns_false_for_wrong_zone() -> None:
    """Track trong zone 'side_left' không kích hoạt alert cho zone 'side_right'."""
    predictor = MotionPredictor(fps=FPS)
    track = make_track(in_roi=True, zone_name="side_left")
    _fill_buffer_constant_velocity(track, (100, 100), dx=2, dy=0)

    assert predictor.should_alert(track, "side_right") is False


def test_predict_trajectory_no_state_mutation() -> None:
    """predict_trajectory() không được thay đổi trạng thái _prev_velocity/_prev_bbox_area."""
    predictor = MotionPredictor(fps=FPS)
    track = make_track()
    _fill_buffer_constant_velocity(track, (100, 100), dx=3, dy=0)

    prev_vel_before = copy.deepcopy(predictor._prev_velocity)
    prev_area_before = copy.deepcopy(predictor._prev_bbox_area)

    traj1 = predictor.predict_trajectory(track)
    traj2 = predictor.predict_trajectory(track)

    # Không có thay đổi state
    assert predictor._prev_velocity == prev_vel_before
    assert predictor._prev_bbox_area == prev_area_before

    # Hai lần gọi phải cho cùng số điểm
    assert len(traj1) == len(traj2)


def test_update_returns_motion_prediction_with_correct_structure() -> None:
    """Smoke test: update() trả về MotionPrediction hợp lệ."""
    horizons = [0.5, 1.0, 2.0]
    predictor = MotionPredictor(prediction_horizons_s=horizons, fps=FPS)
    track = make_track()
    _fill_buffer_constant_velocity(track, (100, 100), dx=2, dy=0)

    result = predictor.update(track, timestamp=9 * FRAME_DT)

    assert isinstance(result, MotionPrediction)
    assert result.track_id == track.track_id
    assert len(result.trajectory) == 3
    for point, expected_ts in zip(result.trajectory, horizons):
        assert isinstance(point, PredictedPoint)
        assert point.timestamp_s == pytest.approx(expected_ts)
        assert 0.0 <= point.confidence <= 1.0


def test_motion_validation_high_speed_reduces_confidence() -> None:
    """Vận tốc bất thường phải bị phạt confidence."""
    predictor_normal = MotionPredictor(fps=FPS)
    predictor_fast = MotionPredictor(fps=FPS)

    # Track bình thường: dx=5 px/frame = 150 px/s (an toàn)
    track_normal = make_track(track_id=1)
    _fill_buffer_constant_velocity(track_normal, (100, 100), dx=5, dy=0)

    # Track tốc độ cao: dx=600 px/frame → vượt ngưỡng MAX_VELOCITY_PX_PER_FRAME=500
    track_fast = make_track(track_id=2)
    buf_fast = VelocityBuffer()
    for i in range(10):
        buf_fast.push((100 + i * 600, 100), i * FRAME_DT)
    track_fast.velocity_buffer = buf_fast
    track_fast.anchor_point = (100 + 9 * 600, 100)

    result_normal = predictor_normal.update(track_normal, timestamp=9 * FRAME_DT)
    result_fast = predictor_fast.update(track_fast, timestamp=9 * FRAME_DT)

    assert result_normal.overall_confidence > result_fast.overall_confidence, (
        f"normal={result_normal.overall_confidence:.3f} "
        f"fast={result_fast.overall_confidence:.3f}"
    )


def test_no_anchor_point_yields_empty_trajectory() -> None:
    """Track không có anchor_point phải trả về trajectory rỗng."""
    predictor = MotionPredictor(fps=FPS)
    track = make_track()
    # anchor_point mặc định là None — không set velocity_buffer
    assert track.anchor_point is None

    result = predictor.update(track, timestamp=0.0)
    assert result.trajectory == []
