"""MotionPredictor — wrapper tổng hợp VelocityBuffer + TrajectoryExtrapolator (Plan.md §2.3)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ...common.models import MotionPrediction, PredictedPoint, Track, Velocity
from .extrapolator import TrajectoryExtrapolator
from .perspective import PerspectiveTransform
from .velocity_buffer import VelocityBuffer

_ALERT_THRESHOLDS = {
    "high": 0.7,
    "medium": 0.4,
}

MAX_VELOCITY_PX_PER_FRAME = 500.0   # sanity limit — above this likely noise (≈500 px/frame at 30fps)
DIRECTION_CHANGE_THRESHOLD = math.pi / 2  # thay đổi 90° đột ngột → đáng ngờ


class MotionPredictor:
    """Tổng hợp VelocityBuffer + TrajectoryExtrapolator thành interface duy nhất.

    Nhận track, tính velocity/acceleration từ velocity_buffer (hoặc Kalman),
    dự đoán vị trí tương lai tại các mốc thời gian định sẵn, và trả về
    MotionPrediction kèm alert_level.
    """

    def __init__(
        self,
        prediction_horizons_s: Optional[List[float]] = None,
        fps: float = 30.0,
        alert_confidence_threshold: float = 0.6,
        perspective_transform: Optional[PerspectiveTransform] = None,
    ) -> None:
        self.prediction_horizons_s = prediction_horizons_s or [0.5, 1.0, 2.0]
        self.fps = fps
        self._max_velocity_px_per_s = MAX_VELOCITY_PX_PER_FRAME * self.fps  # 500 px/frame * fps
        self.alert_confidence_threshold = alert_confidence_threshold
        self._extrapolator = TrajectoryExtrapolator(
            perspective_transform=perspective_transform
        )
        # Cache vận tốc frame trước để kiểm tra tính nhất quán hướng di chuyển
        self._prev_velocity: Dict[int, tuple] = {}
        # Cache diện tích bbox frame trước để kiểm tra tính nhất quán kích thước
        self._prev_bbox_area: Dict[int, float] = {}

    def update(self, track: Track, timestamp: float) -> MotionPrediction:
        """Đẩy dữ liệu mới nhất và trả về MotionPrediction đầy đủ.

        Args:
            track:     Track cần dự đoán.
            timestamp: Thời gian hiện tại (seconds).

        Returns:
            MotionPrediction với trajectory, confidence và alert_level.
        """
        velocity, acceleration = self._get_kinematics(track)

        # Phạt confidence nếu chuyển động bất thường
        confidence_penalty = self._compute_motion_penalty(track, velocity)

        trajectory: List[PredictedPoint] = []
        if track.anchor_point is not None:
            for dt in self.prediction_horizons_s:
                pos = self._extrapolator.extrapolate(
                    current_position=track.anchor_point,
                    velocity=velocity,
                    acceleration=acceleration,
                    dt_seconds=dt,
                )
                raw_conf = self._extrapolator.compute_confidence(
                    track=track,
                    prediction_horizon_s=dt,
                    velocity_variance=self._velocity_variance(track),
                )
                conf = max(0.0, raw_conf - confidence_penalty)
                trajectory.append(
                    PredictedPoint(position=pos, timestamp_s=dt, confidence=conf)
                )

        overall_confidence = (
            sum(p.confidence for p in trajectory) / len(trajectory)
            if trajectory
            else 0.0
        )
        alert_level = self._compute_alert_level(track, overall_confidence)

        # Lưu vận tốc hiện tại để kiểm tra frame kế tiếp
        self._prev_velocity[track.track_id] = velocity

        return MotionPrediction(
            track_id=track.track_id,
            trajectory=trajectory,
            overall_confidence=overall_confidence,
            alert_level=alert_level,
            velocity_px_per_s=velocity,
            acceleration_px_per_s2=acceleration,
        )

    def predict_trajectory(self, track: Track) -> List[PredictedPoint]:
        """Trả về vị trí dự đoán tại mỗi horizon mà không cập nhật trạng thái.

        Dùng khi cần dự đoán read-only (ví dụ render).

        Lưu ý: confidence ở đây không áp dụng motion penalty (xem `update()` để có confidence đầy đủ).

        Args:
            track: Track cần dự đoán.

        Returns:
            Danh sách PredictedPoint tại các mốc prediction_horizons_s.
        """
        velocity, acceleration = self._get_kinematics(track)
        result: List[PredictedPoint] = []
        if track.anchor_point is not None:
            for dt in self.prediction_horizons_s:
                pos = self._extrapolator.extrapolate(
                    current_position=track.anchor_point,
                    velocity=velocity,
                    acceleration=acceleration,
                    dt_seconds=dt,
                )
                conf = self._extrapolator.compute_confidence(
                    track, dt, self._velocity_variance(track)
                )
                result.append(
                    PredictedPoint(position=pos, timestamp_s=dt, confidence=conf)
                )
        return result

    def should_alert(self, track: Track, roi_zone: str) -> bool:
        """Trả về True nếu track có nguy cơ cao trong blind spot zone được chỉ định.

        Args:
            track:    Track cần kiểm tra.
            roi_zone: Tên zone cần so khớp (ví dụ "blind_left").

        Returns:
            True nếu track đang trong zone và confidence ≥ alert_confidence_threshold.
        """
        if not track.in_roi or track.zone_name != roi_zone:
            return False
        velocity, _ = self._get_kinematics(track)
        conf = self._extrapolator.compute_confidence(
            track, self.prediction_horizons_s[0], self._velocity_variance(track)
        )
        return conf >= self.alert_confidence_threshold

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _get_kinematics(self, track: Track) -> tuple:
        """Lấy (velocity, acceleration) từ velocity_buffer hoặc Track.velocity.

        Ưu tiên velocity_buffer nếu đủ mẫu; fallback về Kalman velocity.
        """
        zero: Velocity = (0.0, 0.0)
        if track.velocity_buffer is not None and len(track.velocity_buffer) >= 2:
            velocity: Velocity = track.velocity_buffer.get_velocity() or track.velocity or zero
            acceleration: Velocity = getattr(track.velocity_buffer, "get_acceleration", lambda: None)() or zero
        else:
            velocity = track.velocity or zero
            acceleration = zero
        return velocity, acceleration

    def _velocity_variance(self, track: Track) -> float:
        """Ước lượng phương sai vận tốc từ velocity_buffer.

        Returns:
            Phương sai tổng (vx và vy) hoặc 0.0 nếu không đủ dữ liệu.
        """
        if track.velocity_buffer is None or len(track.velocity_buffer) < 3:
            return 0.0
        positions = list(track.velocity_buffer.positions)
        timestamps = list(track.velocity_buffer.timestamps)
        if len(positions) < 3:
            return 0.0

        # Ước lượng vận tốc từng frame → tính phương sai
        vx_list, vy_list = [], []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                continue
            vx_list.append((positions[i][0] - positions[i - 1][0]) / dt)
            vy_list.append((positions[i][1] - positions[i - 1][1]) / dt)
        if not vx_list:
            return 0.0
        return float(np.var(vx_list) + np.var(vy_list))

    def _compute_motion_penalty(self, track: Track, velocity: Velocity) -> float:
        """Tính phạt confidence nếu chuyển động bất thường.

        Kiểm tra:
        1. Tốc độ quá lớn (> _max_velocity_px_per_s = MAX_VELOCITY_PX_PER_FRAME * fps) → nhiễu.
        2. Thay đổi hướng đột ngột (> 90°) so với frame trước.
        3. Kích thước bbox thay đổi đột ngột (> 3x) → tái định danh đáng ngờ.

        Returns:
            Giá trị phạt trong khoảng [0.0, 0.5].
        """
        penalty = 0.0
        vx, vy = velocity
        speed = math.sqrt(vx ** 2 + vy ** 2)

        # Kiểm tra tốc độ bất thường
        if speed > self._max_velocity_px_per_s:
            penalty += 0.3

        # Kiểm tra tính nhất quán hướng di chuyển
        prev = self._prev_velocity.get(track.track_id)
        if prev is not None and speed > 1.0:
            prev_vx, prev_vy = prev
            prev_speed = math.sqrt(prev_vx ** 2 + prev_vy ** 2)
            if prev_speed > 1.0:
                dot = (vx * prev_vx + vy * prev_vy) / (speed * prev_speed)
                dot = max(-1.0, min(1.0, dot))
                angle_change = math.acos(dot)
                if angle_change > DIRECTION_CHANGE_THRESHOLD:
                    penalty += 0.2

        # Kiểm tra tính nhất quán kích thước bbox
        x1, y1, x2, y2 = track.bbox
        curr_area = max(0.0, (x2 - x1) * (y2 - y1))
        prev_area = self._prev_bbox_area.get(track.track_id)
        if prev_area is not None and prev_area > 0:
            ratio = curr_area / prev_area if curr_area > prev_area else prev_area / curr_area
            if ratio > 3.0:  # diện tích thay đổi > 3x → tái định danh đáng ngờ
                penalty += 0.15
        self._prev_bbox_area[track.track_id] = curr_area

        return min(penalty, 0.5)  # giới hạn tối đa 0.5 để không triệt tiêu hoàn toàn confidence

    def _compute_alert_level(self, track: Track, confidence: float) -> str:
        """Xác định mức cảnh báo dựa trên confidence và trạng thái ROI.

        Returns:
            "none" | "low" | "medium" | "high"
        """
        if not track.in_roi:
            return "none"
        if confidence >= _ALERT_THRESHOLDS["high"]:
            return "high"
        if confidence >= _ALERT_THRESHOLDS["medium"]:
            return "medium"
        if confidence > 0.0:
            return "low"
        return "none"


__all__ = ["MotionPredictor"]
