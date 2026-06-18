"""Dự đoán vị trí tương lai và confidence cho track (Plan.md §2.2)."""
from __future__ import annotations

import math
from typing import Optional

from ..types import Point, Track, Velocity
from .perspective import PerspectiveTransform


class TrajectoryExtrapolator:
    """Extrapolate vị trí và tính confidence cho một track.

    Mô hình động học bậc 2:
        x(t) = x0 + vx*t + 0.5 * ax * t^2
        y(t) = y0 + vy*t + 0.5 * ay * t^2

    Confidence ∈ [0, 1] tổng hợp từ 3 thành phần (weight sum = 1.0):
        - tracking_quality      = hits / (hits + misses)
        - detection_consistency = min(hits, min_hits_threshold) / min_hits_threshold
        - motion_smoothness     = 1 / (1 + velocity_variance)

    Sau đó nhân với time-decay ``exp(-decay_lambda * prediction_horizon_s)``
    để phản ánh độ không chắc chắn tăng theo khoảng dự đoán.

    Quy ước đơn vị:
        - velocity             : pixels / second
        - acceleration         : pixels / second^2
        - prediction_horizon_s : seconds (≥ 0)
    """

    def __init__(
        self,
        perspective_transform: Optional[PerspectiveTransform] = None,
        min_hits_threshold: int = 3,
        tracking_weight: float = 0.4,
        consistency_weight: float = 0.3,
        smoothness_weight: float = 0.3,
        decay_lambda: float = 0.5,
    ) -> None:
        if min_hits_threshold < 1:
            raise ValueError(
                "min_hits_threshold phải ≥ 1 "
                f"(nhận được {min_hits_threshold})."
            )
        weight_sum = tracking_weight + consistency_weight + smoothness_weight
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                "Tổng các weight phải = 1.0 "
                f"(nhận được tracking={tracking_weight}, "
                f"consistency={consistency_weight}, "
                f"smoothness={smoothness_weight}, sum={weight_sum:.4f})."
            )
        if decay_lambda < 0.0:
            raise ValueError(
                f"decay_lambda phải không âm (nhận được {decay_lambda})."
            )

        self.min_hits_threshold = int(min_hits_threshold)
        self.tracking_weight = float(tracking_weight)
        self.consistency_weight = float(consistency_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.decay_lambda = float(decay_lambda)
        self.perspective_transform = perspective_transform

    def extrapolate(
        self,
        current_position: Point,
        velocity: Velocity,
        acceleration: Velocity,
        dt_seconds: float,
    ) -> Point:
        """Trả về vị trí dự đoán sau ``dt_seconds`` giây.

        Args:
            current_position: Vị trí hiện tại ``(x, y)``.
                - Không có transform: đơn vị pixel.
                - Có transform: đầu vào pixel, sẽ được warp sang BEV.
            velocity:         ``(vx, vy)`` theo cùng không gian dự đoán.
            acceleration:     ``(ax, ay)`` theo cùng không gian dự đoán.
            dt_seconds:       Khoảng thời gian dự đoán phía trước (≥ 0).

        Returns:
            Vị trí dự đoán ``(x, y)`` dạng pixel đã làm tròn về int.

        Raises:
            ValueError: Nếu ``dt_seconds`` âm.
        """
        if dt_seconds < 0.0:
            raise ValueError(
                f"dt_seconds phải không âm (nhận được {dt_seconds})."
            )

        # Nếu có homography: warp vị trí pixel sang không gian BEV trước khi
        # ngoại suy (chuyển động trên mặt phẳng thật tuyến tính hơn, dự đoán
        # chính xác hơn). Không có transform thì ngoại suy thẳng trong pixel.
        if self.perspective_transform is not None:
            x0, y0 = self.perspective_transform.pixel_to_bev(current_position)
        else:
            x0, y0 = current_position
        vx, vy = velocity
        ax, ay = acceleration

        # Phương trình động học bậc 2: x(t) = x0 + v*t + 0.5*a*t^2 (tách riêng x, y).
        dt2_half = 0.5 * dt_seconds * dt_seconds
        x = x0 + vx * dt_seconds + ax * dt2_half
        y = y0 + vy * dt_seconds + ay * dt2_half

        # Có transform thì warp ngược điểm BEV vừa dự đoán về pixel để vẽ;
        # không thì làm tròn về int (tọa độ pixel).
        if self.perspective_transform is not None:
            return self.perspective_transform.bev_to_pixel((x, y))
        return (int(round(x)), int(round(y)))

    def compute_confidence(
        self,
        track: Track,
        prediction_horizon_s: float,
        velocity_variance: float = 0.0,
    ) -> float:
        """Tính confidence ∈ [0, 1] cho dự đoán của ``track``.

        Args:
            track:                Track đang đánh giá (dùng ``hits`` & ``misses``).
            prediction_horizon_s: Khoảng dự đoán phía trước, giây (≥ 0).
            velocity_variance:    Phương sai vận tốc (pixels^2/second^2, ≥ 0).
                                  Mặc định 0 → motion_smoothness = 1.0 (trung tính)
                                  khi chưa có tích hợp với ``VelocityBuffer``.

        Returns:
            Giá trị confidence trong khoảng [0, 1].

        Raises:
            ValueError: Nếu ``prediction_horizon_s`` hoặc ``velocity_variance`` âm.
        """
        if prediction_horizon_s < 0.0:
            raise ValueError(
                "prediction_horizon_s phải không âm "
                f"(nhận được {prediction_horizon_s})."
            )
        if velocity_variance < 0.0:
            raise ValueError(
                "velocity_variance phải không âm "
                f"(nhận được {velocity_variance})."
            )

        # (1) tracking_quality: tỉ lệ frame khớp được detection trên tổng số frame
        #     track tồn tại. Khớp càng đều → track càng đáng tin.
        total_frames = track.hits + track.misses
        tracking_quality = (
            track.hits / total_frames if total_frames > 0 else 0.0
        )
        # (2) detection_consistency: track đã được phát hiện đủ "chín" hay chưa,
        #     bão hòa về 1.0 khi hits đạt ngưỡng min_hits_threshold.
        detection_consistency = (
            min(track.hits, self.min_hits_threshold) / self.min_hits_threshold
        )
        # (3) motion_smoothness: chuyển động càng mượt (phương sai vận tốc thấp)
        #     thì điểm càng gần 1.0; nhiễu nhiều → giảm về 0.
        motion_smoothness = 1.0 / (1.0 + velocity_variance)

        # Tổ hợp tuyến tính 3 thành phần theo trọng số (tổng trọng số = 1.0).
        base_confidence = (
            self.tracking_weight * tracking_quality
            + self.consistency_weight * detection_consistency
            + self.smoothness_weight * motion_smoothness
        )
        # Nhân hệ số suy giảm theo thời gian: dự đoán càng xa (dt lớn) càng kém
        # chắc chắn → exp(-lambda * dt). Cuối cùng kẹp về [0, 1].
        decay = math.exp(-self.decay_lambda * prediction_horizon_s)
        return max(0.0, min(1.0, base_confidence * decay))


__all__ = ["TrajectoryExtrapolator"]
