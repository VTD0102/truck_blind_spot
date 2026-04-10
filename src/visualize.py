from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .detector import Detection
    from .roi import ROIZone
except ImportError:
    from detector import Detection
    from roi import ROIZone


class BlindSpotVisualizer:
    """Lớp dùng để vẽ và hiển thị bounding box, ROI và cảnh báo điểm mù."""
    def __init__(
        self,
        normal_color: Tuple[int, int, int] = (0, 200, 0),
        warning_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        font_scale: float = 0.6,
        roi_alpha: float = 0.18,
    ) -> None:
        self.normal_color = normal_color
        self.warning_color = warning_color
        self.thickness = thickness
        self.font_scale = font_scale
        self.roi_alpha = roi_alpha

    def draw(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
        roi_zones: Optional[Sequence[ROIZone]] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Vẽ biểu diễn hình ảnh lên frame."""
        output = frame.copy() if copy else frame

        if roi_zones:
            self._draw_roi_zones(output, roi_zones)

        warning_count = 0
        # Vẽ các bounding box phát hiện được
        for detection in detections:
            in_roi = getattr(detection, "in_roi", False)
            zone_name = getattr(detection, "zone_name", None)
            risk_level = getattr(detection, "risk_level", None)

            color = self._resolve_detection_color(detection, roi_zones)
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            if in_roi:
                warning_count += 1
                prefix = zone_name if zone_name else "ROI"
                if risk_level:
                    prefix = f"{prefix} [{risk_level}]"
                label = f"{prefix} | {label}"

            self._draw_label(output, label, (x1, max(24, y1 - 10)), color)

            # Vẽ điểm tham chiếu của đối tượng (dùng kiểm tra xem có vào ROI không)
            if detection.anchor_point is not None:
                cv2.circle(output, detection.anchor_point, 5, color, -1)

        # Hiển thị số lượng cảnh báo điểm mù
        if warning_count > 0:
            self._draw_banner(output, f"CANH BAO DIEM MU: {warning_count}")

        return output

    def _draw_roi_zones(self, frame: np.ndarray, roi_zones: Sequence[ROIZone]) -> None:
        overlay = frame.copy()

        for zone in roi_zones:
            polygon = np.array(zone.points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], zone.color)

        cv2.addWeighted(overlay, self.roi_alpha, frame, 1.0 - self.roi_alpha, 0, frame)

        for zone in roi_zones:
            polygon = np.array(zone.points, dtype=np.int32)
            cv2.polylines(frame, [polygon], True, zone.color, 2)

            anchor = polygon[0]
            label_pos = (int(anchor[0]), max(24, int(anchor[1]) - 10))
            text = f"{zone.name} [{zone.risk_level}]"
            self._draw_label(frame, text, label_pos, zone.color)

    def _resolve_detection_color(
        self,
        detection: Detection,
        roi_zones: Optional[Sequence[ROIZone]],
    ) -> Tuple[int, int, int]:
        if not detection.in_roi or not detection.zone_name or not roi_zones:
            return self.normal_color if not detection.in_roi else self.warning_color

        for zone in roi_zones:
            if zone.name == detection.zone_name:
                return zone.color
        return self.warning_color

    def _draw_banner(self, frame: np.ndarray, text: str) -> None:
        """Vẽ banner cảnh báo ở góc trên cùng của khung hình."""
        text_size, _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale + 0.1, 2
        )
        width = text_size[0] + 20
        height = text_size[1] + 20
        cv2.rectangle(frame, (10, 10), (10 + width, 10 + height), self.warning_color, -1)
        cv2.putText(
            frame,
            text,
            (20, 10 + height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale + 0.1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        """Vẽ background và chữ cho label."""
        x, y = origin
        text_size, baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
        )
        top_left = (x, max(0, y - text_size[1] - baseline - 6))
        bottom_right = (x + text_size[0] + 10, y + 4)
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.putText(
            frame,
            text,
            (x + 5, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
