from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .detector import Detection
except ImportError:
    from detector import Detection


class BlindSpotVisualizer:
    def __init__(
        self,
        normal_color: Tuple[int, int, int] = (0, 200, 0),
        warning_color: Tuple[int, int, int] = (0, 0, 255),
        roi_color: Tuple[int, int, int] = (0, 165, 255),
        thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self.normal_color = normal_color
        self.warning_color = warning_color
        self.roi_color = roi_color
        self.thickness = thickness
        self.font_scale = font_scale

    def draw(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
        roi_polygon: Optional[Sequence[Tuple[int, int]]] = None,
        copy: bool = True,
    ) -> np.ndarray:
        output = frame.copy() if copy else frame

        polygon = self._normalize_polygon(roi_polygon)
        if polygon is not None:
            cv2.polylines(output, [polygon], True, self.roi_color, 2)
            self._draw_label(output, "ROI", (polygon[0][0], polygon[0][1] - 10), self.roi_color)

        warning_count = 0
        for detection in detections:
            color = self.warning_color if detection.in_roi else self.normal_color
            x1, y1, x2, y2 = detection.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            if detection.in_roi:
                warning_count += 1
                label = f"BLIND SPOT | {label}"

            self._draw_label(output, label, (x1, max(20, y1 - 10)), color)

            if detection.anchor_point is not None:
                cv2.circle(output, detection.anchor_point, 5, color, -1)

        if warning_count > 0:
            self._draw_banner(output, f"BLIND SPOT ALERT: {warning_count}")

        return output

    def _draw_banner(self, frame: np.ndarray, text: str) -> None:
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

    @staticmethod
    def _normalize_polygon(
        roi_polygon: Optional[Sequence[Tuple[int, int]]],
    ) -> Optional[np.ndarray]:
        if roi_polygon is None:
            return None

        polygon = np.array(roi_polygon, dtype=np.int32)
        if polygon.ndim == 3:
            polygon = polygon.reshape((-1, 2))
        return polygon
