from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from .common.models import Detection, Track
    from .roi import ROIZone
except ImportError:
    from common.models import Detection, Track  # type: ignore
    from roi import ROIZone  # type: ignore


class BlindSpotVisualizer:
    """Render detections, ROI overlays, and track metadata onto a frame."""

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
        tracks: Optional[Sequence[Track]] = None,
    ) -> np.ndarray:
        """Draw detections, ROI overlays, and tracks on a frame."""

        output = frame.copy() if copy else frame

        if roi_zones:
            self._draw_roi_zones(output, roi_zones)

        warning_count = 0
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

            if detection.anchor_point is not None:
                cv2.circle(output, detection.anchor_point, 5, color, -1)

        if tracks:
            self._draw_tracks(output, tracks, roi_zones)

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

    def _draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        roi_zones: Optional[Sequence[ROIZone]],
    ) -> None:
        for track in tracks:
            x1, y1, x2, y2 = self._to_int_bbox(track.bbox)
            color = self._resolve_track_color(track, roi_zones)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            self._draw_label(frame, f"ID {track.track_id}", (x1, min(frame.shape[0] - 10, y2 + 22)), color)

            anchor = track.anchor_point or (
                int(round((x1 + x2) / 2.0)),
                int(round((y1 + y2) / 2.0)),
            )

            if track.velocity is None:
                continue

            vx, vy = track.velocity
            if abs(vx) <= 0.5 and abs(vy) <= 0.5:
                continue

            end_point = (
                int(round(anchor[0] + vx)),
                int(round(anchor[1] + vy)),
            )
            cv2.arrowedLine(
                frame,
                anchor,
                end_point,
                color,
                2,
                tipLength=0.25,
            )

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

    def _resolve_track_color(
        self,
        track: Track,
        roi_zones: Optional[Sequence[ROIZone]],
    ) -> Tuple[int, int, int]:
        if not track.in_roi or not track.zone_name or not roi_zones:
            return self.normal_color if not track.in_roi else self.warning_color

        for zone in roi_zones:
            if zone.name == track.zone_name:
                return zone.color
        return self.warning_color

    def _draw_banner(self, frame: np.ndarray, text: str) -> None:
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale + 0.1,
            2,
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
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            2,
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
    def _to_int_bbox(
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        return tuple(int(round(value)) for value in bbox)
