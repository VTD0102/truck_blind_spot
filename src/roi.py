from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]
Color = Tuple[int, int, int]


class ROIZone:
    def __init__(
        self,
        name: str,
        polygon: Sequence[Sequence[int]],
        color: Sequence[int],
        risk_level: str = "WARNING",
    ) -> None:
        if not polygon or len(polygon) < 3:
            raise ValueError(f"ROI zone '{name}' must contain at least 3 points.")

        self.name = name
        self.base_points: List[Point] = [(int(p[0]), int(p[1])) for p in polygon]
        self.points: List[Point] = list(self.base_points)
        self.polygon = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        self.color: Color = tuple(int(c) for c in color)
        self.risk_level = str(risk_level)

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.points = [
            (int(round(x * scale_x)), int(round(y * scale_y)))
            for x, y in self.base_points
        ]
        # Chuyển đổi thành mảng numpy để dùng với OpenCV
        self.polygon = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def contains_point(self, point: Point) -> bool:
        x, y = point
        # Sử dụng pointPolygonTest của OpenCV (trả về >= 0 nếu nằm trong hoặc trên viền)
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False)
        return result >= 0


class MultiPolygonROI:
    def __init__(self, roi_config_path: str, profile_name: str = "front_camera") -> None:
        self.roi_config_path = Path(roi_config_path)
        self.profile_name = profile_name
        self.config = self._load_config()

        profiles = self.config.get("profiles", {})
        if profile_name not in profiles:
            available = ", ".join(sorted(profiles.keys()))
            raise ValueError(
                f"ROI profile '{profile_name}' not found. Available profiles: {available}"
            )

        profile = profiles[profile_name]
        image_size = profile.get("image_size")
        if not image_size:
            raise ValueError(f"ROI profile '{profile_name}' must define image_size.")

        self.base_width = int(image_size["w"])
        self.base_height = int(image_size["h"])
        self.current_width = self.base_width
        self.current_height = self.base_height
        self.check_point = profile.get("check_point", "bbox_bottom_center")

        zones_config = profile.get("zones")
        if not zones_config:
            raise ValueError(f"ROI profile '{profile_name}' must contain at least one zone.")

        self.zones: List[ROIZone] = [
            ROIZone(
                name=zone["name"],
                polygon=zone["polygon"],
                color=zone.get("color", [0, 165, 255]),
                risk_level=zone.get("risk_level", "WARNING"),
            )
            for zone in zones_config
        ]

    def update_frame_size(self, frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Invalid frame size.")

        if frame_width == self.current_width and frame_height == self.current_height:
            return

        scale_x = frame_width / self.base_width
        scale_y = frame_height / self.base_height

        for zone in self.zones:
            zone.scale(scale_x, scale_y)

        self.current_width = frame_width
        self.current_height = frame_height

    def get_reference_point(self, bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        if self.check_point == "bbox_bottom_center":
            return ((x1 + x2) // 2, y2)
        if self.check_point == "bbox_top_center":
            return ((x1 + x2) // 2, y1)
        if self.check_point == "bbox_left_center":
            return (x1, (y1 + y2) // 2)
        if self.check_point == "bbox_right_center":
            return (x2, (y1 + y2) // 2)
        # Mặc định trả về điểm tâm của bounding box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_zone(self, point: Point) -> Optional[ROIZone]:
        matched_zones = [zone for zone in self.zones if zone.contains_point(point)]
        if not matched_zones:
            return None

        priority = {
            "HIGH": 3,
            "MEDIUM": 2,
            "WARNING": 1,
            "LOW": 0,
        }
        matched_zones.sort(
            key=lambda z: priority.get(z.risk_level.upper(), 0),
            reverse=True,
        )
        return matched_zones[0]

    def contains_bbox(self, bbox: BBox) -> bool:
        return self.get_zone(self.get_reference_point(bbox)) is not None

    def _load_config(self) -> Dict:
        """Tải file cấu hình JSON."""
        if not self.roi_config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy cấu hình ROI: {self.roi_config_path}")

        with self.roi_config_path.open("r", encoding="utf-8") as file:
            return json.load(file)