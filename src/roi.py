from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class PolygonROI:
    """Lớp xử lý Vùng quan tâm (ROI) dựa trên đa giác."""
    def __init__(self, roi_config_path: str) -> None:
        # Đường dẫn đến file cấu hình ROI
        self.roi_config_path = Path(roi_config_path)
        self.config = self._load_config()
        self.image_size = self.config.get("image_size")
        
        # Điểm kiểm tra trên bounding box (mặc định là tâm)
        self.check_point = self.config.get("check_point", "bbox_center")

        polygon = self.config.get("polygon")
        if not polygon or len(polygon) < 3:
            raise ValueError("Đa giác ROI phải chứa ít nhất 3 điểm.")

        # Danh sách các điểm của đa giác
        self.points: List[Tuple[int, int]] = [
            (int(point[0]), int(point[1])) for point in polygon
        ]
        # Chuyển đổi thành mảng numpy để dùng với OpenCV
        self.polygon = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def contains_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Kiểm tra xem bounding box có nằm trong ROI hay không."""
        return self.contains_point(self.get_reference_point(bbox))

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Kiểm tra xem một điểm cụ thể có nằm trong đa giác ROI hay không."""
        x, y = point
        # Sử dụng pointPolygonTest của OpenCV (trả về >= 0 nếu nằm trong hoặc trên viền)
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False)
        return result >= 0

    def get_reference_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Lấy điểm tham chiếu của bounding box dựa trên cấu hình."""
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

    def _load_config(self) -> Dict:
        """Tải file cấu hình JSON."""
        if not self.roi_config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy cấu hình ROI: {self.roi_config_path}")

        with self.roi_config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
