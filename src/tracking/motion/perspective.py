"""Biến đổi phối cảnh pixel <-> BEV cho motion prediction (Plan.md §2.0)."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..types import Point

BEVPoint = Tuple[float, float]
MOCK_HOMOGRAPHY_MATRIX: np.ndarray = np.eye(3, dtype=np.float64)


class PerspectiveTransform:
    """Biến đổi điểm ảnh camera sang mặt phẳng BEV bằng homography."""

    def __init__(self, homography_matrix: np.ndarray) -> None:
        if homography_matrix.shape != (3, 3):
            raise ValueError(
                "homography_matrix phải có kích thước 3x3 "
                f"(nhận được {homography_matrix.shape})."
            )
        self.homography_matrix = np.asarray(homography_matrix, dtype=np.float64)
        determinant = float(np.linalg.det(self.homography_matrix))
        if np.isclose(determinant, 0.0):
            raise ValueError("homography_matrix suy biến (determinant ~ 0), không thể nghịch đảo.")
        self._inverse_homography = np.linalg.inv(self.homography_matrix)

    def pixel_to_bev(self, point: Point) -> BEVPoint:
        """Chuyển điểm pixel (x, y) sang tọa độ BEV (float, float)."""
        return self._project_point(point, self.homography_matrix)

    def bev_to_pixel(self, bev_point: BEVPoint) -> Point:
        """Chuyển điểm BEV ngược về pixel để vẽ lên frame."""
        x, y = self._project_point(bev_point, self._inverse_homography)
        return (int(round(x)), int(round(y)))

    @staticmethod
    def estimate_distance(
        bbox_height_px: float,
        focal_length: float,
        real_height_m: float,
    ) -> float:
        """Ước lượng khoảng cách theo pinhole model: Z = (f * H_real) / h_pixel."""
        if bbox_height_px <= 0.0:
            raise ValueError(
                "bbox_height_px phải > 0 để ước lượng khoảng cách "
                f"(nhận được {bbox_height_px})."
            )
        if focal_length <= 0.0:
            raise ValueError(
                f"focal_length phải > 0 (nhận được {focal_length})."
            )
        if real_height_m <= 0.0:
            raise ValueError(
                f"real_height_m phải > 0 (nhận được {real_height_m})."
            )
        return (float(focal_length) * float(real_height_m)) / float(bbox_height_px)

    @staticmethod
    def _project_point(point: Tuple[float, float], matrix: np.ndarray) -> BEVPoint:
        homogeneous = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
        projected = matrix @ homogeneous
        if np.isclose(projected[2], 0.0):
            raise ValueError("Điểm biến đổi có tọa độ homogeneous z gần bằng 0.")
        x = projected[0] / projected[2]
        y = projected[1] / projected[2]
        return (float(x), float(y))


__all__ = ["BEVPoint", "MOCK_HOMOGRAPHY_MATRIX", "PerspectiveTransform"]
