from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import FloatBBox, Velocity

CxCyWhBBox = Tuple[float, float, float, float]
DEFAULT_PROCESS_NOISE = 1.0
DEFAULT_MEASUREMENT_NOISE = 10.0


class BoundingBoxKalmanFilter:
    """Kalman filter for one tracked bounding box.

    The filter keeps an 8D state:
    ``[cx, cy, w, h, vx, vy, vw, vh]``

    Measurements come from detector boxes converted to:
    ``[cx, cy, w, h]``.
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ) -> None:
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive (received {dt}).")
        if self.process_noise <= 0.0:
            raise ValueError(
                "process_noise must be positive. "
                "Image-space tracking assumes non-zero motion uncertainty."
            )
        if self.measurement_noise <= 0.0:
            raise ValueError(
                "measurement_noise must be positive. "
                "Detector boxes always include some localization noise."
            )

        self.state = np.zeros((8, 1), dtype=np.float64)
        self.covariance = np.eye(8, dtype=np.float64)

        self.transition = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.transition[i, i + 4] = self.dt

        self.measurement = np.zeros((4, 8), dtype=np.float64)
        self.measurement[0, 0] = 1.0
        self.measurement[1, 1] = 1.0
        self.measurement[2, 2] = 1.0
        self.measurement[3, 3] = 1.0

        # The defaults assume modest per-frame motion drift (~1 px std dev)
        # and noisier detector boxes (~3 px std dev) in image coordinates.
        self.process_covariance = np.eye(8, dtype=np.float64) * self.process_noise
        self.measurement_covariance = np.eye(4, dtype=np.float64) * self.measurement_noise
        self.identity = np.eye(8, dtype=np.float64)

        self._is_initialized = False

    def initiate(self, bbox_xyxy: FloatBBox) -> FloatBBox:
        """Initialize the filter from a detection bbox in ``xyxy`` format."""

        cx, cy, w, h = self.xyxy_to_cxcywh(bbox_xyxy)
        self.state = np.array(
            [[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]],
            dtype=np.float64,
        )

        # Start with higher uncertainty on velocity than on the measured box.
        self.covariance = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0])
        self._is_initialized = True
        return self.get_bbox_xyxy()

    def predict(self) -> FloatBBox:
        """Predict the next bounding box estimate."""

        self._ensure_initialized()

        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_covariance
        )
        self._clamp_state_size()
        return self.get_bbox_xyxy()

    def update(self, bbox_xyxy: FloatBBox) -> FloatBBox:
        """Correct the predicted state using a detection bbox in ``xyxy`` format."""

        self._ensure_initialized()

        cx, cy, w, h = self.xyxy_to_cxcywh(bbox_xyxy)
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float64)

        innovation = measurement - (self.measurement @ self.state)
        innovation_covariance = (
            self.measurement @ self.covariance @ self.measurement.T
            + self.measurement_covariance
        )
        kalman_gain = (
            self.covariance
            @ self.measurement.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = self.state + kalman_gain @ innovation
        self.covariance = (
            self.identity - kalman_gain @ self.measurement
        ) @ self.covariance
        self._clamp_state_size()
        return self.get_bbox_xyxy()

    def get_bbox_xyxy(self) -> FloatBBox:
        """Return the current estimated box in ``(x1, y1, x2, y2)`` format."""

        self._ensure_initialized()
        cx = float(self.state[0, 0])
        cy = float(self.state[1, 0])
        w = float(self.state[2, 0])
        h = float(self.state[3, 0])
        return self.cxcywh_to_xyxy((cx, cy, w, h))

    def get_velocity(self) -> Velocity:
        """Return the current ``(vx, vy)`` estimate in pixels per frame."""

        self._ensure_initialized()
        return (float(self.state[4, 0]), float(self.state[5, 0]))

    @staticmethod
    def xyxy_to_cxcywh(bbox_xyxy: FloatBBox) -> CxCyWhBBox:
        """Convert ``(x1, y1, x2, y2)`` box format to ``(cx, cy, w, h)``."""

        x1, y1, x2, y2 = bbox_xyxy
        w = max(0.0, float(x2) - float(x1))
        h = max(0.0, float(y2) - float(y1))
        cx = float(x1) + w / 2.0
        cy = float(y1) + h / 2.0
        return (cx, cy, w, h)

    @staticmethod
    def cxcywh_to_xyxy(bbox_cxcywh: CxCyWhBBox) -> FloatBBox:
        """Convert ``(cx, cy, w, h)`` box format to ``(x1, y1, x2, y2)``."""

        cx, cy, w, h = bbox_cxcywh
        half_w = max(0.0, float(w)) / 2.0
        half_h = max(0.0, float(h)) / 2.0
        x1 = float(cx) - half_w
        y1 = float(cy) - half_h
        x2 = float(cx) + half_w
        y2 = float(cy) + half_h
        return (x1, y1, x2, y2)

    def _ensure_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("Kalman filter must be initiated before use.")

    def _clamp_state_size(self) -> None:
        # Keep width and height valid even if prediction/update drifts.
        self.state[2, 0] = max(1e-6, float(self.state[2, 0]))
        self.state[3, 0] = max(1e-6, float(self.state[3, 0]))


__all__ = [
    "BoundingBoxKalmanFilter",
    "CxCyWhBBox",
    "DEFAULT_MEASUREMENT_NOISE",
    "DEFAULT_PROCESS_NOISE",
]
