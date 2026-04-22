from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Tuple

from .enums import AlertLevel, AlertType, TrackStatus

if TYPE_CHECKING:
    from src.tracking.kalman_filter import BoundingBoxKalmanFilter

FloatBBox = Tuple[float, float, float, float]
Point = Tuple[int, int]
Velocity = Tuple[float, float]
KalmanVector8D = Tuple[float, float, float, float, float, float, float, float]
Covariance8x8 = Tuple[
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
    Tuple[float, float, float, float, float, float, float, float],
]


class VelocityBufferLike(Protocol):
    """Protocol tối thiểu cho buffer vận tốc gắn với Track."""

    def push(self, point: Point, timestamp: float) -> None:
        """Thêm mẫu (tọa độ, thời gian) vào buffer."""

    def __len__(self) -> int:
        """Số mẫu hiện có."""

    def get_velocity(self) -> Optional[Velocity]:
        """Ước lượng vận tốc hiện tại."""

    def get_smoothed_velocity(self, window: int = 3) -> Optional[Velocity]:
        """Ước lượng vận tốc có làm mượt."""


@dataclass
class Detection:
    """Object detection output used across detector, ROI, tracking and rendering."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    in_roi: bool = False
    anchor_point: Optional[Point] = None
    zone_name: Optional[str] = None
    risk_level: Optional[str] = None
    distance_m: Optional[float] = None


@dataclass
class Track:
    """Persistent object state maintained across multiple video frames."""

    track_id: int
    bbox: FloatBBox
    confidence: float
    class_id: int
    class_name: str

    age: int = 1
    hits: int = 1
    misses: int = 0
    is_confirmed: bool = False
    status: TrackStatus = TrackStatus.TENTATIVE

    velocity: Optional[Velocity] = None
    kalman: Optional[BoundingBoxKalmanFilter] = None
    velocity_buffer: Optional[VelocityBufferLike] = None
    distance_m: Optional[float] = None
    in_roi: bool = False
    anchor_point: Optional[Point] = None
    zone_name: Optional[str] = None
    risk_level: Optional[str] = None
    trace: List[Point] = field(default_factory=list)


@dataclass
class KalmanState:
    """Snapshot of a Kalman filter state vector and covariance matrix."""

    state_vector: KalmanVector8D
    covariance: Covariance8x8


@dataclass
class PredictedPoint:
    """Vị trí dự đoán tại một thời điểm trong tương lai."""

    position: Point
    timestamp_s: float      # seconds from now (0.5, 1.0, 2.0)
    confidence: float       # [0.0, 1.0]


@dataclass
class MotionPrediction:
    """Kết quả dự đoán chuyển động cho một track."""

    track_id: int
    trajectory: List[PredictedPoint]
    overall_confidence: float           # mean confidence across trajectory
    alert_level: str                    # "none", "low", "medium", "high"
    velocity_px_per_s: Velocity
    acceleration_px_per_s2: Velocity


@dataclass
class AlertEvent:
    """Alert emitted from tracking and motion prediction outputs."""

    alert_type: AlertType
    severity: AlertLevel
    location: Point
    timestamp: float
    track_id: Optional[int] = None
    zone_name: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None
