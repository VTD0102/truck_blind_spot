from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .enums import AlertLevel, AlertType, TrackStatus

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
class MotionPrediction:
    """Predicted motion state for one tracked object."""

    track_id: int
    predicted_position: Point
    confidence: float
    timestamp: float


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
