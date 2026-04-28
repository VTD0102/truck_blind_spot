from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .models import AlertEvent, Detection, KalmanState, MotionPrediction, Track


@dataclass
class DetectionDTO:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    in_roi: bool
    anchor_point: Optional[Tuple[int, int]]
    zone_name: Optional[str]
    risk_level: Optional[str]
    distance_m: Optional[float]

    @classmethod
    def from_detection(cls, detection: Detection) -> "DetectionDTO":
        return cls(
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            in_roi=detection.in_roi,
            anchor_point=detection.anchor_point,
            zone_name=detection.zone_name,
            risk_level=detection.risk_level,
            distance_m=detection.distance_m,
        )


@dataclass
class TrackDTO:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    age: int
    hits: int
    misses: int
    is_confirmed: bool
    status: str
    velocity: Optional[Tuple[float, float]]
    distance_m: Optional[float]
    in_roi: bool
    anchor_point: Optional[Tuple[int, int]]
    zone_name: Optional[str]
    risk_level: Optional[str]

    @classmethod
    def from_track(cls, track: Track) -> "TrackDTO":
        return cls(
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
            age=track.age,
            hits=track.hits,
            misses=track.misses,
            is_confirmed=track.is_confirmed,
            status=track.status.value,
            velocity=track.velocity,
            distance_m=track.distance_m,
            in_roi=track.in_roi,
            anchor_point=track.anchor_point,
            zone_name=track.zone_name,
            risk_level=track.risk_level,
        )


@dataclass
class KalmanStateDTO:
    state_vector: Tuple[float, float, float, float, float, float, float, float]
    covariance: Tuple[
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float, float, float],
    ]

    @classmethod
    def from_kalman_state(cls, state: KalmanState) -> "KalmanStateDTO":
        return cls(state_vector=state.state_vector, covariance=state.covariance)


@dataclass
class MotionPredictionDTO:
    track_id: int
    predicted_position: Tuple[int, int]
    confidence: float
    timestamp: float

    @classmethod
    def from_motion_prediction(cls, prediction: MotionPrediction) -> "MotionPredictionDTO":
        return cls(
            track_id=prediction.track_id,
            predicted_position=prediction.predicted_position,
            confidence=prediction.confidence,
            timestamp=prediction.timestamp,
        )


@dataclass
class AlertEventDTO:
    alert_type: str
    severity: str
    location: Tuple[int, int]
    timestamp: float
    track_id: Optional[int]
    zone_name: Optional[str]
    metadata: Optional[Dict[str, str]]

    @classmethod
    def from_alert_event(cls, event: AlertEvent) -> "AlertEventDTO":
        normalized_metadata: Optional[Dict[str, str]] = None
        if event.metadata is not None:
            normalized_metadata = {
                str(key): str(value) for key, value in event.metadata.items()
            }

        return cls(
            alert_type=event.alert_type.value,
            severity=event.severity.value,
            location=event.location,
            timestamp=event.timestamp,
            track_id=event.track_id,
            zone_name=event.zone_name,
            metadata=normalized_metadata,
        )


__all__ = [
    "AlertEventDTO",
    "DetectionDTO",
    "KalmanStateDTO",
    "MotionPredictionDTO",
    "TrackDTO",
]
