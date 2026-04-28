from __future__ import annotations

from typing import Dict, Tuple

SchemaDefinition = Dict[str, str]


TYPE_SCHEMAS: Dict[str, SchemaDefinition] = {
    "Detection": {
        "bbox": "tuple[int, int, int, int]",
        "confidence": "float",
        "class_id": "int",
        "class_name": "str",
        "anchor_point": "tuple[int, int] | None",
        "in_roi": "bool",
        "zone_name": "str | None",
        "risk_level": "str | None",
        "distance_m": "float | None",
    },
    "Track": {
        "track_id": "int",
        "bbox": "tuple[float, float, float, float]",
        "confidence": "float",
        "class_id": "int",
        "class_name": "str",
        "age": "int",
        "hits": "int",
        "misses": "int",
        "is_confirmed": "bool",
        "status": "TrackStatus",
        "velocity": "tuple[float, float] | None",
        "kalman": "BoundingBoxKalmanFilter | None",
        "distance_m": "float | None",
        "velocity_buffer": "VelocityBufferLike | None",
        "in_roi": "bool",
        "anchor_point": "tuple[int, int] | None",
        "zone_name": "str | None",
        "risk_level": "str | None",
        "trace": "list[tuple[int, int]]",
    },
    "KalmanState": {
        "state_vector": "8D tuple[float, ...]",
        "covariance": "8x8 tuple[tuple[float, ...], ...]",
    },
    "MotionPrediction": {
        "track_id": "int",
        "predicted_position": "tuple[int, int]",
        "confidence": "float",
        "timestamp": "float",
    },
    "AlertEvent": {
        "alert_type": "AlertType",
        "severity": "AlertLevel",
        "location": "tuple[int, int]",
        "timestamp": "float",
        "track_id": "int | None",
        "zone_name": "str | None",
        "metadata": "dict[str, object] | None",
    },
}


__all__ = ["TYPE_SCHEMAS", "SchemaDefinition"]
