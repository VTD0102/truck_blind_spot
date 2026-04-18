from __future__ import annotations

try:
    from src.common.enums import TrackStatus
    from src.common.models import FloatBBox, Point, Track, Velocity
except ImportError:
    from common.enums import TrackStatus  # type: ignore
    from common.models import FloatBBox, Point, Track, Velocity  # type: ignore

__all__ = ["FloatBBox", "Point", "Track", "TrackStatus", "Velocity"]
