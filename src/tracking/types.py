from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

FloatBBox = Tuple[float, float, float, float]
Point = Tuple[int, int]
Velocity = Tuple[float, float]


@dataclass
class Track:
    """Persistent object state maintained across multiple video frames."""

    track_id: int

    # Current track state aligned with the detection schema where possible.
    bbox: FloatBBox
    confidence: float
    class_id: int
    class_name: str

    # Minimal lifecycle state for matching and pruning.
    age: int = 1
    hits: int = 1
    misses: int = 0
    is_confirmed: bool = False

    # Optional motion state for Kalman-based prediction later.
    velocity: Optional[Velocity] = None

    # ROI-enriched fields kept compatible with Detection.
    in_roi: bool = False
    anchor_point: Optional[Point] = None
    zone_name: Optional[str] = None
    risk_level: Optional[str] = None

    # Short point history for visualization or motion estimation.
    trace: List[Point] = field(default_factory=list)
