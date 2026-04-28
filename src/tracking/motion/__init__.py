"""Motion prediction utilities for tracked objects (Plan.md §2)."""
from __future__ import annotations

from ...common.models import MotionPrediction, PredictedPoint  # re-export tiện lợi
from .extrapolator import TrajectoryExtrapolator
from .perspective import BEVPoint, MOCK_HOMOGRAPHY_MATRIX, PerspectiveTransform
from .predictor import MotionPredictor
from .velocity_buffer import VelocityBuffer

__all__ = [
    "BEVPoint",
    "MOCK_HOMOGRAPHY_MATRIX",
    "MotionPrediction",
    "MotionPredictor",
    "PerspectiveTransform",
    "PredictedPoint",
    "TrajectoryExtrapolator",
    "VelocityBuffer",
]
