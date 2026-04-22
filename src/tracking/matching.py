from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import FloatBBox

Match = Tuple[int, int]
DistanceMetric = Literal["iou", "mahalanobis"]


class HasBBox(Protocol):
    bbox: FloatBBox


PairwiseScore = Callable[[HasBBox, HasBBox], float]
MatrixTransform = Callable[[np.ndarray], np.ndarray]
MatchPredicate = Callable[[float, float], bool]


@dataclass(frozen=True)
class AssignmentMetric:
    pairwise_score: PairwiseScore
    to_cost_matrix: MatrixTransform
    is_valid_match: MatchPredicate


def compute_iou(box1: FloatBBox, box2: FloatBBox) -> float:
    """Compute intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    if intersection <= 0.0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def _score_iou(track: HasBBox, detection: HasBBox) -> float:
    return compute_iou(track.bbox, detection.bbox)


def _resolve_assignment_metric(
    distance_metric: str | DistanceMetric,
) -> AssignmentMetric:
    normalized_metric = distance_metric.strip().lower()

    if normalized_metric == "iou":
        return AssignmentMetric(
            pairwise_score=_score_iou,
            to_cost_matrix=lambda score_matrix: 1.0 - score_matrix,
            is_valid_match=lambda score, threshold: score >= threshold,
        )

    if normalized_metric == "mahalanobis":
        raise NotImplementedError(
            "Mahalanobis matching is reserved for a future gating implementation."
        )

    raise ValueError(
        "Unsupported distance_metric. "
        "Expected one of: 'iou', 'mahalanobis'."
    )


def match_tracks_detections(
    tracks: Sequence[HasBBox],
    detections: Sequence[HasBBox],
    iou_threshold: float = 0.3,
    distance_metric: str | DistanceMetric = "iou",
) -> Tuple[List[Match], List[int], List[int]]:
    """Match tracks to detections using Hungarian assignment."""

    if not tracks and not detections:
        return [], [], []
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    assignment_metric = _resolve_assignment_metric(distance_metric)
    num_tracks = len(tracks)
    num_detections = len(detections)
    score_matrix = np.zeros((num_tracks, num_detections), dtype=np.float64)

    for track_idx, track in enumerate(tracks):
        for detection_idx, detection in enumerate(detections):
            score_matrix[track_idx, detection_idx] = assignment_metric.pairwise_score(
                track,
                detection,
            )

    cost_matrix = assignment_metric.to_cost_matrix(score_matrix)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Match] = []
    matched_track_indices = set()
    matched_detection_indices = set()

    for track_idx, detection_idx in zip(row_indices, col_indices):
        score = score_matrix[track_idx, detection_idx]
        if not assignment_metric.is_valid_match(score, iou_threshold):
            continue

        matches.append((track_idx, detection_idx))
        matched_track_indices.add(track_idx)
        matched_detection_indices.add(detection_idx)

    unmatched_tracks = [
        track_idx
        for track_idx in range(num_tracks)
        if track_idx not in matched_track_indices
    ]
    unmatched_detections = [
        detection_idx
        for detection_idx in range(num_detections)
        if detection_idx not in matched_detection_indices
    ]

    return matches, unmatched_tracks, unmatched_detections


__all__ = [
    "AssignmentMetric",
    "DistanceMetric",
    "Match",
    "compute_iou",
    "match_tracks_detections",
]
