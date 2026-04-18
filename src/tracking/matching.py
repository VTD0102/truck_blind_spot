from __future__ import annotations

from typing import List, Protocol, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import FloatBBox, Track

Match = Tuple[int, int]


class HasBBox(Protocol):
    bbox: FloatBBox


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


def match_tracks_detections(
    tracks: Sequence[Track],
    detections: Sequence[HasBBox],
    iou_threshold: float = 0.3,
) -> Tuple[List[Match], List[int], List[int]]:
    """Match tracks to detections using Hungarian assignment with IoU gating."""

    if not tracks and not detections:
        return [], [], []
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    num_tracks = len(tracks)
    num_detections = len(detections)
    iou_matrix = np.zeros((num_tracks, num_detections), dtype=np.float64)

    for track_idx, track in enumerate(tracks):
        for detection_idx, detection in enumerate(detections):
            iou_matrix[track_idx, detection_idx] = compute_iou(
                track.bbox,
                detection.bbox,
            )

    cost_matrix = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Match] = []
    matched_track_indices = set()
    matched_detection_indices = set()

    for track_idx, detection_idx in zip(row_indices, col_indices):
        iou = iou_matrix[track_idx, detection_idx]
        if iou < iou_threshold:
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


__all__ = ["Match", "compute_iou", "match_tracks_detections"]