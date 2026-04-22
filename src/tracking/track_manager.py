from __future__ import annotations

import time
from typing import Callable, List, Protocol, Sequence

from .kalman_filter import BoundingBoxKalmanFilter
from .motion.velocity_buffer import VelocityBuffer
from .matching import match_tracks_detections
from .types import Point, Track, TrackStatus


class TrackDetection(Protocol):
    """Detection-like object required by the track manager."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    anchor_point: Point | None
    in_roi: bool
    zone_name: str | None
    risk_level: str | None
    distance_m: float | None


class TrackManager:
    """Manage track lifecycle using IoU-based assignment."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 5,
        min_hits: int = 2,
        max_trace_length: int = 30,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.max_trace_length = max_trace_length
        self._time_provider = time_provider or time.time

        self.tracks: List[Track] = []
        self._next_track_id = 1

    def update(self, detections: Sequence[TrackDetection]) -> List[Track]:
        """Update active tracks from current-frame detections."""

        self._increment_track_ages()

        matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
            self.tracks,
            detections,
            iou_threshold=self.iou_threshold,
        )

        for track_idx, detection_idx in matches:
            self._update_track(self.tracks[track_idx], detections[detection_idx])

        for track_idx in unmatched_tracks:
            self._mark_missed(self.tracks[track_idx])

        for detection_idx in unmatched_detections:
            self.tracks.append(self._create_track(detections[detection_idx]))

        self._prune_dead_tracks()
        return list(self.tracks)

    def reset(self) -> None:
        """Clear all active tracks and restart ID assignment."""
        self.tracks.clear()
        self._next_track_id = 1

    def _increment_track_ages(self) -> None:
        for track in self.tracks:
            track.age += 1
            if track.kalman is not None:
                track.bbox = track.kalman.predict()
                track.velocity = track.kalman.get_velocity()

    def _create_track(self, detection: TrackDetection) -> Track:
        status = (
            TrackStatus.CONFIRMED
            if self.min_hits <= 1
            else TrackStatus.TENTATIVE
        )
        kalman = BoundingBoxKalmanFilter()
        initial_bbox = kalman.initiate(detection.bbox)
        track = Track(
            track_id=self._next_track_id,
            bbox=initial_bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age=1,
            hits=1,
            misses=0,
            is_confirmed=self.min_hits <= 1,
            status=status,
            velocity=kalman.get_velocity(),
            kalman=kalman,
            anchor_point=detection.anchor_point,
            in_roi=detection.in_roi,
            zone_name=detection.zone_name,
            risk_level=detection.risk_level,
            distance_m=detection.distance_m,
        )

        timestamp = float(self._time_provider())
        if detection.anchor_point is not None:
            track.velocity_buffer = VelocityBuffer()
            track.velocity_buffer.push(detection.anchor_point, timestamp)

        self._append_trace(track, detection.anchor_point)

        self._next_track_id += 1
        return track

    def _update_track(self, track: Track, detection: TrackDetection) -> None:
        if track.kalman is None:
            track.kalman = BoundingBoxKalmanFilter()
            track.kalman.initiate(track.bbox)

        track.bbox = track.kalman.update(detection.bbox)
        track.confidence = detection.confidence
        track.class_id = detection.class_id
        track.class_name = detection.class_name

        track.anchor_point = detection.anchor_point
        track.in_roi = detection.in_roi
        track.zone_name = detection.zone_name
        track.risk_level = detection.risk_level
        track.distance_m = detection.distance_m

        track.hits += 1
        track.misses = 0
        track.is_confirmed = track.hits >= self.min_hits
        track.status = (
            TrackStatus.CONFIRMED
            if track.is_confirmed
            else TrackStatus.TENTATIVE
        )
        track.velocity = track.kalman.get_velocity()

        timestamp = float(self._time_provider())
        self._update_velocity_buffer(track, detection.anchor_point, timestamp)

        self._append_trace(track, detection.anchor_point)

    def _update_velocity_buffer(
        self,
        track: Track,
        point: Point | None,
        timestamp: float,
    ) -> None:
        if point is None:
            return

        if track.velocity_buffer is None:
            track.velocity_buffer = VelocityBuffer()

        track.velocity_buffer.push(point, timestamp)

    def _mark_missed(self, track: Track) -> None:
        track.misses += 1
        track.status = TrackStatus.LOST

    def _prune_dead_tracks(self) -> None:
        self.tracks = [
            track for track in self.tracks if track.misses <= self.max_misses
        ]

    def _append_trace(self, track: Track, point: Point | None) -> None:
        if point is None:
            return

        track.trace.append(point)
        if len(track.trace) > self.max_trace_length:
            track.trace = track.trace[-self.max_trace_length :]


__all__ = ["TrackDetection", "TrackManager"]
