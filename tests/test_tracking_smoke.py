from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.common.enums import TrackStatus
from src.common.models import Detection, Track
from src.tracking.kalman_filter import BoundingBoxKalmanFilter
from src.tracking.matching import match_tracks_detections
from src.tracking.track_manager import TrackManager


@dataclass
class FakeDetection:
    bbox: tuple[float, float, float, float]
    confidence: float = 0.9
    class_id: int = 0
    class_name: str = "person"
    anchor_point: tuple[int, int] | None = None
    in_roi: bool = False
    zone_name: str | None = None
    risk_level: str | None = None
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if self.anchor_point is None:
            x1, y1, x2, y2 = self.bbox
            self.anchor_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _make_track(
    bbox: tuple[float, float, float, float],
    track_id: int = 1,
) -> Track:
    return Track(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name="person",
    )


def test_kalman_basic_predict_update() -> None:
    kf = BoundingBoxKalmanFilter()
    initiated_box = kf.initiate((100.0, 100.0, 200.0, 200.0))
    predicted_box = kf.predict()
    updated_box = kf.update((110.0, 100.0, 210.0, 200.0))

    assert initiated_box == pytest.approx((100.0, 100.0, 200.0, 200.0))
    assert predicted_box == pytest.approx((100.0, 100.0, 200.0, 200.0))
    assert updated_box[0] > predicted_box[0]
    assert updated_box[2] > predicted_box[2]
    assert kf.get_velocity()[0] > 0.0


def test_matching_returns_all_detections_when_no_tracks() -> None:
    matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
        [],
        [FakeDetection((10.0, 10.0, 30.0, 30.0)), FakeDetection((40.0, 40.0, 60.0, 60.0))],
        distance_metric="iou",
    )

    assert matches == []
    assert unmatched_tracks == []
    assert unmatched_detections == [0, 1]


def test_matching_returns_all_tracks_when_no_detections() -> None:
    matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
        [_make_track((10.0, 10.0, 30.0, 30.0)), _make_track((40.0, 40.0, 60.0, 60.0), track_id=2)],
        [],
        distance_metric="iou",
    )

    assert matches == []
    assert unmatched_tracks == [0, 1]
    assert unmatched_detections == []


def test_matching_perfect_overlap_pairs_track_and_detection() -> None:
    matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
        [_make_track((10.0, 10.0, 30.0, 30.0))],
        [FakeDetection((10.0, 10.0, 30.0, 30.0))],
        iou_threshold=0.3,
    )

    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_matching_zero_overlap_keeps_items_unmatched() -> None:
    matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
        [_make_track((10.0, 10.0, 30.0, 30.0))],
        [FakeDetection((100.0, 100.0, 130.0, 130.0))],
        iou_threshold=0.3,
    )

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_matching_rejects_unknown_distance_metric() -> None:
    with pytest.raises(ValueError):
        match_tracks_detections(
            [_make_track((10.0, 10.0, 30.0, 30.0))],
            [FakeDetection((10.0, 10.0, 30.0, 30.0))],
            distance_metric="unsupported",
        )


def test_track_manager_track_birth_respects_min_hits() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=2, min_hits=2)

    first_pass = manager.update([FakeDetection((100.0, 100.0, 200.0, 200.0))])
    assert len(first_pass) == 1
    assert first_pass[0].is_confirmed is False
    assert first_pass[0].status == TrackStatus.TENTATIVE

    second_pass = manager.update([FakeDetection((105.0, 100.0, 205.0, 200.0))])
    assert len(second_pass) == 1
    assert second_pass[0].hits == 2
    assert second_pass[0].is_confirmed is True
    assert second_pass[0].status == TrackStatus.CONFIRMED


def test_track_manager_track_death_respects_max_misses() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=1, min_hits=1)

    created = manager.update([FakeDetection((100.0, 100.0, 200.0, 200.0))])
    assert len(created) == 1

    missed_once = manager.update([])
    assert len(missed_once) == 1
    assert missed_once[0].status == TrackStatus.LOST
    assert missed_once[0].misses == 1

    removed = manager.update([])
    assert removed == []


def test_track_manager_uses_kalman_state_for_bbox_and_velocity() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=2, min_hits=1)

    manager.update([FakeDetection((100.0, 100.0, 200.0, 200.0))])
    tracks = manager.update([FakeDetection((110.0, 100.0, 210.0, 200.0))])

    assert len(tracks) == 1
    track = tracks[0]
    assert track.kalman is not None
    assert track.bbox == pytest.approx(track.kalman.get_bbox_xyxy())
    assert track.velocity == pytest.approx(track.kalman.get_velocity())
    assert track.velocity[0] > 0.0


def test_track_manager_maintains_identity_with_predicted_matching() -> None:
    manager = TrackManager(iou_threshold=0.5, max_misses=2, min_hits=1)

    first = manager.update([FakeDetection((0.0, 0.0, 100.0, 100.0))])
    second = manager.update([FakeDetection((30.0, 0.0, 130.0, 100.0))])
    third = manager.update([FakeDetection((70.0, 0.0, 170.0, 100.0))])

    assert len(first) == 1
    assert len(second) == 1
    assert len(third) == 1
    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert third[0].hits == 3
    assert third[0].status == TrackStatus.CONFIRMED


def test_pipeline_process_frame_returns_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.pipeline as pipeline_module

    @dataclass
    class FakeZone:
        name: str = "front"
        risk_level: str = "high"
        color: tuple[int, int, int] = (0, 0, 255)
        points: tuple[tuple[int, int], ...] = ((0, 0), (20, 0), (20, 20), (0, 20))

    class FakeDetector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def predict(self, frame: np.ndarray) -> list[Detection]:
            return [
                Detection(
                    bbox=(10, 10, 30, 30),
                    confidence=0.95,
                    class_id=0,
                    class_name="person",
                )
            ]

    class FakeROI:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.zones = [FakeZone()]

        def update_frame_size(self, width: int, height: int) -> None:
            self.frame_size = (width, height)

        def get_reference_point(
            self,
            bbox: tuple[int, int, int, int],
        ) -> tuple[int, int]:
            x1, y1, x2, y2 = bbox
            return (int((x1 + x2) / 2), int(y2))

        def get_zone(self, point: tuple[int, int] | None) -> FakeZone | None:
            return self.zones[0] if point is not None else None

    class FakeVisualizer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.last_tracks: list[Track] | None = None

        def draw(
            self,
            frame: np.ndarray,
            detections: list[Detection],
            roi_zones: list[FakeZone] | None = None,
            copy: bool = True,
            tracks: list[Track] | None = None,
        ) -> np.ndarray:
            self.last_tracks = tracks
            return frame.copy() if copy else frame

    monkeypatch.setattr(pipeline_module, "YOLOv9Detector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "MultiPolygonROI", FakeROI)
    monkeypatch.setattr(pipeline_module, "BlindSpotVisualizer", FakeVisualizer)

    pipeline = pipeline_module.BlindSpotPipeline()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    annotated_frame, detections, tracks = pipeline.process_frame(frame)

    assert annotated_frame.shape == frame.shape
    assert len(detections) == 1
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].anchor_point == (20, 30)
    assert pipeline.visualizer.last_tracks == tracks
