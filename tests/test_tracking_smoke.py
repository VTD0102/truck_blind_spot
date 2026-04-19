from __future__ import annotations

from dataclasses import dataclass

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


def test_kalman_basic_predict_update() -> None:
    kf = BoundingBoxKalmanFilter()
    initial_box = (100.0, 100.0, 200.0, 200.0)

    initiated_box = kf.initiate(initial_box)
    predicted_box = kf.predict()
    updated_box = kf.update((110.0, 100.0, 210.0, 200.0))

    print("Kalman initiated:", initiated_box)
    print("Kalman predicted:", predicted_box)
    print("Kalman updated:", updated_box)

    assert len(initiated_box) == 4
    assert len(predicted_box) == 4
    assert len(updated_box) == 4


def test_matching_basic() -> None:
    tracker = TrackManager()
    detections = [FakeDetection((100.0, 100.0, 200.0, 200.0))]
    tracks = tracker.update(detections)

    new_detections = [FakeDetection((105.0, 100.0, 205.0, 200.0))]
    matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
        tracks,
        new_detections,
        iou_threshold=0.3,
    )

    print("Matches:", matches)
    print("Unmatched tracks:", unmatched_tracks)
    print("Unmatched detections:", unmatched_detections)

    assert len(matches) == 1
    assert len(unmatched_tracks) == 0
    assert len(unmatched_detections) == 0


def test_track_manager_same_id_across_frames() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=2, min_hits=1)

    frame1 = [FakeDetection((100.0, 100.0, 200.0, 200.0))]
    tracks1 = manager.update(frame1)
    assert len(tracks1) == 1

    first_id = tracks1[0].track_id

    frame2 = [FakeDetection((110.0, 100.0, 210.0, 200.0))]
    tracks2 = manager.update(frame2)
    assert len(tracks2) == 1

    second_id = tracks2[0].track_id

    print("Frame1 ID:", first_id)
    print("Frame2 ID:", second_id)

    assert first_id == second_id


def test_track_manager_survives_one_miss() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=2, min_hits=1)

    frame1 = [FakeDetection((100.0, 100.0, 200.0, 200.0))]
    tracks1 = manager.update(frame1)
    assert len(tracks1) == 1

    track_id = tracks1[0].track_id

    frame2: list[FakeDetection] = []
    tracks2 = manager.update(frame2)

    print("Tracks after one miss:", [(t.track_id, t.misses) for t in tracks2])

    assert len(tracks2) == 1
    assert tracks2[0].track_id == track_id
    assert tracks2[0].misses == 1


def test_track_manager_removes_after_too_many_misses() -> None:
    manager = TrackManager(iou_threshold=0.3, max_misses=1, min_hits=1)

    frame1 = [FakeDetection((100.0, 100.0, 200.0, 200.0))]
    tracks1 = manager.update(frame1)
    assert len(tracks1) == 1

    manager.update([])
    tracks3 = manager.update([])

    print("Tracks after too many misses:", tracks3)

    assert len(tracks3) == 0


def test_track_manager_uses_delta_velocity_before_buffer_ready() -> None:
    timeline = iter([0.0, 1.0])
    manager = TrackManager(
        iou_threshold=0.3,
        max_misses=2,
        min_hits=1,
        time_provider=lambda: next(timeline),
    )

    manager.update([FakeDetection((100.0, 100.0, 200.0, 200.0))])
    tracks = manager.update([FakeDetection((110.0, 100.0, 210.0, 200.0))])
    assert len(tracks) == 1
    track = tracks[0]
    assert track.velocity == (10.0, 0.0)
    assert track.velocity_buffer is not None
    assert len(track.velocity_buffer) == 2


def test_track_manager_prefers_buffer_velocity_when_ready() -> None:
    timeline = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    manager = TrackManager(
        iou_threshold=0.3,
        max_misses=2,
        min_hits=1,
        time_provider=lambda: next(timeline),
    )

    for i in range(6):
        x1 = 100.0 + 10.0 * i
        x2 = 200.0 + 10.0 * i
        tracks = manager.update([FakeDetection((x1, 100.0, x2, 200.0))])
        assert len(tracks) == 1

    track = tracks[0]
    assert track.velocity_buffer is not None
    assert len(track.velocity_buffer) >= 5
    assert track.velocity is not None
    assert abs(track.velocity[0] - 10.0) < 1e-6
    assert abs(track.velocity[1] - 0.0) < 1e-6
