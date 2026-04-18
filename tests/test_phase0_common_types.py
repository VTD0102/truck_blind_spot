from __future__ import annotations

from dataclasses import asdict

from src.tracking.track_manager import TrackManager


class FakeDetection:
    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        confidence: float = 0.9,
        class_id: int = 0,
        class_name: str = "person",
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.in_roi = False
        self.zone_name = None
        self.risk_level = None
        x1, y1, x2, y2 = bbox
        self.anchor_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))


def test_shared_models_are_reused_by_modules() -> None:
    from src.common.models import Detection, Track
    from src.detector import Detection as DetectorDetection
    from src.tracking.types import Track as TrackingTrack

    assert DetectorDetection is Detection
    assert TrackingTrack is Track


def test_track_status_lifecycle_enum() -> None:
    from src.common.enums import TrackStatus

    manager = TrackManager(iou_threshold=0.3, max_misses=2, min_hits=2)
    tracks = manager.update([FakeDetection((10.0, 10.0, 50.0, 50.0))])
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.TENTATIVE

    tracks = manager.update([FakeDetection((12.0, 10.0, 52.0, 50.0))])
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.CONFIRMED

    tracks = manager.update([])
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.LOST


def test_dto_and_schema_exports() -> None:
    from src.common.dto import DetectionDTO
    from src.common.models import Detection
    from src.common.schemas import TYPE_SCHEMAS

    detection = Detection(
        bbox=(10, 20, 30, 40),
        confidence=0.8,
        class_id=1,
        class_name="bike",
    )

    dto = DetectionDTO.from_detection(detection)
    dto_payload = asdict(dto)
    assert dto_payload["bbox"] == (10, 20, 30, 40)
    assert dto_payload["class_name"] == "bike"
    assert "Detection" in TYPE_SCHEMAS
