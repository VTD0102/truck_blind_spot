from __future__ import annotations

import numpy as np

from src.common.models import Detection, MotionPrediction, PredictedPoint, Track
from src.roi import ROIZone
from src.visualize import BlindSpotVisualizer


def test_visualizer_draws_object_boxes_without_per_object_text(monkeypatch) -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(30, 30, 70, 80),
        confidence=0.9,
        class_id=0,
        class_name="person",
        anchor_point=(50, 80),
    )
    track = Track(
        track_id=1,
        bbox=(30.0, 30.0, 70.0, 80.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    visualizer = BlindSpotVisualizer()
    labels: list[str] = []

    def fake_draw_label(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        labels.append(text)

    monkeypatch.setattr(visualizer, "_draw_label", fake_draw_label)

    visualizer.draw(frame, [detection], [], tracks=[track])

    assert labels == []


def test_visualizer_moves_danger_color_annotation_to_side_legend(monkeypatch) -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    zone = ROIZone(
        name="blind_left",
        polygon=[(10, 10), (140, 10), (140, 100), (10, 100)],
        color=(0, 0, 255),
        risk_level="HIGH",
    )
    detection = Detection(
        bbox=(30, 30, 70, 80),
        confidence=0.9,
        class_id=0,
        class_name="person",
        in_roi=True,
        zone_name="blind_left",
        risk_level="HIGH",
    )
    visualizer = BlindSpotVisualizer()
    labels: list[str] = []

    def fake_draw_label(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        labels.append(text)

    monkeypatch.setattr(visualizer, "_draw_label", fake_draw_label)

    visualizer.draw(frame, [detection], [zone])

    assert labels == ["NGUY HIEM", "HIGH"]
    assert all("blind_left" not in label for label in labels)
    assert all("person" not in label for label in labels)


def test_visualizer_defaults_use_compact_overlay_text() -> None:
    visualizer = BlindSpotVisualizer()

    assert visualizer.thickness == 1
    assert visualizer.font_scale <= 0.45
    assert visualizer.text_thickness == 1


def test_visualizer_uses_short_warning_banner(monkeypatch) -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    detection = Detection(
        bbox=(30, 30, 70, 80),
        confidence=0.9,
        class_id=0,
        class_name="person",
        in_roi=True,
    )
    visualizer = BlindSpotVisualizer()
    banners: list[str] = []

    def fake_draw_banner(image: np.ndarray, text: str) -> None:
        banners.append(text)

    monkeypatch.setattr(visualizer, "_draw_banner", fake_draw_banner)

    visualizer.draw(frame, [detection], [])

    assert banners == ["CANH BAO: 1"]


def test_visualizer_only_labels_medium_and_high_prediction_alerts(monkeypatch) -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    track = Track(
        track_id=1,
        bbox=(30.0, 30.0, 70.0, 80.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    predictions = {
        1: MotionPrediction(
            track_id=1,
            trajectory=[PredictedPoint(position=(90, 90), timestamp_s=1.0, confidence=0.8)],
            overall_confidence=0.3,
            alert_level="low",
            velocity_px_per_s=(0.0, 0.0),
            acceleration_px_per_s2=(0.0, 0.0),
        )
    }
    visualizer = BlindSpotVisualizer()
    labels: list[str] = []

    def fake_draw_label(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        labels.append(text)

    monkeypatch.setattr(visualizer, "_draw_label", fake_draw_label)

    visualizer.draw(frame, [], [], tracks=[track], predictions=predictions)

    assert labels == []

    predictions[1].alert_level = "high"
    visualizer.draw(frame, [], [], tracks=[track], predictions=predictions)

    assert labels == ["high 0.30"]
