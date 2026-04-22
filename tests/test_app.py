from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import app


class _FakeCapture:
    def __init__(self) -> None:
        self._reads = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._reads == 0:
            self._reads += 1
            return True, np.zeros((32, 32, 3), dtype=np.uint8)
        return False, None

    def get(self, prop: int) -> float:
        if prop == app.cv2.CAP_PROP_FRAME_WIDTH:
            return 32.0
        if prop == app.cv2.CAP_PROP_FRAME_HEIGHT:
            return 32.0
        if prop == app.cv2.CAP_PROP_FPS:
            return 25.0
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        return True

    def release(self) -> None:
        self.released = True


class _FakePipeline:
    def __init__(self) -> None:
        self.frames_processed = 0

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, list[object], list[object]]:
        self.frames_processed += 1
        return frame.copy(), [], []


def test_app_main_accepts_pipeline_track_output(monkeypatch) -> None:
    fake_pipeline = _FakePipeline()
    fake_capture = _FakeCapture()

    monkeypatch.setattr(
        app,
        "parse_args",
        lambda: SimpleNamespace(
            source="demo.mp4",
            weights="weights/best_6k.pt",
            roi="configs/roi.json",
            roi_profile="front_camera",
            classes_config="configs/classes.yaml",
            device="",
            conf_thres=0.25,
            iou_thres=0.45,
            output=None,
            loop=False,
            prediction_horizon=1.0,
            alert_threshold=0.6,
        ),
    )
    monkeypatch.setattr(app, "BlindSpotPipeline", lambda **kwargs: fake_pipeline)
    monkeypatch.setattr(app, "open_capture", lambda source: fake_capture)
    monkeypatch.setattr(app, "draw_overlay", lambda frame, fps, paused: None)
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)

    app.main()

    assert fake_pipeline.frames_processed == 1
    assert fake_capture.released is True
