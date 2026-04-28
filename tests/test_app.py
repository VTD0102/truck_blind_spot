from __future__ import annotations

import sys
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
    window_calls: list[tuple[str, tuple[object, ...]]] = []

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
            no_display=False,
            prediction_horizon=1.0,
            alert_threshold=0.6,
        ),
    )
    monkeypatch.setattr(app, "BlindSpotPipeline", lambda **kwargs: fake_pipeline)
    monkeypatch.setattr(app, "open_capture", lambda source: fake_capture)
    monkeypatch.setattr(app, "draw_overlay", lambda frame, fps, paused: None)
    monkeypatch.setattr(
        app.cv2,
        "namedWindow",
        lambda *args, **kwargs: window_calls.append(("namedWindow", args)),
    )
    monkeypatch.setattr(
        app.cv2,
        "resizeWindow",
        lambda *args, **kwargs: window_calls.append(("resizeWindow", args)),
    )
    monkeypatch.setattr(
        app.cv2,
        "setWindowProperty",
        lambda *args, **kwargs: window_calls.append(("setWindowProperty", args)),
    )
    monkeypatch.setattr(app.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)

    app.main()

    assert fake_pipeline.frames_processed == 1
    assert fake_capture.released is True
    assert (
        "resizeWindow",
        (app.WINDOW_NAME, app.DEFAULT_WINDOW_WIDTH, app.DEFAULT_WINDOW_HEIGHT),
    ) in window_calls
    assert (
        "setWindowProperty",
        (app.WINDOW_NAME, app.cv2.WND_PROP_FULLSCREEN, app.cv2.WINDOW_FULLSCREEN),
    ) in window_calls


def test_app_main_no_display_skips_opencv_window_calls(monkeypatch) -> None:
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
            device="0",
            conf_thres=0.25,
            iou_thres=0.45,
            output=None,
            loop=False,
            no_display=True,
            prediction_horizon=1.0,
            alert_threshold=0.6,
        ),
    )
    monkeypatch.setattr(app, "BlindSpotPipeline", lambda **kwargs: fake_pipeline)
    monkeypatch.setattr(app, "open_capture", lambda source: fake_capture)
    monkeypatch.setattr(app, "draw_overlay", lambda frame, fps, paused: None)
    monkeypatch.setattr(
        app.cv2,
        "namedWindow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Không được mở cửa sổ khi chạy no-display.")
        ),
    )
    monkeypatch.setattr(
        app.cv2,
        "imshow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Không được hiển thị frame khi chạy no-display.")
        ),
    )
    monkeypatch.setattr(
        app.cv2,
        "waitKey",
        lambda delay: (_ for _ in ()).throw(
            AssertionError("Không được chờ phím khi chạy no-display.")
        ),
    )
    monkeypatch.setattr(
        app.cv2,
        "destroyAllWindows",
        lambda: (_ for _ in ()).throw(
            AssertionError("Không được gọi destroyAllWindows khi chạy no-display.")
        ),
    )

    app.main()

    assert fake_pipeline.frames_processed == 1
    assert fake_capture.released is True


def test_parse_args_accepts_no_display(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py", "--no-display"])

    args = app.parse_args()

    assert args.no_display is True
