from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest
import torch

from src.detector import YOLOv9Detector


@pytest.fixture
def mock_detector(monkeypatch):
    """Detector giả không cần GPU hay file weights."""
    mock_model = mock.MagicMock()
    mock_model.stride = 32
    mock_model.pt = True
    mock_model.fp16 = False
    mock_model.names = {0: "person", 1: "car"}
    mock_model.triton = False
    mock_model.warmup = mock.MagicMock()
    mock_model.device = torch.device("cpu")

    monkeypatch.setattr("src.detector.DetectMultiBackend", mock.MagicMock(return_value=mock_model))
    monkeypatch.setattr("src.detector.select_device", lambda d: torch.device("cpu"))
    monkeypatch.setattr("src.detector.check_img_size", lambda size, s: size)

    det = YOLOv9Detector(weights_path="weights/best_roiv2.pt", device="cpu")
    det._mock_model = mock_model
    return det


def test_preprocess_returns_chw_float32_normalized(mock_detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = mock_detector._preprocess(frame)

    assert result.dtype == np.float32
    assert result.ndim == 3
    assert result.shape[0] == 3   # C first
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_nonzero_pixel_normalized_correctly(mock_detector):
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    result = mock_detector._preprocess(frame)
    assert abs(result.max() - 1.0) < 1e-5


def test_predict_returns_empty_list_when_no_detections(mock_detector):
    """predict() với mock model trả về tensor rỗng → List[Detection] rỗng."""
    empty_det = torch.zeros((0, 6))
    with mock.patch("src.detector.non_max_suppression", return_value=[empty_det]):
        mock_detector._mock_model.return_value = torch.zeros((1, 25200, 7))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = mock_detector.predict(frame)

    assert isinstance(result, list)
    assert result == []


def test_mps_init_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with mock.patch("src.detector.DetectMultiBackend"):
        with mock.patch("src.detector.select_device", return_value=torch.device("cpu")):
            with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
                with pytest.raises(RuntimeError, match="MPS không khả dụng"):
                    YOLOv9Detector(weights_path="weights/best_roiv2.pt", device="mps", backend="pytorch")


def test_mps_init_sets_mps_device_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    mock_model = mock.MagicMock()
    mock_model.stride = 32
    mock_model.pt = True
    mock_model.fp16 = False
    mock_model.names = {0: "person"}
    mock_model.triton = False
    mock_model.warmup = mock.MagicMock()

    with mock.patch("src.detector.DetectMultiBackend", return_value=mock_model):
        with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
            det = YOLOv9Detector(weights_path="weights/best_roiv2.pt", device="mps", backend="pytorch")

    assert det.device == torch.device("mps")
    assert det.fp16 is False  # FP16 tắt cho MPS


def test_mps_predict_returns_cpu_tensor_before_nms(monkeypatch):
    """Predictions phải về CPU trước NMS khi dùng MPS — test không cần hardware MPS."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    mock_model = mock.MagicMock()
    mock_model.stride = 32
    mock_model.pt = True
    mock_model.fp16 = False
    mock_model.names = {0: "person"}
    mock_model.triton = False
    mock_model.warmup = mock.MagicMock()
    fake_output = torch.zeros((1, 25200, 7))
    mock_model.return_value = fake_output

    captured = {}

    def fake_nms(preds, *args, **kwargs):
        captured["device"] = preds.device.type
        return [torch.zeros((0, 6))]

    # Mock torch.from_numpy để tensor không thực sự lên MPS (không cần hardware)
    def fake_from_numpy(arr):
        return torch.zeros(arr.shape, dtype=torch.float32)

    with mock.patch("src.detector.DetectMultiBackend", return_value=mock_model):
        with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
            with mock.patch("src.detector.non_max_suppression", side_effect=fake_nms):
                with mock.patch("src.detector.torch.from_numpy", side_effect=fake_from_numpy):
                    det = YOLOv9Detector(
                        weights_path="weights/best_roiv2.pt",
                        device="mps",
                        backend="pytorch",
                    )
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    det.predict(frame)

    assert captured["device"] == "cpu"


def test_coreml_init_raises_when_mlpackage_missing(monkeypatch):
    with mock.patch("src.detector.DetectMultiBackend"):
        with mock.patch("src.detector.select_device", return_value=torch.device("cpu")):
            with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
                with pytest.raises(FileNotFoundError, match="export_coreml.py"):
                    YOLOv9Detector(
                        weights_path="weights/best_roiv2.pt",
                        device="",
                        backend="coreml",
                    )


def test_coreml_predict_calls_model_predict(monkeypatch):
    """CoreML path gọi self._coreml_model.predict() với input đúng shape."""
    mock_coreml_model = mock.MagicMock()
    fake_output_tensor = np.zeros((1, 25200, 7), dtype=np.float32)
    mock_coreml_model.predict.return_value = {"output0": fake_output_tensor}
    mock_spec = mock.MagicMock()
    mock_spec.description.output[0].name = "output0"
    mock_spec.description.input[0].name = "images"
    mock_coreml_model.get_spec.return_value = mock_spec

    mock_ct = mock.MagicMock()
    mock_ct.models.MLModel.return_value = mock_coreml_model

    with mock.patch("src.detector.DetectMultiBackend"):
        with mock.patch("src.detector.select_device", return_value=torch.device("cpu")):
            with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
                with mock.patch.dict("sys.modules", {"coremltools": mock_ct}):
                    with mock.patch("pathlib.Path.exists", return_value=True):
                        det = YOLOv9Detector(
                            weights_path="weights/best_roiv2.pt",
                            device="",
                            backend="coreml",
                        )

    with mock.patch("src.detector.non_max_suppression", return_value=[torch.zeros((0, 6))]):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.predict(frame)

    assert mock_coreml_model.predict.called
    call_args = mock_coreml_model.predict.call_args[0][0]
    assert "images" in call_args
    assert call_args["images"].shape == (1, 3, 640, 640)
    assert result == []
