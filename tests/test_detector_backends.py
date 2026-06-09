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
