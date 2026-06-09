# MPS + CoreML Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đạt ≥ 28 FPS trên macOS M3 bằng cách thêm PyTorch MPS backend (Phase 1) và CoreML backend (Phase 2) vào `YOLOv9Detector`.

**Architecture:** Phase 1 bypass `select_device()` của YOLOv9 vendored khi `device="mps"`, chuyển predictions về CPU trước NMS. Phase 2 thêm CoreML inference path với `coremltools`, tách shared `_preprocess()` để cả hai backend dùng chung. `--backend coreml` và `--device mps` là hai flag độc lập, không dùng chung.

**Tech Stack:** PyTorch MPS (torch>=2.0), coremltools>=7.0, onnx>=1.14 (Phase 2), pytest + monkeypatch cho tests.

---

## File Map

| File | Hành động | Nội dung thay đổi |
|------|-----------|-------------------|
| `src/detector.py` | Sửa | Thêm `backend` param, `_preprocess()`, MPS bypass, CoreML path |
| `src/pipeline.py` | Sửa | Thêm `backend` param, forward xuống detector |
| `app.py` | Sửa | Thêm `--backend` arg + validation xung đột |
| `tests/test_detector_backends.py` | Tạo mới | Unit tests cho MPS + CoreML backends |
| `tests/test_app.py` | Sửa | Thêm `backend` vào 3 SimpleNamespace |
| `tools/export_coreml.py` | Tạo mới | Script export PT → ONNX → mlpackage |
| `requirements.txt` | Sửa | Thêm coremltools, onnx (Phase 2) |
| `docs/run-commands.md` | Tạo mới | Tài liệu tất cả câu lệnh chạy |

---

## Task 1: Tách `_preprocess()` thành shared method

**Files:**
- Sửa: `src/detector.py`
- Tạo: `tests/test_detector_backends.py`

- [ ] **Bước 1: Tạo file test mới với failing test cho `_preprocess`**

Tạo `tests/test_detector_backends.py`:

```python
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
```

- [ ] **Bước 2: Chạy test để xác nhận fail**

```bash
python -m pytest tests/test_detector_backends.py -v
```

Kết quả kỳ vọng: `FAILED — AttributeError: 'YOLOv9Detector' object has no attribute '_preprocess'`

- [ ] **Bước 3: Tách `_preprocess()` trong `src/detector.py`**

Thêm method sau `predict()`:

```python
def _preprocess(self, frame: np.ndarray) -> np.ndarray:
    """Letterbox + CHW transpose + BGR→RGB + normalize float32 [0, 1]."""
    img = letterbox(frame, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    return np.ascontiguousarray(img).astype(np.float32) / 255.0
```

Cập nhật `predict()` — thay 5 dòng preprocess cũ bằng:

```python
# Đẩy dữ liệu lên GPU/CPU
img = self._preprocess(frame)
tensor = torch.from_numpy(img).to(self.device)
if self.fp16:
    tensor = tensor.half()
if tensor.ndim == 3:
    tensor = tensor.unsqueeze(0)
```

Đồng thời đổi `self.model.device` → `self.device` trong dòng `.to()` vừa viết (device bây giờ được lưu ở `self.device`, không lấy từ model).

- [ ] **Bước 4: Chạy test để xác nhận pass**

```bash
python -m pytest tests/test_detector_backends.py -v
```

Kết quả kỳ vọng: `2 passed`

- [ ] **Bước 5: Chạy toàn bộ test suite để đảm bảo không regression**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Kết quả kỳ vọng: tất cả tests pass (tracking smoke tests không cần GPU).

- [ ] **Bước 6: Commit**

```bash
git add src/detector.py tests/test_detector_backends.py
git commit -m "refactor(detector): tách _preprocess() thành shared method"
```

---

## Task 2: MPS device bypass + NMS CPU fix

**Files:**
- Sửa: `src/detector.py`
- Sửa: `tests/test_detector_backends.py`

- [ ] **Bước 1: Thêm failing tests vào `tests/test_detector_backends.py`**

Thêm vào cuối file:

```python
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
    """Predictions phải về CPU trước NMS khi dùng MPS."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    mock_model = mock.MagicMock()
    mock_model.stride = 32
    mock_model.pt = True
    mock_model.fp16 = False
    mock_model.names = {0: "person"}
    mock_model.triton = False
    mock_model.warmup = mock.MagicMock()
    # Giả lập inference trả về tensor trên "mps" (giả)
    fake_output = torch.zeros((1, 25200, 7))  # CPU tensor, đại diện cho output
    mock_model.return_value = fake_output

    captured = {}

    def fake_nms(preds, *args, **kwargs):
        captured["device"] = preds.device.type
        return [torch.zeros((0, 6))]

    with mock.patch("src.detector.DetectMultiBackend", return_value=mock_model):
        with mock.patch("src.detector.check_img_size", return_value=(640, 640)):
            with mock.patch("src.detector.non_max_suppression", side_effect=fake_nms):
                det = YOLOv9Detector(weights_path="weights/best_roiv2.pt", device="mps", backend="pytorch")
                det.device = torch.device("mps")  # force mps device type
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                det.predict(frame)

    assert captured["device"] == "cpu"
```

- [ ] **Bước 2: Chạy test để xác nhận fail**

```bash
python -m pytest tests/test_detector_backends.py::test_mps_init_raises_when_unavailable tests/test_detector_backends.py::test_mps_init_sets_mps_device_when_available -v
```

Kết quả kỳ vọng: `FAILED — TypeError: __init__() got unexpected keyword argument 'backend'`

- [ ] **Bước 3: Thêm `backend` param và MPS bypass vào `YOLOv9Detector.__init__`**

Thêm `backend: str = "pytorch"` vào cuối danh sách param của `__init__`.

Trong thân `__init__`, **thay thế đúng một dòng** `self.device = select_device(device)` bằng khối dưới đây (vị trí đầu `__init__`, trước khi `self.image_size`, `self.conf_threshold`, v.v. được set):

```python
self.backend = backend

if backend == "pytorch" and device == "mps":
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS không khả dụng trên máy này. Chạy lại với --device cpu."
        )
    self.device = torch.device("mps")
elif backend == "coreml":
    self.device = torch.device("cpu")
else:
    self.device = select_device(device)
```

Tiếp tục **xuống dưới**, thay dòng FP16 (`use_fp16 = half or (self.device.type != "cpu")`):

```python
# FP16 chỉ bật cho CUDA — MPS FP32 đủ nhanh, CoreML tự quản lý precision
use_fp16 = half or (self.device.type == "cuda")
```

Không thay đổi bất kỳ dòng nào khác trong `__init__`.

- [ ] **Bước 4: Thêm `.cpu()` trước NMS trong `predict()`**

Trong `predict()`, sau dòng `predictions = self._unwrap_predictions(predictions)`:

```python
predictions = self._unwrap_predictions(predictions)
# NMS có op không support trên MPS — chuyển về CPU
if self.device.type == "mps":
    predictions = predictions.cpu()
predictions = non_max_suppression(
    ...
```

- [ ] **Bước 5: Chạy tests MPS**

```bash
python -m pytest tests/test_detector_backends.py -v
```

Kết quả kỳ vọng: `5 passed`

- [ ] **Bước 6: Chạy toàn bộ test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Kết quả kỳ vọng: tất cả pass.

- [ ] **Bước 7: Commit**

```bash
git add src/detector.py tests/test_detector_backends.py
git commit -m "feat(detector): thêm MPS backend — bypass select_device, fix NMS CPU transfer"
```

---

## Task 3: Wire `backend` qua `pipeline.py` và `app.py`

**Files:**
- Sửa: `src/pipeline.py`
- Sửa: `app.py`
- Sửa: `tests/test_app.py`

- [ ] **Bước 1: Thêm failing test cho `--backend` arg**

Thêm vào cuối `tests/test_app.py`:

```python
def test_parse_args_accepts_backend_coreml(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py", "--backend", "coreml"])
    args = app.parse_args()
    assert args.backend == "coreml"


def test_parse_args_default_backend_is_pytorch(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["app.py"])
    args = app.parse_args()
    assert args.backend == "pytorch"
```

- [ ] **Bước 2: Chạy để xác nhận fail**

```bash
python -m pytest tests/test_app.py::test_parse_args_accepts_backend_coreml tests/test_app.py::test_parse_args_default_backend_is_pytorch -v
```

Kết quả kỳ vọng: `FAILED — error: unrecognized arguments: --backend`

- [ ] **Bước 3: Thêm `--backend` vào `app.py`**

Trong `parse_args()` của `app.py`, thêm sau `--device`:

```python
parser.add_argument(
    "--backend",
    type=str,
    default="pytorch",
    choices=["pytorch", "coreml"],
    help="Inference backend: pytorch (default) hoặc coreml (cần export trước).",
)
```

Trong `main()`, sau `args = parse_args()`, thêm validation:

```python
if args.backend == "coreml" and args.device == "mps":
    parser.error("--backend coreml và --device mps không dùng chung. Chọn một trong hai.")
```

Và thêm `backend=args.backend` vào `BlindSpotPipeline(...)`:

```python
pipeline = BlindSpotPipeline(
    ...
    device=args.device,
    backend=args.backend,
    ...
)
```

- [ ] **Bước 4: Thêm `backend` param vào `BlindSpotPipeline`**

Trong `src/pipeline.py`, thêm `backend: str = "pytorch"` vào `BlindSpotPipeline.__init__` sau `device`:

```python
def __init__(
    self,
    ...
    device: str = "",
    backend: str = "pytorch",
    ...
) -> None:
```

Và forward xuống detector trong cùng method:

```python
self.detector = YOLOv9Detector(
    weights_path=weights_path,
    classes_config_path=classes_config_path,
    device=device,
    backend=backend,
    image_size=image_size,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold,
)
```

- [ ] **Bước 5: Cập nhật 3 `SimpleNamespace` trong `tests/test_app.py`**

Trong cả 3 hàm test `test_app_main_accepts_pipeline_track_output`, `test_app_main_no_display_skips_opencv_window_calls`, `test_parse_args_accepts_no_display` — thêm `backend="pytorch"` vào mỗi `SimpleNamespace`:

```python
lambda: SimpleNamespace(
    source="demo.mp4",
    ...
    enable_adaptive_scale=False,
    backend="pytorch",   # ← thêm dòng này
)
```

> Lưu ý: `test_parse_args_accepts_no_display` không dùng `SimpleNamespace` — bỏ qua.

- [ ] **Bước 6: Chạy toàn bộ test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Kết quả kỳ vọng: tất cả pass.

- [ ] **Bước 7: Commit**

```bash
git add src/pipeline.py app.py tests/test_app.py
git commit -m "feat: wire backend param qua pipeline + app, thêm --backend CLI arg"
```

---

## Task 4: CoreML backend trong `YOLOv9Detector`

**Files:**
- Sửa: `src/detector.py`
- Sửa: `tests/test_detector_backends.py`

- [ ] **Bước 1: Thêm failing tests cho CoreML**

Thêm vào cuối `tests/test_detector_backends.py`:

```python
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
    mock_coreml_model.predict.return_value = {"output": fake_output_tensor}
    mock_spec = mock.MagicMock()
    mock_spec.description.output[0].name = "output"
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
    assert "image" in call_args
    assert call_args["image"].shape == (1, 3, 640, 640)
    assert result == []
```

- [ ] **Bước 2: Chạy để xác nhận fail**

```bash
python -m pytest tests/test_detector_backends.py::test_coreml_init_raises_when_mlpackage_missing tests/test_detector_backends.py::test_coreml_predict_calls_model_predict -v
```

Kết quả kỳ vọng: `FAILED`

- [ ] **Bước 3: Implement `_init_coreml()` và CoreML early-exit trong `src/detector.py`**

**3a.** Thêm method `_init_coreml()` vào `YOLOv9Detector`:

```python
def _init_coreml(self, mlpackage_path: str) -> None:
    """Khởi tạo CoreML model từ .mlpackage."""
    import coremltools as ct  # import lazy — chỉ khi dùng CoreML

    if not Path(mlpackage_path).exists():
        raise FileNotFoundError(
            f"CoreML model không tìm thấy: {mlpackage_path}\n"
            f"Chạy trước: python3 tools/export_coreml.py --weights {self.weights_path}"
        )
    self._coreml_model = ct.models.MLModel(mlpackage_path)
    spec = self._coreml_model.get_spec()
    self._coreml_output_key = spec.description.output[0].name
```

**3b.** Trong `__init__`, thêm khối CoreML early-exit **sau dòng `use_fp16 = ...` và trước dòng `self.model = DetectMultiBackend(...)`**:

```python
# CoreML path: bỏ qua DetectMultiBackend và warmup
if self.backend == "coreml":
    self.stride = 32
    self.pt = False
    self.fp16 = False
    self.imgsz = check_img_size(image_size, s=self.stride)
    mlpackage_path = str(Path(self.weights_path).with_suffix(".mlpackage"))
    self._init_coreml(mlpackage_path)
    return
```

Tại thời điểm này `self.class_names`, `self.conf_threshold`, `self.augment`, `self.agnostic_nms`, v.v. đã được set bởi các dòng trước đó trong `__init__` — không cần set lại.

- [ ] **Bước 4: Thêm CoreML dispatch trong `predict()`**

Trong `predict()`, trước đoạn preprocess PyTorch, thêm dispatch:

```python
def predict(self, frame: np.ndarray) -> List[Detection]:
    if frame is None or frame.size == 0:
        raise ValueError("Khung hình đầu vào rỗng (empty).")

    if self.backend == "coreml":
        return self._predict_coreml(frame)

    # PyTorch path (giữ nguyên, chỉ đổi preprocess thành _preprocess)
    ...
```

Thêm method `_predict_coreml()`:

```python
def _predict_coreml(self, frame: np.ndarray) -> List[Detection]:
    """Inference qua CoreML backend."""
    img = self._preprocess(frame)
    input_arr = img[np.newaxis]  # (1, 3, H, W) float32

    out = self._coreml_model.predict({"image": input_arr})
    raw = out[self._coreml_output_key]
    predictions = torch.from_numpy(raw if isinstance(raw, np.ndarray) else np.array(raw))

    predictions = non_max_suppression(
        predictions,
        self.conf_threshold,
        self.iou_threshold,
        self.classes,
        self.agnostic_nms,
        max_det=self.max_det,
    )

    detections: List[Detection] = []
    det = predictions[0]
    if not len(det):
        return detections

    det[:, :4] = scale_boxes(
        (self.imgsz[0], self.imgsz[1]), det[:, :4], frame.shape
    ).round()

    for *xyxy, confidence, class_id in det.tolist():
        class_index = int(class_id)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence),
                class_id=class_index,
                class_name=self.class_names.get(class_index, str(class_index)),
            )
        )
    return detections
```

- [ ] **Bước 5: Chạy tests CoreML**

```bash
python -m pytest tests/test_detector_backends.py -v
```

Kết quả kỳ vọng: tất cả pass.

- [ ] **Bước 6: Chạy toàn bộ test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Kết quả kỳ vọng: tất cả pass.

- [ ] **Bước 7: Commit**

```bash
git add src/detector.py tests/test_detector_backends.py
git commit -m "feat(detector): thêm CoreML backend — _init_coreml, _predict_coreml"
```

---

## Task 5: `tools/export_coreml.py` + `requirements.txt`

**Files:**
- Tạo mới: `tools/export_coreml.py`
- Sửa: `requirements.txt`

- [ ] **Bước 1: Thêm dependencies vào `requirements.txt`**

Thêm section mới vào cuối `requirements.txt`:

```
# ── Apple Silicon Acceleration (Phase 2 — optional) ─────────────────────────
coremltools>=7.0    # export + load .mlpackage trên macOS Apple Silicon
onnx>=1.14.0        # intermediate format khi export PT → ONNX → CoreML
```

- [ ] **Bước 2: Tạo `tools/export_coreml.py`**

```python
"""
tools/export_coreml.py — Export YOLOv9 weights sang CoreML (.mlpackage).

Chạy một lần trên máy macOS Apple Silicon:
    python3 tools/export_coreml.py --weights weights/best_roiv2.pt

Output:
    weights/best_roiv2.onnx
    weights/best_roiv2.mlpackage/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov9"
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

import numpy as np
import torch
import coremltools as ct

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_coreml")


def export(weights: str, imgsz: tuple[int, int] = (640, 640)) -> None:
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    onnx_path = weights_path.with_suffix(".onnx")
    mlpackage_path = weights_path.with_suffix(".mlpackage")

    # ── Bước 1: Load model trên CPU ──────────────────────────────────────
    logger.info("Load model từ %s ...", weights_path)
    device = select_device("cpu")
    model = DetectMultiBackend(str(weights_path), device=device, dnn=False, fp16=False)
    model.eval()

    dummy = torch.zeros(1, 3, *imgsz)

    # ── Bước 2: PT → ONNX ────────────────────────────────────────────────
    logger.info("Export sang ONNX: %s ...", onnx_path)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info("ONNX export thành công.")

    # ── Bước 3: ONNX → CoreML ────────────────────────────────────────────
    logger.info("Convert sang CoreML: %s ...", mlpackage_path)
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    coreml_model = ct.convert(
        onnx_model,
        compute_units=ct.ComputeUnit.ALL,  # Neural Engine + GPU + CPU
        minimum_deployment_target=ct.target.iOS15,
    )
    coreml_model.save(str(mlpackage_path))

    # In output key để dùng khi inference
    spec = coreml_model.get_spec()
    out_key = spec.description.output[0].name
    logger.info("CoreML export thành công.")
    logger.info("Output key: %r — tự động detect khi chạy app.", out_key)
    logger.info("mlpackage: %s", mlpackage_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv9 PT → ONNX → CoreML")
    parser.add_argument("--weights", type=str, default="weights/best_roiv2.pt")
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (default: 640 640)",
    )
    args = parser.parse_args()
    export(args.weights, imgsz=tuple(args.imgsz))


if __name__ == "__main__":
    main()
```

- [ ] **Bước 3: Kiểm tra tool chạy được (smoke test thủ công — không cần GPU)**

```bash
python3 tools/export_coreml.py --help
```

Kết quả kỳ vọng: hiển thị help text không lỗi.

- [ ] **Bước 4: Commit**

```bash
git add tools/export_coreml.py requirements.txt
git commit -m "feat: thêm tools/export_coreml.py và dependencies coremltools/onnx"
```

---

## Task 6: `docs/run-commands.md`

**Files:**
- Tạo mới: `docs/run-commands.md`

- [ ] **Bước 1: Tạo file `docs/run-commands.md`**

```markdown
# Truck Blind Spot Detection — Câu lệnh chạy

Tài liệu này tổng hợp tất cả câu lệnh thường dùng trong dự án.

---

## Cài đặt môi trường

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Chạy app chính (`app.py`)

### CPU (mặc định)

```bash
python3 app.py
```

### Apple Silicon GPU — MPS (Phase 1)

Dùng cho macOS M1/M2/M3. Không cần export trước.

```bash
python3 app.py --device mps
```

### Apple Neural Engine — CoreML (Phase 2)

Cần export model một lần trước:

```bash
# Bước 1: Export (chạy một lần)
python3 tools/export_coreml.py --weights weights/best_roiv2.pt

# Bước 2: Chạy app với CoreML backend
python3 app.py --backend coreml
```

### Tùy chọn thêm

```bash
# Đổi ROI profile (camera sau xe)
python3 app.py --roi-profile rear_camera

# Nguồn video / webcam
python3 app.py --source assets/videos/demo4.mp4
python3 app.py --source 0                       # webcam index 0

# Lặp lại video
python3 app.py --loop

# Lưu kết quả ra file
python3 app.py --output outputs/result.mp4

# Thay đổi ngưỡng detection
python3 app.py --conf-thres 0.3 --iou-thres 0.5

# Bật frame skipping khi FPS thấp (Kalman predict-only cho frame bị skip)
python3 app.py --device mps --enable-frame-skip

# Bật tự động giảm resolution khi FPS < target
python3 app.py --device mps --enable-adaptive-scale

# Đặt target FPS (default 25)
python3 app.py --device mps --target-fps 30

# Ghi log cảnh báo ra CSV
python3 app.py --device mps --alert-log outputs/alerts.csv

# Chạy không hiển thị cửa sổ (headless)
python3 app.py --no-display --output outputs/result.mp4

# Kết hợp MPS + tất cả optimizations
python3 app.py --device mps --enable-frame-skip --enable-adaptive-scale --target-fps 30
```

Phím tắt trong cửa sổ: `p` pause/resume · `r` restart · `q` thoát.

---

## Pipeline CLI (`src/pipeline.py`)

Chạy inference trên ảnh hoặc video, không cần cửa sổ `app.py`.

```bash
# Ảnh
python3 src/pipeline.py --source path/to/image.jpg --show

# Video
python3 src/pipeline.py --source assets/videos/demo4.mp4 --output outputs/out.mp4 --show
```

---

## Export CoreML

```bash
python3 tools/export_coreml.py --weights weights/best_roiv2.pt
# Output: weights/best_roiv2.onnx + weights/best_roiv2.mlpackage/
```

---

## Đánh giá ROI recall (`src/roi_evaluation.py`)

```bash
python -m src.roi_evaluation \
  --weights weights/best_roiv2.pt \
  --data configs/blindspot.yaml \
  --roi configs/roi.json \
  --roi-profile front_camera \
  --split val --conf-thres 0.25 --iou-match 0.5 \
  --output-dir outputs/roi_eval
```

---

## Tests

```bash
# Toàn bộ test suite (không cần GPU hay weights)
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py

# Chỉ tracking tests
python -m pytest tests/test_tracking_smoke.py -v

# Chỉ backend tests (MPS + CoreML)
python -m pytest tests/test_detector_backends.py -v
```
```

- [ ] **Bước 2: Chạy toàn bộ tests lần cuối để xác nhận không có gì vỡ**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Kết quả kỳ vọng: tất cả pass.

- [ ] **Bước 3: Commit**

```bash
git add docs/run-commands.md
git commit -m "docs: thêm run-commands.md tổng hợp tất cả câu lệnh chạy"
```

---

## Kiểm tra thực tế sau khi hoàn thành

Sau khi tất cả tasks xong, chạy tuần tự để verify:

```bash
# Verify Phase 1 MPS
python3 app.py --device mps --no-display --source assets/videos/demo4.mp4
# Quan sát FPS trong log — kỳ vọng ≥ 28 FPS

# Nếu Phase 1 chưa đủ 28 FPS, chạy Phase 2:
python3 tools/export_coreml.py --weights weights/best_roiv2.pt
python3 app.py --backend coreml --no-display --source assets/videos/demo4.mp4
# Quan sát FPS trong log — kỳ vọng ≥ 40 FPS
```
