# OpenVINO CPU Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay thế PyTorch CPU inference bằng OpenVINO backend trong `YOLOv9Detector` để đạt ≥ 25 FPS trên Intel laptop.

**Architecture:** Thêm `backend` param vào `YOLOv9Detector` — khi `backend="openvino"` thì bỏ qua PyTorch model loading và dùng OpenVINO `InferRequest` trực tiếp. Pre/post processing (letterbox, NMS, scale_boxes) được tách thành `_preprocess()` dùng chung cho cả hai backend. Export script chạy một lần để sinh file model.xml/model.bin.

**Tech Stack:** `openvino>=2024.0`, `nncf>=2.9` (optional INT8), `torch.onnx.export`, `pytest`, Python 3.10+

---

## File Map

| File | Thay đổi |
|------|---------|
| `requirements.txt` | Thêm `openvino>=2024.0`, `nncf>=2.9` |
| `src/detector.py` | Thêm `backend`, `openvino_model_dir` params; tách `_preprocess()`; thêm `_init_openvino()`, `_infer_pytorch()`, `_infer_openvino()` |
| `src/pipeline.py` | Thêm `backend: str = "pytorch"` param, forward xuống `YOLOv9Detector` |
| `app.py` | Thêm `--backend` CLI arg, pass xuống `BlindSpotPipeline` |
| `tools/export_openvino.py` | File mới: export PT → ONNX → OpenVINO IR, optional INT8 |
| `tests/test_openvino_detector.py` | File mới: unit tests cho backend mới |

---

## Task 1: Thêm dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Thêm openvino và nncf vào requirements.txt**

Thêm vào cuối section `# ── Deep Learning`:

```
openvino>=2024.0
nncf>=2.9                   # Chỉ cần cho INT8 quantization
```

- [ ] **Step 2: Cài đặt dependencies**

```bash
pip install openvino>=2024.0 nncf>=2.9
```

Expected: cài thành công, `python -c "import openvino; print(openvino.__version__)"` in ra version.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: thêm openvino và nncf vào requirements"
```

---

## Task 2: Viết failing tests cho detector backend mới

**Files:**
- Create: `tests/test_openvino_detector.py`

- [ ] **Step 1: Viết test file**

```python
"""Tests cho OpenVINO backend của YOLOv9Detector."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import YOLOv9Detector


# ── path derivation ────────────────────────────────────────────────────────

def test_derive_openvino_dir_default():
    """Suy ra đúng thư mục OpenVINO từ weights path."""
    result = YOLOv9Detector._derive_openvino_dir("weights/best_6k.pt")
    assert result == str(Path("weights/best_6k_openvino"))


def test_derive_openvino_dir_absolute():
    """Hoạt động đúng với absolute path."""
    result = YOLOv9Detector._derive_openvino_dir("/opt/models/my_model.pt")
    assert result == "/opt/models/my_model_openvino"


# ── init validation ────────────────────────────────────────────────────────

def test_invalid_backend_raises():
    """Backend không hợp lệ phải raise ValueError."""
    with pytest.raises(ValueError, match="backend"):
        with patch.object(YOLOv9Detector, "_init_pytorch"):
            YOLOv9Detector(
                weights_path="weights/best_6k.pt",
                backend="tensorrt",  # không hỗ trợ
            )


def test_openvino_missing_model_dir_raises(tmp_path):
    """FileNotFoundError khi model.xml không tồn tại."""
    with pytest.raises(FileNotFoundError, match="model.xml"):
        with patch.object(YOLOv9Detector, "_init_pytorch"):
            YOLOv9Detector(
                weights_path="weights/best_6k.pt",
                backend="openvino",
                openvino_model_dir=str(tmp_path / "nonexistent_openvino"),
            )


# ── preprocess ─────────────────────────────────────────────────────────────

def _make_detector_no_model() -> YOLOv9Detector:
    """Tạo detector stub với đủ attributes để test _preprocess."""
    det = YOLOv9Detector.__new__(YOLOv9Detector)
    det.imgsz = (640, 640)
    det.stride = 32
    det.pt = False
    return det


def test_preprocess_output_shape():
    """_preprocess trả về shape [1, 3, 640, 640]."""
    det = _make_detector_no_model()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = det._preprocess(frame)
    assert result.shape == (1, 3, 640, 640)


def test_preprocess_output_dtype():
    """_preprocess trả về float32."""
    det = _make_detector_no_model()
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    result = det._preprocess(frame)
    assert result.dtype == np.float32


def test_preprocess_normalized():
    """Giá trị pixel sau preprocess nằm trong [0, 1]."""
    det = _make_detector_no_model()
    frame = np.full((640, 640, 3), 255, dtype=np.uint8)
    result = det._preprocess(frame)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


# ── openvino inference dispatch ────────────────────────────────────────────

def test_openvino_infer_called_not_pytorch(tmp_path):
    """Khi backend=openvino, _infer_openvino được gọi thay vì PyTorch model."""
    # Tạo dummy model.xml để pass FileNotFoundError check
    model_dir = tmp_path / "model_openvino"
    model_dir.mkdir()
    (model_dir / "model.xml").write_text("<net/>")

    fake_output = np.random.rand(1, 300, 85).astype(np.float32)

    with patch("src.detector.ov") as mock_ov:
        mock_compiled = MagicMock()
        mock_infer_req = MagicMock()
        mock_infer_req.infer.return_value = {mock_compiled.output.return_value: fake_output}
        mock_compiled.create_infer_request.return_value = mock_infer_req
        mock_ov.Core.return_value.compile_model.return_value = mock_compiled

        det = YOLOv9Detector.__new__(YOLOv9Detector)
        det.imgsz = (640, 640)
        det.stride = 32
        det.pt = False
        det.fp16 = False
        det.conf_threshold = 0.25
        det.iou_threshold = 0.45
        det.classes = None
        det.agnostic_nms = False
        det.max_det = 300
        det.class_names = {0: "person"}
        det._backend = "openvino"
        det._ov_infer = mock_infer_req
        det._ov_output_key = mock_compiled.output.return_value

        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Kết quả là list (có thể rỗng nếu fake output không qua NMS)
        result = det.predict(frame)
        assert isinstance(result, list)
        mock_infer_req.infer.assert_called_once()
```

- [ ] **Step 2: Chạy để xác nhận tất cả đều FAIL**

```bash
python -m pytest tests/test_openvino_detector.py -v 2>&1 | head -50
```

Expected: Nhiều lỗi như `AttributeError: type object 'YOLOv9Detector' has no attribute '_derive_openvino_dir'`, `ImportError`, hoặc tất cả FAILED.

---

## Task 3: Refactor `src/detector.py`

**Files:**
- Modify: `src/detector.py`

- [ ] **Step 1: Thêm import openvino (lazy) và thay đổi `__init__` signature**

Thay toàn bộ `src/detector.py` bằng nội dung sau:

```python
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

try:
    from .common.models import Detection
except ImportError:
    from common.models import Detection  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov9"

if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

try:
    import openvino as ov
except ImportError:
    ov = None  # type: ignore


class YOLOv9Detector:
    """Lớp bao bọc (wrapper) cho mô hình YOLOv9."""

    def __init__(
        self,
        weights_path: str = "weights/best_small.pt",
        classes_config_path: str = "configs/classes.yaml",
        device: str = "",
        image_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        classes: Optional[Sequence[int]] = None,
        half: bool = False,
        dnn: bool = False,
        augment: bool = False,
        agnostic_nms: bool = False,
        backend: str = "pytorch",
        openvino_model_dir: str = "",
    ) -> None:
        if backend not in ("pytorch", "openvino"):
            raise ValueError(f"backend phải là 'pytorch' hoặc 'openvino', nhận: {backend!r}")

        self._backend = backend
        self.weights_path = self._resolve_path(weights_path)
        self.classes_config_path = self._resolve_path(classes_config_path)
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.classes = list(classes) if classes is not None else None
        self.augment = augment
        self.agnostic_nms = agnostic_nms

        self.class_names = self._load_class_names(self.classes_config_path)

        if backend == "openvino":
            model_dir = openvino_model_dir or self._derive_openvino_dir(self.weights_path)
            self.stride = 32
            self.pt = False
            self.fp16 = False
            self.imgsz = check_img_size(self.image_size, s=self.stride)
            self._init_openvino(model_dir)
        else:
            self._init_pytorch(dnn=dnn, half=half, device=device)

    # ── backend init ──────────────────────────────────────────────────────

    def _init_pytorch(self, dnn: bool, half: bool, device: str) -> None:
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            self.weights_path,
            device=self.device,
            dnn=dnn,
            data=None,
            fp16=half and self.device.type != "cpu",
        )
        self.stride = self.model.stride
        self.pt = self.model.pt
        self.imgsz = check_img_size(self.image_size, s=self.stride)
        self.fp16 = bool(self.model.fp16)

        if not self.class_names:
            self.class_names = self._normalize_model_names(self.model.names)

        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))

    def _init_openvino(self, model_dir: str) -> None:
        if ov is None:
            raise ImportError("openvino chưa được cài: pip install openvino>=2024.0")
        xml_path = Path(model_dir) / "model.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"OpenVINO model.xml không tìm thấy: {xml_path}")
        core = ov.Core()
        compiled = core.compile_model(str(xml_path), "CPU")
        self._ov_infer = compiled.create_infer_request()
        self._ov_output_key = compiled.output(0)

    # ── inference ─────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Hàm dự đoán nhãn trên một khung hình ảnh."""
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng (empty).")

        img_np = self._preprocess(frame)

        if self._backend == "openvino":
            raw = self._ov_infer.infer({0: img_np})[self._ov_output_key]
            predictions = torch.from_numpy(np.array(raw))
        else:
            tensor = torch.from_numpy(img_np).to(self.model.device)
            tensor = tensor.half() if self.fp16 else tensor
            with torch.inference_mode():
                predictions = self.model(tensor, augment=self.augment)
            predictions = self._unwrap_predictions(predictions)

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

        det[:, :4] = scale_boxes(img_np.shape[2:], det[:, :4], frame.shape).round()

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

    # ── preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Tiền xử lý frame → numpy float32 [1, 3, H, W] chuẩn hóa [0,1]."""
        image = letterbox(frame, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        return image[np.newaxis]

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _derive_openvino_dir(weights_path: str) -> str:
        """Suy ra đường dẫn OpenVINO model dir từ weights path."""
        p = Path(weights_path)
        return str(p.parent / (p.stem + "_openvino"))

    @staticmethod
    def _unwrap_predictions(predictions):
        current = predictions
        for _ in range(4):
            if isinstance(current, torch.Tensor):
                return current
            if isinstance(current, (list, tuple)) and len(current) > 0:
                if isinstance(current[0], torch.Tensor):
                    return current[0]
                current = current[0]
                continue
            break

        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            first = predictions[0]
            if isinstance(first, list) and len(first) > 1 and isinstance(first[1], torch.Tensor):
                return first[1]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(f"Unsupported prediction type for NMS: {type(predictions)}")

    @staticmethod
    def _resolve_path(path: str) -> str:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        return str(path_obj)

    @staticmethod
    def _normalize_model_names(model_names: object) -> Dict[int, str]:
        if isinstance(model_names, dict):
            return {int(key): str(value) for key, value in model_names.items()}
        if isinstance(model_names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(model_names)}
        return {}

    @staticmethod
    def _load_class_names(classes_config_path: str) -> Dict[int, str]:
        config_path = Path(classes_config_path)
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        names = config.get("names", {})
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        return {}
```

- [ ] **Step 2: Chạy tests để xác nhận pass**

```bash
python -m pytest tests/test_openvino_detector.py -v
```

Expected output:
```
tests/test_openvino_detector.py::test_derive_openvino_dir_default PASSED
tests/test_openvino_detector.py::test_derive_openvino_dir_absolute PASSED
tests/test_openvino_detector.py::test_invalid_backend_raises PASSED
tests/test_openvino_detector.py::test_openvino_missing_model_dir_raises PASSED
tests/test_openvino_detector.py::test_preprocess_output_shape PASSED
tests/test_openvino_detector.py::test_preprocess_output_dtype PASSED
tests/test_openvino_detector.py::test_preprocess_normalized PASSED
tests/test_openvino_detector.py::test_openvino_infer_called_not_pytorch PASSED
8 passed
```

- [ ] **Step 3: Chạy toàn bộ test suite để kiểm tra không có regression**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py -x
```

Expected: tất cả test cũ vẫn pass (tracking smoke tests, extrapolator, velocity buffer, v.v.)

- [ ] **Step 4: Commit**

```bash
git add src/detector.py tests/test_openvino_detector.py
git commit -m "feat: thêm OpenVINO backend vào YOLOv9Detector"
```

---

## Task 4: Tạo `tools/export_openvino.py`

**Files:**
- Create: `tools/export_openvino.py`

- [ ] **Step 1: Tạo file export script**

```python
"""
tools/export_openvino.py — Export YOLOv9 weights sang OpenVINO IR.

Sử dụng:
    python3 tools/export_openvino.py --weights weights/best_6k.pt
    python3 tools/export_openvino.py --weights weights/best_6k.pt --int8 --calib-dir assets/videos/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "yolov9") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "yolov9"))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device


def export_onnx(weights_path: Path, output_onnx: Path, imgsz: tuple) -> None:
    print(f"[Step 1/2] PT → ONNX: {output_onnx}")
    device = select_device("cpu")
    model = DetectMultiBackend(str(weights_path), device=device, dnn=False, fp16=False)
    model.eval()

    dummy = torch.zeros(1, 3, *imgsz)
    torch.onnx.export(
        model.model,
        dummy,
        str(output_onnx),
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        verbose=False,
    )
    print(f"  ✓ {output_onnx} ({output_onnx.stat().st_size // 1024 // 1024} MB)")


def export_openvino(onnx_path: Path, output_dir: Path) -> None:
    import openvino as ov
    print(f"[Step 2/2] ONNX → OpenVINO IR: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(output_dir / "model.xml"))
    print(f"  ✓ {output_dir / 'model.xml'}")
    print(f"  ✓ {output_dir / 'model.bin'}")


def export_int8(ov_dir: Path, calib_dir: Path, output_dir: Path) -> None:
    import openvino as ov
    import nncf
    print(f"[INT8] Quantizing từ calibration data: {calib_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    ov_model = core.read_model(str(ov_dir / "model.xml"))

    frames = _collect_calibration_frames(calib_dir, n=100)
    print(f"  Thu thập {len(frames)} calibration frames.")

    dataset = nncf.Dataset(frames, lambda x: {"images": x})
    quantized = nncf.quantize(ov_model, dataset)
    ov.save_model(quantized, str(output_dir / "model.xml"))
    print(f"  ✓ INT8 model: {output_dir / 'model.xml'}")


def _collect_calibration_frames(calib_dir: Path, n: int = 100) -> list:
    import cv2
    frames: list = []
    search_path = PROJECT_ROOT / calib_dir if not calib_dir.is_absolute() else calib_dir

    for p in sorted(search_path.rglob("*")):
        if len(frames) >= n:
            break
        if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            cap = cv2.VideoCapture(str(p))
            while len(frames) < n:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(_preprocess_frame(frame))
            cap.release()
        elif p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            frame = cv2.imread(str(p))
            if frame is not None:
                frames.append(_preprocess_frame(frame))

    if not frames:
        raise RuntimeError(f"Không tìm thấy ảnh/video trong: {search_path}")
    return frames


def _preprocess_frame(frame: np.ndarray, imgsz: tuple = (640, 640)) -> np.ndarray:
    import cv2
    resized = cv2.resize(frame, imgsz)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return arr[np.newaxis]  # [1, 3, H, W]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv9 PT weights → ONNX → OpenVINO IR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Path đến file .pt")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640], metavar=("H", "W"))
    parser.add_argument("--int8", action="store_true", help="Thêm INT8 quantization sau FP32 export")
    parser.add_argument(
        "--calib-dir",
        type=str,
        default="assets/videos/",
        help="Thư mục chứa video/ảnh để calibrate INT8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy weights: {weights_path}")

    imgsz = tuple(args.imgsz)
    onnx_path = weights_path.with_suffix(".onnx")
    ov_dir = weights_path.parent / (weights_path.stem + "_openvino")

    export_onnx(weights_path, onnx_path, imgsz)
    export_openvino(onnx_path, ov_dir)

    if args.int8:
        int8_dir = weights_path.parent / (weights_path.stem + "_openvino_int8")
        export_int8(ov_dir, Path(args.calib_dir), int8_dir)
        print(f"\n[INFO] Chạy INT8: python3 app.py --backend openvino --openvino-model-dir {int8_dir}")

    print(f"\n[INFO] Chạy FP32 OpenVINO:")
    print(f"  python3 app.py --backend openvino")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Kiểm tra syntax**

```bash
python3 -c "import ast; ast.parse(open('tools/export_openvino.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add tools/export_openvino.py
git commit -m "feat: thêm export script PT → ONNX → OpenVINO IR"
```

---

## Task 5: Cập nhật `src/pipeline.py`

**Files:**
- Modify: `src/pipeline.py`

- [ ] **Step 1: Thêm `backend` param vào `BlindSpotPipeline.__init__`**

Tìm signature `__init__` của `BlindSpotPipeline` (dòng ~34) và thêm param:

```python
def __init__(
    self,
    weights_path: str = "weights/best_small.pt",
    roi_config_path: str = "configs/roi.json",
    roi_profile: str = "front_camera",
    classes_config_path: str = "configs/classes.yaml",
    device: str = "",
    image_size: Tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    track_iou_threshold: float = 0.3,
    track_max_misses: int = 5,
    track_min_hits: int = 2,
    track_max_trace_length: int = 30,
    prediction_horizons_s: Optional[List[float]] = None,
    alert_confidence_threshold: float = 0.6,
    backend: str = "pytorch",
    openvino_model_dir: str = "",
) -> None:
```

- [ ] **Step 2: Forward `backend` và `openvino_model_dir` xuống `YOLOv9Detector`**

Tìm block khởi tạo detector (~dòng 51) và sửa:

```python
self.detector = YOLOv9Detector(
    weights_path=weights_path,
    classes_config_path=classes_config_path,
    device=device,
    image_size=image_size,
    conf_threshold=conf_threshold,
    iou_threshold=iou_threshold,
    backend=backend,
    openvino_model_dir=openvino_model_dir,
)
```

- [ ] **Step 3: Chạy tracking smoke test để kiểm tra không có regression**

```bash
python -m pytest tests/test_tracking_smoke.py -v
```

Expected: tất cả pass.

- [ ] **Step 4: Commit**

```bash
git add src/pipeline.py
git commit -m "feat: forward backend param qua BlindSpotPipeline xuống detector"
```

---

## Task 6: Cập nhật `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Thêm `--backend` và `--openvino-model-dir` vào `parse_args()`**

Tìm hàm `parse_args()` và thêm sau arg `--device`:

```python
parser.add_argument(
    "--backend",
    type=str,
    default="pytorch",
    choices=["pytorch", "openvino"],
    help="Backend inference: 'pytorch' (mặc định) hoặc 'openvino' (nhanh hơn trên CPU Intel).",
)
parser.add_argument(
    "--openvino-model-dir",
    type=str,
    default="",
    help="Đường dẫn thư mục OpenVINO model (chứa model.xml). Mặc định tự suy ra từ --weights.",
)
```

- [ ] **Step 2: Pass `backend` và `openvino_model_dir` vào `BlindSpotPipeline`**

Tìm block khởi tạo `pipeline = BlindSpotPipeline(...)` (~dòng 106) và thêm:

```python
pipeline = BlindSpotPipeline(
    weights_path=args.weights,
    roi_config_path=args.roi,
    roi_profile=args.roi_profile,
    classes_config_path=args.classes_config,
    device=args.device,
    conf_threshold=args.conf_thres,
    iou_threshold=args.iou_thres,
    prediction_horizons_s=[args.prediction_horizon],
    alert_confidence_threshold=args.alert_threshold,
    backend=args.backend,
    openvino_model_dir=args.openvino_model_dir,
)
```

- [ ] **Step 3: Kiểm tra syntax và import**

```bash
python3 -c "import app; print('app.py import OK')"
```

Expected: `app.py import OK` (không có ImportError hay SyntaxError).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: thêm --backend và --openvino-model-dir CLI arg vào app.py"
```

---

## Task 7: Export model và đo FPS thực tế

**Files:**
- Generate: `weights/best_6k_openvino/model.xml`, `weights/best_6k_openvino/model.bin`

- [ ] **Step 1: Export FP32 OpenVINO model**

```bash
python3 tools/export_openvino.py --weights weights/best_roiv2.pt
```

Expected output:
```
[Step 1/2] PT → ONNX: weights/best_roiv2.onnx
  ✓ weights/best_roiv2.onnx (XX MB)
[Step 2/2] ONNX → OpenVINO IR: weights/best_roiv2_openvino
  ✓ weights/best_roiv2_openvino/model.xml
  ✓ weights/best_roiv2_openvino/model.bin

[INFO] Chạy FP32 OpenVINO:
  python3 app.py --backend openvino
```

- [ ] **Step 2: Chạy smoke test 50 frame để đo FPS**

```bash
python3 app.py --backend openvino --no-display --source assets/videos/demo4.mp4 2>&1 | head -20
```

Quan sát FPS log. **Nếu FPS ≥ 25**: xong, bỏ qua Step 3–4.

- [ ] **Step 3 (nếu FPS < 25): Export INT8**

```bash
python3 tools/export_openvino.py --weights weights/best_roiv2.pt --int8 --calib-dir assets/videos/
```

Expected: tạo `weights/best_roiv2_openvino_int8/model.xml`

- [ ] **Step 4 (nếu cần INT8): Đo FPS với INT8**

```bash
python3 app.py --backend openvino --openvino-model-dir weights/best_roiv2_openvino_int8 --no-display --source assets/videos/demo4.mp4 2>&1 | head -20
```

- [ ] **Step 5: Chạy full test suite lần cuối**

```bash
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py
```

Expected: tất cả tests pass.

- [ ] **Step 6: Commit cuối**

```bash
git add requirements.txt
git commit -m "docs: cập nhật requirements và hoàn thiện OpenVINO optimization"
```

---

## Tóm Tắt Thay Đổi

| File | Loại | Nội dung |
|------|------|---------|
| `requirements.txt` | Sửa | +openvino, +nncf |
| `src/detector.py` | Sửa | +backend param, +_preprocess(), +_init_openvino(), +_derive_openvino_dir() |
| `src/pipeline.py` | Sửa | +backend, +openvino_model_dir params |
| `app.py` | Sửa | +--backend, +--openvino-model-dir args |
| `tools/export_openvino.py` | Mới | Export PT→ONNX→OpenVINO, optional INT8 |
| `tests/test_openvino_detector.py` | Mới | 8 unit tests cho backend mới |
