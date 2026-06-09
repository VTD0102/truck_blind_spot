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

# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov9"

# Thêm đường dẫn YOLOv9 vào sys.path để import các module nội bộ của nó
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

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
    ) -> None:
        self.weights_path = self._resolve_path(weights_path)
        self.classes_config_path = self._resolve_path(classes_config_path)
        self.device = select_device(device)
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.classes = list(classes) if classes is not None else None
        self.augment = augment
        self.agnostic_nms = agnostic_nms

        # Tải tên các lớp từ tập tin yaml
        self.class_names = self._load_class_names(self.classes_config_path)

        # Auto-detect FP16: bật mặc định trên GPU CUDA để tăng tốc ~30%
        use_fp16 = half or (self.device.type != "cpu")
        
        # Nạp mô hình YOLOv9 backend
        self.model = DetectMultiBackend(
            self.weights_path,
            device=self.device,
            dnn=dnn,
            data=None,
            fp16=use_fp16 and self.device.type != "cpu",
        )
        self.stride = self.model.stride
        self.pt = self.model.pt
        self.imgsz = check_img_size(self.image_size, s=self.stride)
        self.fp16 = bool(self.model.fp16)

        if not self.class_names:
            self.class_names = self._normalize_model_names(self.model.names)

        # Chạy warmup cho model
        self.model.warmup(
            imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz)
        )

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Hàm dự đoán nhãn trên một khung hình ảnh.

        Lưu ý: Không copy frame — letterbox tạo array mới, scale_boxes chỉ đọc shape.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Khung hình đầu vào rỗng (empty).")

        # Đẩy dữ liệu lên GPU/CPU
        img = self._preprocess(frame)
        tensor = torch.from_numpy(img).to(self.device)
        if self.fp16:
            tensor = tensor.half()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Thực hiện suy luận (inference)
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

        # Đưa các bbox về tọa độ của ảnh gốc (chỉ đọc frame.shape, không mutate)
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()

        for *xyxy, confidence, class_id in det.tolist():
            class_index = int(class_id)
            x1, y1, x2, y2 = [int(value) for value in xyxy]
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=class_index,
                    class_name=self.class_names.get(class_index, str(class_index)),
                )
            )

        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox + CHW transpose + BGR→RGB + normalize float32 [0, 1]."""
        img = letterbox(frame, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        return np.ascontiguousarray(img).astype(np.float32) / 255.0

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
        """Xử lý đường dẫn tuyệt đối/tương đối."""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        return str(path_obj)

    @staticmethod
    def _normalize_model_names(model_names: object) -> Dict[int, str]:
        """Chuẩn hóa dictionary lưu tên của class."""
        if isinstance(model_names, dict):
            return {int(key): str(value) for key, value in model_names.items()}
        if isinstance(model_names, (list, tuple)):
            return {index: str(name) for index, name in enumerate(model_names)}
        return {}

    @staticmethod
    def _load_class_names(classes_config_path: str) -> Dict[int, str]:
        """Đọc và lấy tên class từ tệp cấu hình yaml."""
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
