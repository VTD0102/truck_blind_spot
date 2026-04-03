from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov9"

if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    in_roi: bool = False
    anchor_point: Optional[Tuple[int, int]] = None


class YOLOv9Detector:
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

        self.class_names = self._load_class_names(self.classes_config_path)
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

        self.model.warmup(
            imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz)
        )

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        original_frame = frame.copy()
        image = letterbox(
            original_frame,
            new_shape=self.imgsz,
            stride=self.stride,
            auto=self.pt,
        )[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)

        tensor = torch.from_numpy(image).to(self.model.device)
        tensor = tensor.half() if self.fp16 else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with torch.inference_mode():
            predictions = self.model(tensor, augment=self.augment)

        predictions = (
            predictions[0][1] if isinstance(predictions[0], list) else predictions[0]
        )
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

        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], original_frame.shape).round()
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
