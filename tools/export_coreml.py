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

import torch

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_coreml")


def export(weights: str, imgsz: tuple[int, int] = (640, 640)) -> None:
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights không tồn tại: {weights_path}")

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
    import coremltools as ct  # lazy import — tránh lỗi khi chạy --help
    import onnx               # lazy import — tránh lỗi khi chạy --help

    logger.info("Convert sang CoreML: %s ...", mlpackage_path)
    onnx_model = onnx.load(str(onnx_path))
    coreml_model = ct.convert(
        onnx_model,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS15,
    )
    coreml_model.save(str(mlpackage_path))

    spec = coreml_model.get_spec()
    out_key = spec.description.output[0].name
    in_key = spec.description.input[0].name
    logger.info("CoreML export thành công.")
    logger.info("Input key: %r  Output key: %r", in_key, out_key)
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
