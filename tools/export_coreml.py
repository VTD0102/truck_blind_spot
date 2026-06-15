"""
tools/export_coreml.py — Export YOLOv9 weights sang CoreML (.mlpackage).

Chạy một lần trên máy macOS Apple Silicon:
    python3 tools/export_coreml.py --weights weights/best_roiv2.pt

Output:
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

    mlpackage_path = weights_path.with_suffix(".mlpackage")

    # ── Bước 1: Load model trên CPU ──────────────────────────────────────
    logger.info("Load model từ %s ...", weights_path)
    device = select_device("cpu")
    model = DetectMultiBackend(str(weights_path), device=device, dnn=False, fp16=False)

    # Bật cờ export trên tất cả Detect heads để bỏ decoder khỏi graph
    for m in model.modules():
        if hasattr(m, "export"):
            m.export = True

    model.eval()

    dummy = torch.zeros(1, 3, *imgsz)

    # ── Bước 2: PT → TorchScript ─────────────────────────────────────────
    logger.info("Trace model sang TorchScript ...")
    with torch.no_grad():
        # Warmup: chạy 1 lần để DDetect pre-compute self.anchors + self.shape.
        # Sau warmup, nhánh make_anchors() sẽ bị skip khi trace (shape đã khớp)
        # → loại bỏ int() cast của anchor_generator khỏi TorchScript graph.
        _ = model(dummy)
        # check_trace=False: YOLOv9 anchor generator dùng dynamic control flow
        # gây false-positive sanity check; trace vẫn đúng với input cố định 640×640
        traced_model = torch.jit.trace(model, dummy, strict=False, check_trace=False)
    logger.info("TorchScript trace thành công.")

    # ── Bước 3: TorchScript → CoreML ─────────────────────────────────────
    # coremltools 8+ không còn hỗ trợ ONNX trực tiếp; dùng pytorch source
    import coremltools as ct  # lazy import — tránh lỗi khi chạy --help

    logger.info("Convert sang CoreML: %s ...", mlpackage_path)
    coreml_model = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(shape=dummy.shape, name="images")],
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
    parser = argparse.ArgumentParser(description="Export YOLOv9 PT → TorchScript → CoreML")
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
