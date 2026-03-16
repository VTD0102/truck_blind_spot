from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

try:
    from .detector import Detection, YOLOv9Detector
    from .roi import PolygonROI
    from .visualize import BlindSpotVisualizer
except ImportError:
    from detector import Detection, YOLOv9Detector
    from roi import PolygonROI
    from visualize import BlindSpotVisualizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BlindSpotPipeline:
    def __init__(
        self,
        weights_path: str = "weights/best_small.pt",
        roi_config_path: str = "configs/roi.json",
        classes_config_path: str = "configs/classes.yaml",
        device: str = "",
        image_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.detector = YOLOv9Detector(
            weights_path=weights_path,
            classes_config_path=classes_config_path,
            device=device,
            image_size=image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self.roi = PolygonROI(self._resolve_path(roi_config_path))
        self.visualizer = BlindSpotVisualizer()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        detections = self.detector.predict(frame)
        for detection in detections:
            detection.anchor_point = self.roi.get_reference_point(detection.bbox)
            detection.in_roi = self.roi.contains_point(detection.anchor_point)

        annotated_frame = self.visualizer.draw(
            frame=frame,
            detections=detections,
            roi_polygon=self.roi.points,
            copy=True,
        )
        return annotated_frame, detections

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Detection]]:
        resolved_image_path = self._resolve_path(image_path)
        frame = cv2.imread(resolved_image_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {resolved_image_path}")

        annotated_frame, detections = self.process_frame(frame)
        if output_path:
            cv2.imwrite(self._resolve_path(output_path), annotated_frame)
        return annotated_frame, detections

    def run_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        capture_source = int(source) if str(source).isdigit() else self._resolve_path(source)
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = self._create_writer(cap, output_path)
        window_name = "Blind Spot Inference"

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                annotated_frame, _ = self.process_frame(frame)
                if writer is not None:
                    writer.write(annotated_frame)

                if show:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

    def _create_writer(
        self,
        cap: cv2.VideoCapture,
        output_path: Optional[str],
    ) -> Optional[cv2.VideoWriter]:
        if not output_path:
            return None

        output = self._resolve_path(output_path)
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output, fourcc, fps, (width, height))

    @staticmethod
    def _resolve_path(path: str) -> str:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        return str(path_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv9 blind spot inference pipeline")
    parser.add_argument("--source", type=str, required=True, help="Image path, video path, or webcam index")
    parser.add_argument("--weights", type=str, default="weights/best_small.pt", help="Path to YOLOv9 weights")
    parser.add_argument("--roi", type=str, default="configs/roi.json", help="Path to ROI json config")
    parser.add_argument(
        "--classes-config",
        type=str,
        default="configs/classes.yaml",
        help="Path to classes yaml config",
    )
    parser.add_argument("--device", type=str, default="", help="CUDA device or cpu")
    parser.add_argument("--imgsz", nargs=2, type=int, default=(640, 640), help="Inference image size (h w)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--output", type=str, default=None, help="Optional output image/video path")
    parser.add_argument("--show", action="store_true", help="Display inference result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BlindSpotPipeline(
        weights_path=args.weights,
        roi_config_path=args.roi,
        classes_config_path=args.classes_config,
        device=args.device,
        image_size=tuple(args.imgsz),
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
    )

    source_path = Path(args.source)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        annotated_frame, detections = pipeline.process_image(args.source, args.output)
        if args.show:
            cv2.imshow("Blind Spot Inference", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        for detection in detections:
            print(
                {
                    "bbox": detection.bbox,
                    "confidence": round(detection.confidence, 4),
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "in_roi": detection.in_roi,
                }
            )
        return

    pipeline.run_video(args.source, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
