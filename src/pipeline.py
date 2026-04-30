from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from .common.models import MotionPrediction, Track
    from .detector import Detection, YOLOv9Detector
    from .roi import MultiPolygonROI
    from .tracking.motion import MotionPredictor
    from .tracking.track_manager import TrackManager
    from .visualize import BlindSpotVisualizer
except ImportError:
    from common.models import MotionPrediction, Track  # type: ignore
    from detector import Detection, YOLOv9Detector  # type: ignore
    from roi import MultiPolygonROI  # type: ignore
    from tracking.motion import MotionPredictor  # type: ignore
    from tracking.track_manager import TrackManager  # type: ignore
    from visualize import BlindSpotVisualizer  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

logger = logging.getLogger(__name__)


class PerfStats:
    """Bộ đo thời gian per-stage cho pipeline — dùng để hiển thị breakdown trên frame."""

    __slots__ = ("detection_ms", "roi_ms", "tracking_ms", "prediction_ms", "visualization_ms", "total_ms")

    def __init__(self) -> None:
        self.detection_ms: float = 0.0
        self.roi_ms: float = 0.0
        self.tracking_ms: float = 0.0
        self.prediction_ms: float = 0.0
        self.visualization_ms: float = 0.0
        self.total_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"Det:{self.detection_ms:.1f}ms "
            f"ROI:{self.roi_ms:.1f}ms "
            f"Trk:{self.tracking_ms:.1f}ms "
            f"Pred:{self.prediction_ms:.1f}ms "
            f"Viz:{self.visualization_ms:.1f}ms "
            f"Total:{self.total_ms:.1f}ms"
        )


class BlindSpotPipeline:
    """Pipeline that combines detection, ROI labeling, tracking, and rendering.

    Tối ưu cho real-time ≥ 25 FPS:
    - Cache frame size để skip update_frame_size() khi không đổi
    - In-place rendering (copy=False) mặc định
    - Gộp ROI labeling + prediction thành 1 loop
    - Per-stage timing stats
    - Graceful degradation khi detection fail
    """

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
        target_fps: float = 25.0,
    ) -> None:
        self.detector = YOLOv9Detector(
            weights_path=weights_path,
            classes_config_path=classes_config_path,
            device=device,
            image_size=image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        self.roi = MultiPolygonROI(
            roi_config_path=self._resolve_path(roi_config_path),
            profile_name=roi_profile,
        )
        self.track_manager = TrackManager(
            iou_threshold=track_iou_threshold,
            max_misses=track_max_misses,
            min_hits=track_min_hits,
            max_trace_length=track_max_trace_length,
        )
        self.visualizer = BlindSpotVisualizer()
        # Khởi tạo bộ dự đoán chuyển động với các mốc thời gian mặc định
        self.motion_predictor = MotionPredictor(
            prediction_horizons_s=prediction_horizons_s or [0.5, 1.0, 2.0],
            fps=30.0,
            alert_confidence_threshold=alert_confidence_threshold,
        )
        # Cache dự đoán mới nhất để app.py có thể truy cập cảnh báo
        self.last_predictions: Optional[Dict[int, MotionPrediction]] = None

        # Performance tracking
        self.target_fps = target_fps
        self.perf: PerfStats = PerfStats()

        # Cache frame size — chỉ gọi update_frame_size() khi thay đổi
        self._cached_frame_size: Tuple[int, int] = (0, 0)

        # Cache detections gần nhất cho frame skipping
        self._last_detections: List[Detection] = []

    def process_frame(
        self,
        frame: np.ndarray,
        skip_visualization: bool = False,
        skip_detection: bool = False,
    ) -> Tuple[np.ndarray, List[Detection], List[Track]]:
        """Xử lý một frame qua toàn bộ pipeline.

        Args:
            frame:              Frame BGR từ OpenCV.
            skip_visualization: Bỏ qua render (cho benchmark).
            skip_detection:     Bỏ qua inference, dùng Kalman predict-only (frame skipping).

        Returns:
            (annotated_frame, detections, tracks)
        """
        t_total = time.perf_counter()

        frame_height, frame_width = frame.shape[:2]
        # Chỉ cập nhật ROI khi frame size thay đổi
        if (frame_width, frame_height) != self._cached_frame_size:
            self.roi.update_frame_size(frame_width, frame_height)
            self._cached_frame_size = (frame_width, frame_height)

        # ── Detection ──────────────────────────────────────────
        t0 = time.perf_counter()
        if skip_detection:
            # Frame skipping: tái sử dụng detections trước đó
            detections = self._last_detections
        else:
            try:
                detections = self.detector.predict(frame)
            except Exception:
                # Graceful degradation: detection fail → dùng list rỗng
                logger.warning("Detection failed, sử dụng Kalman predict-only cho frame này.")
                detections = []
            self._last_detections = detections
        self.perf.detection_ms = (time.perf_counter() - t0) * 1000

        # ── ROI labeling cho detections ────────────────────────
        t0 = time.perf_counter()
        for detection in detections:
            detection.anchor_point = self.roi.get_reference_point(detection.bbox)
            zone = self.roi.get_zone(detection.anchor_point)
            detection.in_roi = zone is not None
            detection.zone_name = zone.name if zone is not None else None
            detection.risk_level = zone.risk_level if zone is not None else None
        self.perf.roi_ms = (time.perf_counter() - t0) * 1000

        # ── Tracking ──────────────────────────────────────────
        t0 = time.perf_counter()
        tracks = self.track_manager.update(detections)
        self.perf.tracking_ms = (time.perf_counter() - t0) * 1000

        # ── ROI labeling + Motion Prediction (gộp 1 loop) ─────
        t0 = time.perf_counter()
        current_time = time.time()
        predictions: Dict[int, MotionPrediction] = {}
        for track in tracks:
            # ROI labeling cho track
            track.anchor_point = self.roi.get_reference_point(self._round_bbox(track.bbox))
            zone = self.roi.get_zone(track.anchor_point) if track.anchor_point else None
            track.in_roi = zone is not None
            track.zone_name = zone.name if zone is not None else None
            track.risk_level = zone.risk_level if zone is not None else None

            # Motion prediction
            try:
                predictions[track.track_id] = self.motion_predictor.update(track, current_time)
            except Exception:
                logger.debug("Motion prediction failed cho track %d", track.track_id)
        self.last_predictions = predictions
        self.perf.prediction_ms = (time.perf_counter() - t0) * 1000

        # ── Visualization ─────────────────────────────────────
        t0 = time.perf_counter()
        if skip_visualization:
            annotated_frame = frame
        else:
            # In-place rendering: không copy frame để tiết kiệm ~1ms
            annotated_frame = self.visualizer.draw(
                frame=frame,
                detections=detections,
                roi_zones=self.roi.zones,
                copy=False,
                tracks=tracks,
                predictions=predictions,
            )
        self.perf.visualization_ms = (time.perf_counter() - t0) * 1000

        self.perf.total_ms = (time.perf_counter() - t_total) * 1000
        return annotated_frame, detections, tracks

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[Detection], List[Track]]:
        resolved_image_path = self._resolve_path(image_path)
        frame = cv2.imread(resolved_image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {resolved_image_path}")

        annotated_frame, detections, tracks = self.process_frame(frame)
        if output_path:
            output_resolved = self._resolve_path(output_path)
            Path(output_resolved).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_resolved, annotated_frame)
        return annotated_frame, detections, tracks

    def run_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        capture_source = int(source) if str(source).isdigit() else self._resolve_path(source)
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        writer = self._create_writer(cap, output_path)
        window_name = "Blind Spot Inference"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Video frame size: %dx%d", width, height)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                annotated_frame, _, _ = self.process_frame(frame)
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

    @staticmethod
    def _round_bbox(
        bbox: Tuple[float, float, float, float],
    ) -> Tuple[int, int, int, int]:
        return tuple(int(round(value)) for value in bbox)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run blind-spot detection, ROI labeling, and tracking with YOLOv9."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path, video path, or webcam index.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best_small.pt",
        help="Path to YOLOv9 weights.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="configs/roi.json",
        help="Path to ROI configuration (.json).",
    )
    parser.add_argument(
        "--roi-profile",
        type=str,
        default="front_camera",
        choices=["front_camera", "rear_camera"],
        help="ROI profile to use.",
    )
    parser.add_argument(
        "--classes-config",
        type=str,
        default="configs/classes.yaml",
        help="Path to class label configuration (.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device or CPU.",
    )
    parser.add_argument(
        "--imgsz",
        nargs=2,
        type=int,
        default=(640, 640),
        help="Inference image size as height width.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="Detector NMS IoU threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image/video path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display rendered output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BlindSpotPipeline(
        weights_path=args.weights,
        roi_config_path=args.roi,
        roi_profile=args.roi_profile,
        classes_config_path=args.classes_config,
        device=args.device,
        image_size=tuple(args.imgsz),
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
    )

    source_path = Path(args.source)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        annotated_frame, detections, tracks = pipeline.process_image(args.source, args.output)
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
                    "zone_name": detection.zone_name,
                    "risk_level": detection.risk_level,
                    "anchor_point": detection.anchor_point,
                }
            )

        for track in tracks:
            print(
                {
                    "track_id": track.track_id,
                    "bbox": tuple(round(value, 2) for value in track.bbox),
                    "velocity": (
                        tuple(round(value, 2) for value in track.velocity)
                        if track.velocity is not None
                        else None
                    ),
                    "hits": track.hits,
                    "misses": track.misses,
                    "status": track.status.value,
                }
            )
        return

    pipeline.run_video(args.source, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
