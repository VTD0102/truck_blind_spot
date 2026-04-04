from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from .detector import Detection, YOLOv9Detector
    from .roi import PolygonROI
    from .visualize import BlindSpotVisualizer
except ImportError:
    from detector import Detection, YOLOv9Detector
    from roi import PolygonROI
    from visualize import BlindSpotVisualizer

# Xác định thư mục root của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BlindSpotPipeline:
    """Đường ống (Pipeline) kết hợp giữa nhận diện và phân tích điểm mù."""
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
        # Khởi tạo mô hình nhận diện
        self.detector = YOLOv9Detector(
            weights_path=weights_path,
            classes_config_path=classes_config_path,
            device=device,
            image_size=image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        # Khởi tạo thuật toán phân tích vùng ROI
        self.roi = PolygonROI(self._resolve_path(roi_config_path))
        # Công cụ vẽ báo cáo trực quan
        self.visualizer = BlindSpotVisualizer()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """Xử lý từng khung hình ảnh cụ thể."""
        # Chạy inference
        detections = self.detector.predict(frame)
        for detection in detections:
            # Lấy vị trí tâm/đáy của bbox
            detection.anchor_point = self.roi.get_reference_point(detection.bbox)
            # Kiểm tra xem có lọt vào ROI không
            detection.in_roi = self.roi.contains_point(detection.anchor_point)

        # Tạo ảnh đầu ra đã viền bbox
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
        """Đọc và xử lý đối với tệp hình ảnh."""
        resolved_image_path = self._resolve_path(image_path)
        frame = cv2.imread(resolved_image_path)
        if frame is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {resolved_image_path}")

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
        """Chạy pipeline với video đầu vào (tệp tin hoặc webcam)."""
        capture_source = int(source) if str(source).isdigit() else self._resolve_path(source)
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở nguồn video: {source}")

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

                # Hiện hình ảnh ra màn hình
                if show:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # Nhất Esc hoặc Q để thoát
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
        """Trình quản lý đối tượng ghi video kết quả."""
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
        """Chuẩn hóa đường dẫn tương đối thành tuyệt đối."""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / path_obj
        return str(path_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline nhận diện điểm mù trên xe tải dùng YOLOv9")
    parser.add_argument("--source", type=str, required=True, help="Đường dẫn file ảnh, video hoặc số tham chiếu webcam")
    parser.add_argument("--weights", type=str, default="weights/best_small.pt", help="Đường dẫn đến file weights YOLOv9")
    parser.add_argument("--roi", type=str, default="configs/roi.json", help="Đường dẫn file cấu hình ROI (.json)")
    parser.add_argument(
        "--classes-config",
        type=str,
        default="configs/classes.yaml",
        help="Đường dẫn file thiết lập tên các Class (.yaml)",
    )
    parser.add_argument("--device", type=str, default="", help="Card đồ họa (CUDA) hoặc cpu")
    parser.add_argument("--imgsz", nargs=2, type=int, default=(640, 640), help="Kích thước nhận diện (cao rộng)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Ngưỡng độ tin cậy (Confidence threshold)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Ngưỡng bóc tách NMS (IoU threshold)")
    parser.add_argument("--output", type=str, default=None, help="Vị trí lưu file ảnh/video đầu ra tùy chọn")
    parser.add_argument("--show", action="store_true", help="Có hiển thị kết quả ra màn hình hay không")
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
    # Xử lý trường hợp đối với ảnh
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        annotated_frame, detections = pipeline.process_image(args.source, args.output)
        if args.show:
            cv2.imshow("Kq Nhận diện điểm mù", annotated_frame)
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

    # Nếu không phải hình ảnh, tiến hành đọc video/webcam
    pipeline.run_video(args.source, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
