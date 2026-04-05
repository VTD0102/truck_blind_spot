"""
app.py – Demo entry point cho Truck Blind Spot Detection.

Chạy nhanh (dùng video mặc định):
    python3 app.py

Truyền tham số tùy chỉnh:
    python3 app.py --source assets/videos/demo.mp4 --weights weights/best_small.pt
    python3 app.py --source 0                          # webcam
    python3 app.py --source assets/videos/demo.mp4 --output output/result.mp4 --loop

Phím tắt trong cửa sổ hiển thị:
    p   Pause / Resume
    r   Restart từ đầu video (chỉ có tác dụng với file video)
    q   Thoát
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.pipeline import BlindSpotPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "YOLOv9 Blind Spot Demo"

DEFAULT_SOURCE = str(PROJECT_ROOT / "assets" / "videos" / "demo3.mp4")
DEFAULT_WEIGHTS = "weights/best_small.pt"
DEFAULT_ROI = "configs/roi.json"
DEFAULT_CLASSES = "configs/classes.yaml"

# Helpers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truck Blind Spot Detection – demo player",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE,
        help="Đường dẫn video/ảnh hoặc index webcam (0, 1, …)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Đường dẫn file weights YOLOv9 (.pt)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=DEFAULT_ROI,
        help="Đường dẫn file cấu hình ROI (.json)",
    )
    parser.add_argument(
        "--classes-config",
        type=str,
        default=DEFAULT_CLASSES,
        help="Đường dẫn file cấu hình class (.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device: '' (auto), 'cpu', 'cuda:0', …",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="IoU threshold cho NMS",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file video output (để lưu kết quả). Bỏ qua nếu không muốn lưu.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Tự động replay video khi kết thúc (chỉ áp dụng cho file video)",
    )
    return parser.parse_args()


def draw_overlay(frame, fps: float, paused: bool) -> None:
    """Vẽ FPS và trạng thái lên góc trên-trái của frame."""
    status = "PAUSED" if paused else "RUNNING"
    text = f"FPS: {fps:.1f} | {status}"
    cv2.rectangle(frame, (10, 55), (240, 95), (30, 30, 30), -1)
    cv2.putText(
        frame,
        text,
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def open_capture(source: str) -> cv2.VideoCapture:
    """Mở VideoCapture từ path hoặc webcam index."""
    cap_source: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video: {source!r}")
    return cap


def create_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """Tạo VideoWriter khớp thông số với capture."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))


# Main


def main() -> None:
    args = parse_args()

    pipeline = BlindSpotPipeline(
        weights_path=args.weights,
        roi_config_path=args.roi,
        classes_config_path=args.classes_config,
        device=args.device,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
    )

    cap = open_capture(args.source)
    is_file = not args.source.isdigit()

    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = create_writer(cap, args.output)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused = False
    last_frame = None
    smoothed_fps = 0.0

    print(f"[INFO] Source  : {args.source}")
    print(f"[INFO] Weights : {args.weights}")
    print(f"[INFO] Device  : {args.device or 'auto'}")
    print("[INFO] Phím tắt: [p] Pause  [r] Restart  [q] Thoát")

    try:
        while True:
            if not paused:
                success, frame = cap.read()

                # Hết video
                if not success:
                    if args.loop and is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                t0 = time.perf_counter()
                annotated, _ = pipeline.process_frame(frame)
                elapsed = time.perf_counter() - t0

                current_fps = 1.0 / max(elapsed, 1e-6)
                smoothed_fps = (
                    current_fps
                    if smoothed_fps == 0.0
                    else smoothed_fps * 0.9 + current_fps * 0.1
                )

                draw_overlay(annotated, smoothed_fps, paused=False)
                last_frame = annotated

                if writer is not None:
                    writer.write(annotated)

            # Hiển thị
            if last_frame is None:
                continue

            display = last_frame.copy() if paused else last_frame
            if paused:
                draw_overlay(display, smoothed_fps, paused=True)

            cv2.imshow(WINDOW_NAME, display)

            wait_ms = 30 if paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("r") and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                smoothed_fps = 0.0
                print("[INFO] Restart video.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Video đã lưu tại: {args.output}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
