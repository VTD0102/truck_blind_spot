"""
app.py – Demo entry point cho Truck Blind Spot Detection.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.pipeline import BlindSpotPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "YOLOv9 Blind Spot Demo"

DEFAULT_SOURCE = str(PROJECT_ROOT / "assets" / "videos" / "demo.mp4")
DEFAULT_WEIGHTS = "weights/best_6k.pt"
DEFAULT_ROI = "configs/roi.json"
DEFAULT_CLASSES = "configs/classes.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truck Blind Spot Detection – demo player",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--roi", type=str, default=DEFAULT_ROI)
    parser.add_argument(
        "--roi-profile",
        type=str,
        default="front_camera",
        choices=["front_camera", "rear_camera"],
    )
    parser.add_argument("--classes-config", type=str, default=DEFAULT_CLASSES)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def draw_overlay(frame, fps: float, paused: bool) -> None:
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
    cap_source: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video: {source!r}")
    return cap


def create_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))


def main() -> None:
    args = parse_args()

    pipeline = BlindSpotPipeline(
        weights_path=args.weights,
        roi_config_path=args.roi,
        roi_profile=args.roi_profile,
        classes_config_path=args.classes_config,
        device=args.device,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
    )

    cap = open_capture(args.source)
    is_file = not args.source.isdigit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = create_writer(cap, args.output)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused = False
    last_frame = None
    smoothed_fps = 0.0

    print(f"[INFO] Source     : {args.source}")
    print(f"[INFO] Frame size : {width}x{height}")
    print(f"[INFO] Weights    : {args.weights}")
    print(f"[INFO] ROI profile: {args.roi_profile}")
    print(f"[INFO] Device     : {args.device or 'auto'}")
    print("[INFO] Phím tắt: [p] Pause  [r] Restart  [q] Thoát")

    try:
        while True:
            if not paused:
                success, frame = cap.read()

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