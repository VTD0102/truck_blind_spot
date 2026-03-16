from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.pipeline import BlindSpotPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "videos" / "test_video.mp4"
WINDOW_NAME = "YOLOv9 Blind Spot Demo"


def draw_fps(frame, fps: float, paused: bool) -> None:
    """Draw FPS and player status on the output frame."""
    status = "PAUSED" if paused else "RUNNING"
    text = f"FPS: {fps:.2f} | {status}"

    cv2.rectangle(frame, (10, 55), (230, 95), (40, 40, 40), -1)
    cv2.putText(
        frame,
        text,
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    """Run blind spot inference on a local demo video."""
    pipeline = BlindSpotPipeline(
        weights_path="weights/best_small.pt",
        roi_config_path="configs/roi.json",
        classes_config_path="configs/classes.yaml",
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    # Load video from the predefined demo path.
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused = False
    last_rendered_frame = None
    smoothed_fps = 0.0

    try:
        while True:
            if not paused:
                success, frame = cap.read()
                if not success:
                    break

                start_time = time.perf_counter()
                annotated_frame, _ = pipeline.process_frame(frame)
                inference_time = time.perf_counter() - start_time

                current_fps = 1.0 / max(inference_time, 1e-6)
                if smoothed_fps == 0.0:
                    smoothed_fps = current_fps
                else:
                    smoothed_fps = (smoothed_fps * 0.9) + (current_fps * 0.1)

                draw_fps(annotated_frame, smoothed_fps, paused=False)
                last_rendered_frame = annotated_frame

            if last_rendered_frame is None:
                continue

            if paused:
                paused_frame = last_rendered_frame.copy()
                draw_fps(paused_frame, smoothed_fps, paused=True)
                cv2.imshow(WINDOW_NAME, paused_frame)
            else:
                cv2.imshow(WINDOW_NAME, last_rendered_frame)

            key = cv2.waitKey(1 if not paused else 30) & 0xFF

            if key == ord("q"):
                break

            if key == ord("p"):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
