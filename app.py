"""
app.py – Demo entry point cho Truck Blind Spot Detection.

Phase 3: Tối ưu pipeline cho real-time ≥ 25 FPS.
- Frame skipping (Kalman predict-only) khi FPS thấp
- Adaptive quality scaling (giảm resolution tự động)
- Performance monitoring overlay (timing breakdown)
- Alert logging ra file CSV
- Graceful shutdown + session statistics
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.pipeline import BlindSpotPipeline

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ─── Constants ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "YOLOv9 Blind Spot Demo"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
OVERLAY_FONT_SCALE = 0.5
OVERLAY_TEXT_THICKNESS = 1

DEFAULT_SOURCE = str(PROJECT_ROOT / "assets" / "videos" / "demo4.mp4")
DEFAULT_WEIGHTS = "weights/best_roiv2.pt"
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
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Chạy xử lý video không mở cửa sổ OpenCV, phù hợp môi trường headless/Colab.",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=float,
        default=1.0,
        help="Mốc thời gian dự đoán chuyển động tính bằng giây (ví dụ: 0.5, 1.0, 2.0).",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.6,
        help="Ngưỡng confidence tối thiểu để kích hoạt cảnh báo chuyển động.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="FPS mục tiêu — bật frame skipping khi không đạt.",
    )
    parser.add_argument(
        "--alert-log",
        type=str,
        default=None,
        help="Đường dẫn file CSV ghi log cảnh báo (nếu không set thì chỉ log console).",
    )
    parser.add_argument(
        "--enable-frame-skip",
        action="store_true",
        help="Bật frame skipping: skip detection + dùng Kalman predict-only khi FPS thấp.",
    )
    parser.add_argument(
        "--enable-adaptive-scale",
        action="store_true",
        help="Bật adaptive quality scaling: tự giảm resolution khi FPS < target.",
    )
    return parser.parse_args()


# ─── Alert Logger ────────────────────────────────────────────────────────────


class AlertLogger:
    """Ghi log cảnh báo ra file CSV và console."""

    def __init__(self, csv_path: Optional[str] = None) -> None:
        self._csv_file = None
        self._writer = None
        self.alert_counts = {"high": 0, "medium": 0, "low": 0}

        if csv_path:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow([
                "timestamp", "frame_idx", "track_id", "alert_level",
                "zone", "confidence", "velocity_x", "velocity_y",
            ])
            logger.info("Alert log CSV: %s", csv_path)

    def log(
        self,
        frame_idx: int,
        track_id: int,
        alert_level: str,
        zone: str,
        confidence: float,
        velocity: tuple,
    ) -> None:
        self.alert_counts[alert_level] = self.alert_counts.get(alert_level, 0) + 1

        logger.warning(
            "ALERT Track %d: %s risk | zone=%s | conf=%.2f",
            track_id, alert_level.upper(), zone, confidence,
        )

        if self._writer:
            vx, vy = velocity
            self._writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                frame_idx, track_id, alert_level,
                zone, f"{confidence:.4f}", f"{vx:.2f}", f"{vy:.2f}",
            ])

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()

    def summary(self) -> str:
        parts = []
        for level in ("high", "medium", "low"):
            count = self.alert_counts.get(level, 0)
            if count > 0:
                parts.append(f"{count} {level}")
        return ", ".join(parts) if parts else "none"


# ─── Overlay Drawing ─────────────────────────────────────────────────────────


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    paused: bool,
    perf_summary: str = "",
    scale_factor: float = 1.0,
    frame_skipping: bool = False,
) -> None:
    """Vẽ overlay thông tin hiệu năng lên frame."""
    status = "PAUSED" if paused else "RUNNING"
    line1 = f"FPS: {fps:.1f} | {status}"

    # Dòng 1: FPS + status
    cv2.rectangle(frame, (10, 55), (190, 86), (30, 30, 30), -1)
    cv2.putText(
        frame, line1, (18, 77),
        cv2.FONT_HERSHEY_SIMPLEX, OVERLAY_FONT_SCALE,
        (255, 255, 255), OVERLAY_TEXT_THICKNESS, cv2.LINE_AA,
    )

    # Dòng 2: Timing breakdown
    if perf_summary:
        text_w = max(400, len(perf_summary) * 7 + 20)
        cv2.rectangle(frame, (10, 88), (10 + text_w, 112), (30, 30, 30), -1)
        cv2.putText(
            frame, perf_summary, (18, 106),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (180, 255, 180), 1, cv2.LINE_AA,
        )

    # Dòng 3: Adaptive indicators
    indicators = []
    if scale_factor < 1.0:
        indicators.append(f"Scale:{scale_factor:.0%}")
    if frame_skipping:
        indicators.append("SKIP")
    if indicators:
        ind_text = " | ".join(indicators)
        cv2.rectangle(frame, (10, 114), (200, 138), (30, 30, 80), -1)
        cv2.putText(
            frame, ind_text, (18, 132),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (100, 180, 255), 1, cv2.LINE_AA,
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────


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


# ─── Main ────────────────────────────────────────────────────────────────────


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
        prediction_horizons_s=[args.prediction_horizon],
        alert_confidence_threshold=args.alert_threshold,
        target_fps=args.target_fps,
    )

    cap = open_capture(args.source)
    is_file = not args.source.isdigit()

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    if args.output:
        writer = create_writer(cap, args.output)

    display_enabled = not args.no_display
    if display_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    # Session state
    paused = False
    last_frame = None
    smoothed_fps = 0.0
    frame_idx = 0
    total_frames = 0
    dropped_frames = 0
    scale_factor = 1.0
    is_frame_skipping = False

    # Target timing
    target_frame_ms = 1000.0 / args.target_fps

    # Alert logger
    alert_logger = AlertLogger(csv_path=args.alert_log)

    logger.info("Source     : %s", args.source)
    logger.info("Frame size : %dx%d", orig_width, orig_height)
    logger.info("Weights    : %s", args.weights)
    logger.info("ROI profile: %s", args.roi_profile)
    logger.info("Device     : %s", args.device or "auto")
    logger.info("Target FPS : %.0f", args.target_fps)
    if args.enable_frame_skip:
        logger.info("Frame skipping: ENABLED")
    if args.enable_adaptive_scale:
        logger.info("Adaptive scaling: ENABLED")
    if display_enabled:
        logger.info("Phím tắt: [p] Pause  [r] Restart  [q] Thoát")
    else:
        logger.info("Chế độ no-display: xử lý hết nguồn video rồi thoát.")

    session_start = time.perf_counter()

    try:
        while True:
            if not paused:
                success, frame = cap.read()

                if not success:
                    if args.loop and is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                total_frames += 1
                frame_idx += 1

                # ── Adaptive quality scaling ──────────────────────
                if args.enable_adaptive_scale and smoothed_fps > 0:
                    if smoothed_fps < args.target_fps * 0.8:
                        scale_factor = max(0.5, scale_factor - 0.05)
                    elif smoothed_fps > args.target_fps * 1.1 and scale_factor < 1.0:
                        scale_factor = min(1.0, scale_factor + 0.02)

                if scale_factor < 1.0:
                    frame = cv2.resize(
                        frame, None,
                        fx=scale_factor, fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR,
                    )

                # ── Frame skipping logic ──────────────────────────
                skip_detection = False
                if args.enable_frame_skip and smoothed_fps > 0:
                    if smoothed_fps < args.target_fps * 0.7:
                        # Skip detection mỗi 2 frame khi FPS quá thấp
                        skip_detection = (frame_idx % 2 != 0)
                        if skip_detection:
                            dropped_frames += 1

                is_frame_skipping = skip_detection

                t0 = time.perf_counter()
                annotated, _, tracks = pipeline.process_frame(
                    frame,
                    skip_detection=skip_detection,
                )
                elapsed = time.perf_counter() - t0

                # ── Resize output về kích thước gốc nếu đã scale ─
                if scale_factor < 1.0 and writer is not None:
                    annotated = cv2.resize(
                        annotated,
                        (orig_width, orig_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                # ── Alert logging ─────────────────────────────────
                for track_id, pred in (pipeline.last_predictions or {}).items():
                    if pred.alert_level in ("medium", "high"):
                        zone = next(
                            (t.zone_name for t in tracks if t.track_id == track_id),
                            "?",
                        )
                        alert_logger.log(
                            frame_idx=frame_idx,
                            track_id=track_id,
                            alert_level=pred.alert_level,
                            zone=zone or "?",
                            confidence=pred.overall_confidence,
                            velocity=pred.velocity_px_per_s,
                        )

                # ── FPS calculation ───────────────────────────────
                current_fps = 1.0 / max(elapsed, 1e-6)
                smoothed_fps = (
                    current_fps
                    if smoothed_fps == 0.0
                    else smoothed_fps * 0.9 + current_fps * 0.1
                )

                draw_overlay(
                    annotated,
                    smoothed_fps,
                    paused=False,
                    perf_summary=pipeline.perf.summary(),
                    scale_factor=scale_factor,
                    frame_skipping=is_frame_skipping,
                )
                last_frame = annotated

                if writer is not None:
                    writer.write(annotated)

            if last_frame is None:
                continue

            if display_enabled:
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
                    frame_idx = 0
                    scale_factor = 1.0
                    logger.info("Restart video.")

    finally:
        session_elapsed = time.perf_counter() - session_start

        cap.release()
        if writer is not None:
            writer.release()
            logger.info("Video đã lưu tại: %s", args.output)
        if display_enabled:
            cv2.destroyAllWindows()

        alert_logger.close()

        # ── Session statistics ────────────────────────────────
        avg_fps = total_frames / session_elapsed if session_elapsed > 0 else 0.0
        logger.info("=" * 50)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info("Total frames  : %d", total_frames)
        logger.info("Session time  : %.1fs", session_elapsed)
        logger.info("Average FPS   : %.1f", avg_fps)
        logger.info("Dropped frames: %d", dropped_frames)
        logger.info("Alerts        : %s", alert_logger.summary())
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
