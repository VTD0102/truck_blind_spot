# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered truck blind spot detection system using YOLOv9-based object detection. Detects objects (persons, bicycles, motorcycles, cars, trucks, buses) entering dangerous zones around a truck and provides real-time visual warnings. This is an ADAS (Advanced Driver Assistance System) demo project.

## Setup & Running

```bash
# Install dependencies (Python 3.11.6 required)
pip install -r requirements.txt

# Run demo (default: front_camera profile, demo4.mp4)
python3 app.py

# Run with specific profile and source
python3 app.py --roi-profile rear_camera --source assets/videos/demo.mp4

# Webcam input
python3 app.py --source 0

# Save output video
python3 app.py --output output/result.mp4 --loop

# Adjust detection thresholds
python3 app.py --conf-thres 0.35 --iou-thres 0.45

# Process a single image via pipeline CLI
python3 src/pipeline.py --source assets/test.jpg --roi-profile front_camera --show
```

Interactive controls while running: `p` to pause/resume, `r` to restart, `q` to quit.

## Architecture

The system is a linear pipeline with three composable stages, orchestrated by `BlindSpotPipeline`:

```
app.py (entry point / video loop)
    └── BlindSpotPipeline (src/pipeline.py)
            ├── YOLOv9Detector (src/detector.py)   → List[Detection]
            ├── MultiPolygonROI (src/roi.py)        → enriches detections with zone/risk info
            └── BlindSpotVisualizer (src/visualize.py) → annotated frame
```

**Per-frame data flow:**
1. `YOLOv9Detector` runs inference on the frame → returns `Detection` dataclasses (bbox, confidence, class_id, class_name)
2. `MultiPolygonROI` checks if each detection's anchor point (bbox bottom-center) falls in a configured polygon zone → adds `in_roi`, `zone_name`, `risk_level` to each Detection
3. `BlindSpotVisualizer` draws semi-transparent zone overlays, color-coded bboxes, and a warning banner if any dangerous objects are detected

## Configuration

**`configs/roi.json`** — defines two camera profiles (`front_camera`, `rear_camera`) each with polygon zones at 1280×720 base resolution. Each zone has a name, risk level (`HIGH`/`MEDIUM`/`WARNING`/`LOW`), color, and polygon coordinates. The ROI class scales coordinates to match actual frame size at runtime.

**`configs/classes.yaml`** — maps 6 class indices to names: person (0), bike (1), motor (2), car (3), truck (4), bus (5).

**Model weights** in `weights/`: `best_6k.pt` is the primary model (YOLOv9-Small, 19 MB).

**YOLOv9 upstream** lives in `yolov9/` — do not modify these files.

## Key Design Decisions

- **Priority-based zone matching**: when a detection falls in overlapping zones, highest risk level wins (HIGH > MEDIUM > WARNING > LOW)
- **Anchor point**: bottom-center of the bounding box is used for zone membership, not the full bbox area
- **Configuration-driven ROI**: all zone shapes and risk levels are in `roi.json`, not hardcoded
- **No training code in `src/`**: training uses the upstream `yolov9/train.py` directly
