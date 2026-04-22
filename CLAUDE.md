# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Truck Blind Spot Detection** is a real-time object detection system built on **YOLOv9** to identify objects (persons, bicycles, cars, motorcycles) in the blind spot zones of trucks. The system is designed as an ADAS (Advanced Driver Assistance System) safety feature that provides visual alerts when dangerous objects enter a configurable Region of Interest (ROI).

### Key Features
- Fine-tuned YOLOv9-Small model (mAP@0.5 = 0.70)
- ROI-based blind spot detection with polygon geometry
- Real-time inference with FPS monitoring
- Multi-source input support (video files, webcam, images)
- Video output with annotations
- Configurable thresholds and device selection (CPU/GPU)

---

## Architecture & Code Structure

The pipeline is modular with clear separation of concerns:

```
Input Frame
    ↓
YOLOv9Detector (src/detector.py)
  - Load model weights from .pt file
  - Preprocess: letterbox resizing, BGR→CHW conversion
  - Inference on frame
  - Apply NMS filtering
  - Scale bboxes back to original frame dimensions
    ↓
Detection[] (dataclass with bbox, conf, class_id, in_roi flag)
    ↓
PolygonROI (src/roi.py)
  - Parse roi.json polygon configuration
  - Calculate anchor_point (bbox_bottom_center) for each detection
  - Test point-in-polygon membership
  - Mark detection.in_roi = True/False
    ↓
BlindSpotVisualizer (src/visualize.py)
  - Draw ROI polygon overlay
  - Draw bboxes: green (safe) or red (in blind spot)
  - Render detection labels and "BLIND SPOT" warnings
  - Add FPS counter and status text
    ↓
Output Frame (annotated)
```

### Core Modules

| File | Purpose |
|------|---------|
| **app.py** | Main CLI entry point. Handles video capture, frame loop, keyboard controls (pause/resume/restart), video writer, FPS smoothing |
| **src/detector.py** | `YOLOv9Detector` wrapper. Loads YOLOv9 model via `DetectMultiBackend`, preprocesses frames, runs inference, applies NMS. Uses YOLOv9 utils for letterbox and scale_boxes |
| **src/roi.py** | `PolygonROI` class. Parses roi.json (polygon + check_point strategy), implements point-in-polygon test |
| **src/visualize.py** | `BlindSpotVisualizer` class. Renders annotations on frames (ROI polygon, bboxes, labels, warnings) |
| **src/pipeline.py** | `BlindSpotPipeline` orchestrator. Chains detector → ROI check → visualizer. Also supports standalone image/video processing with CLI |
| **yolov9/** | Upstream YOLOv9 source. Used for model loading and inference utilities |

### Configuration Files

- **configs/classes.yaml** — Class name mappings (person, bicycle, car, motorcycle)
- **configs/roi.json** — Polygon vertices and check_point strategy for blind spot region
- **weights/best_small.pt** — Fine-tuned YOLOv9-Small weights (20.3 MB)

---

## Common Commands

### Development & Testing

```bash
# Run demo with default video
python3 app.py

# Process custom video, save output
python3 app.py --source assets/videos/demo.mp4 --output output/result.mp4

# Use webcam (index 0)
python3 app.py --source 0

# Auto-loop video until interrupted
python3 app.py --source assets/videos/demo.mp4 --loop

# Process single image
python3 src/pipeline.py --source path/to/image.jpg --show

# Process video via pipeline (CLI alternative to app.py)
python3 src/pipeline.py --source assets/videos/demo.mp4 --output output.mp4 --show

# Adjust detection thresholds
python3 app.py --conf-thres 0.3 --iou-thres 0.5

# Force CPU inference
python3 app.py --device cpu

# Use GPU (auto-detected if available)
python3 app.py --device cuda:0
```

### Keyboard Controls (during playback)

| Key | Action |
|-----|--------|
| `p` | Pause/Resume |
| `r` | Restart video from beginning (file input only) |
| `q` | Quit |
| `Esc` | Quit (pipeline.py mode) |

### Virtual Environment

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Key Dependencies & Integration Points

### Deep Learning Stack
- **PyTorch** (≥2.0) — Model inference backend
- **YOLOv9** — Detection model (source code in `yolov9/`)
  - Imports: `DetectMultiBackend`, `letterbox`, `non_max_suppression`, `scale_boxes`, `select_device`
  - Located in: `yolov9/models/common.py`, `yolov9/utils/augmentations.py`, `yolov9/utils/general.py`, `yolov9/utils/torch_utils.py`

### Computer Vision
- **OpenCV** (≥4.8.0) — Video capture, frame I/O, drawing primitives (rectangles, polygons, text)
- **NumPy** — Array operations and geometry

### Configuration & Utilities
- **PyYAML** — Parse `configs/classes.yaml`
- **Python pathlib** — Cross-platform path handling (used throughout)

### Important Pattern: sys.path Injection
`src/detector.py` injects YOLOv9 into sys.path to enable direct imports of YOLOv9 modules. Any new code that needs YOLOv9 utils should follow this pattern or import from detector.py.

---

## Development Patterns & Conventions

### Path Handling
All modules use `Path(__file__).resolve().parents[N]` to locate PROJECT_ROOT, then resolve relative paths from there. This ensures portability regardless of where the script is invoked from. Use `_resolve_path()` static method or the `PROJECT_ROOT` constant.

### Device Selection
Device selection is delegated to YOLOv9's `select_device()` utility. Empty string `""` means auto-detect GPU if available; otherwise use explicit `"cpu"`, `"cuda:0"`, etc.

### Dataclass Usage
`Detection` is a dataclass that flows through the pipeline. Mutable fields like `in_roi` and `anchor_point` are set as the detection moves through ROI and visualization stages.

### Error Handling
- File/path errors raise `RuntimeError` or `FileNotFoundError` with descriptive messages
- Video capture failures explicitly check `cap.isOpened()`
- Empty/invalid frames are validated before processing

### Code Comments
Comments in Vietnamese (matching the project language). Complex logic is explained inline; trivial operations are left uncommented.

---

## Testing & Debugging Tips

### Quick Validation
Run demo with verbose output to see:
- Model loading and device selection
- Frame dimensions and FPS
- Number of detections per frame
- ROI membership test results

Print detection details via pipeline CLI:
```bash
python3 src/pipeline.py --source test.jpg --show
```
Output shows per-detection dict with bbox, confidence, class_id, and in_roi flag.

### Performance Profiling
- FPS counter in top-left of output (smoothed over 10 frames)
- Adjust `--conf-thres` and `--iou-thres` to trade accuracy for speed
- Use `--device cpu` vs `--device cuda:0` to compare inference time

### Common Issues
- **Model weights not found**: Ensure `weights/best_small.pt` exists
- **CUDA out of memory**: Reduce frame resolution or use CPU
- **ROI polygon visualization incorrect**: Verify `configs/roi.json` polygon vertices match frame dimensions (1280x720 by default)
- **Slow inference on CPU**: Expected; YOLOv9-Small still requires ~30 FPS on modern CPUs

---

## Code Quality & Contribution Notes

- Type hints are used throughout (e.g., `Tuple[int, int, int, int]` for bbox, `Optional[str]` for paths)
- Dataclasses and static methods organize related logic
- Relative imports use `try/except` fallback in pipeline.py for flexibility
- No external logging framework; uses print() for console feedback
- Video codec is hardcoded to `"mp4v"` (h.264 MP4 container)

---

## Configuration Deep Dive

### Model Hyperparameters (src/detector.py)
- **image_size**: (640, 640) — YOLOv9 input resolution
- **conf_threshold**: 0.25 default — Confidence score filter
- **iou_threshold**: 0.45 default — NMS IoU overlap threshold
- **max_det**: 300 — Max detections per frame

### ROI Configuration (configs/roi.json)
Example structure:
```json
{
  "image_size": {"w": 1280, "h": 720},
  "polygon": [[900, 150], [1270, 250], [1270, 710], [800, 710], [760, 520]],
  "check_point": "bbox_bottom_center"
}
```
- **polygon**: 5 vertices defining the blind spot region (right side in typical truck view)
- **check_point**: Strategy for anchor point (`"bbox_bottom_center"` — bottom-middle of bbox, suitable for detecting ground-level objects near the truck)

### Class Configuration (configs/classes.yaml)
```yaml
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
```
YOLOv9-Small is trained on these 4 classes; detection output class_id references this mapping.

---

## Project History & Dependencies

This project builds on:
- **YOLOv9** detection framework (https://github.com/WongKinYiu/yolov9) — embedded in `yolov9/` directory
- Custom fine-tuning on truck-specific dataset (see `report/training_report.md`)
- ROI-based post-processing to identify dangerous blind spot regions

The upstream YOLOv9 code is vendored to avoid API breakage from upstream updates.
