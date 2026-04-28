# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Truck Blind Spot Detection** is a real-time YOLOv9-based ADAS system that flags objects (person, bike, motor, car, truck, bus) entering configurable blind-spot zones of a truck. The current default checkpoint is `weights/best_6k.pt` (see `app.py` defaults).

> **Read `RULES.md`** — it's the project's enforced rulebook (Python 3.10+, Vietnamese comments/error messages, dependency direction, performance budgets, visualization conventions). This file does not duplicate those rules.

---

## Architecture & Code Structure

```
Frame
  → YOLOv9Detector.predict()          (src/detector.py)      → List[Detection]
  → MultiPolygonROI.get_zone()        (src/roi.py)           → detection.in_roi / zone_name / risk_level
  → TrackManager.update()             (src/tracking/*)       → List[Track]
  → BlindSpotVisualizer.draw()        (src/visualize.py)     → annotated frame
```

### Core Modules

| File | Purpose |
|------|---------|
| **app.py** | Main CLI entry point. Video capture loop, keyboard controls (`p`/`r`/`q`), video writer, FPS smoothing. |
| **src/detector.py** | `YOLOv9Detector` + `Detection` dataclass. Loads model via `DetectMultiBackend`, runs letterbox preprocess → inference → NMS → `scale_boxes`. **Only place in the repo that injects `yolov9/` into `sys.path`.** |
| **src/roi.py** | `MultiPolygonROI` + `ROIZone`. Multi-zone ROI (per-profile: `front_camera`, `rear_camera`) with per-zone `risk_level` and color. Scales vertices to current frame size. |
| **src/visualize.py** | `BlindSpotVisualizer` — draws zone polygons, detections, track IDs, velocity vectors, and warning overlays. |
| **src/pipeline.py** | `BlindSpotPipeline` orchestrator (detector → ROI → tracking → visualizer). `process_frame()` returns `(annotated_frame, detections, tracks)`. |
| **src/roi_evaluation.py** | Standalone tool: per-zone / per-class recall evaluation. Run via `python -m src.roi_evaluation`. |
| **src/tracking/** | Kalman filter + Hungarian matching + `TrackManager`. Tracks now own a per-object Kalman filter and the package exports public symbols via `src/tracking/__init__.py`. **Architectural invariant: must not import from `yolov9/` or `src/detector.py`** (see RULES §2.1). |
| **yolov9/** | Vendored upstream. **Do not edit** (RULES §5). |

### `Detection` dataclass (flows through pipeline)

`bbox (x1,y1,x2,y2)`, `confidence`, `class_id`, `class_name`, `anchor_point`, `in_roi`, `zone_name`, `risk_level`. Mutated in-place by ROI stage.

### Configuration Files

| File | Purpose |
|------|---------|
| `configs/classes.yaml` | 6 classes: `person, bike, motor, car, truck, bus`. Must stay in sync with `configs/blindspot.yaml`. |
| `configs/roi.json` | Multi-zone ROI. Schema: `profiles.{front_camera,rear_camera}.zones[].{name, risk_level, color, polygon}`. |
| `configs/blindspot.yaml` | Dataset YAML for YOLOv9 training/eval (used by `roi_evaluation.py`). |
| `weights/best_6k.pt`, `weights/best_pilot_4k5.pt` | Fine-tuned checkpoints. `best_6k.pt` is the current `app.py` default. |

---

## Common Commands

### Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Demo (app.py)

```bash
# Default: weights/best_6k.pt + assets/videos/demo4.mp4 + roi-profile=front_camera
python3 app.py

# Swap ROI profile (required when camera is rear-facing)
python3 app.py --roi-profile rear_camera

# Custom source / output / thresholds / device
python3 app.py --source assets/videos/demo.mp4 --output outputs/result.mp4
python3 app.py --source 0                         # webcam index 0
python3 app.py --loop                             # loop a file source
python3 app.py --conf-thres 0.3 --iou-thres 0.5
python3 app.py --device cpu                       # or cuda:0 (empty = auto)
```

Playback keys: `p` pause/resume · `r` restart (file only) · `q` quit. `src/pipeline.py`'s standalone viewer also accepts `Esc`.

### Pipeline CLI (image/video without `app.py`)

```bash
python3 src/pipeline.py --source path/to/image.jpg --show
python3 src/pipeline.py --source assets/videos/demo.mp4 --output outputs/out.mp4 --show
```

Image mode prints per-detection dicts and per-track dicts (track ID, bbox, velocity, hits, misses, status).

### Tests

```bash
python -m pytest tests/ -v                        # RULES §8.1 — pytest, not unittest
python -m pytest tests/test_tracking_smoke.py -v  # tracking module only (no GPU/weights needed)
```

### ROI-aware recall evaluation

```bash
python -m src.roi_evaluation \
  --weights weights/best_6k.pt \
  --data configs/blindspot.yaml \
  --roi configs/roi.json \
  --roi-profile front_camera \
  --split val --conf-thres 0.25 --iou-match 0.5 \
  --output-dir outputs/roi_eval
```

---

## Project-Specific Gotchas

- **Vendored YOLOv9** — never edit `yolov9/`. All imports (`DetectMultiBackend`, `letterbox`, `non_max_suppression`, `scale_boxes`, `select_device`) go through `src/detector.py`, which is the single place that adds `yolov9/` to `sys.path`.
- **Tracking dependency direction is enforced** — `src/tracking/` must not import from `yolov9/` or `src/detector.py`; it operates on dataclasses only. Tracking tests must pass without GPU or `.pt` weights.
- **Tracking is wired into `BlindSpotPipeline`** — `pipeline.py` now returns `(frame, detections, tracks)` and recomputes ROI metadata on tracked boxes before rendering.
- **`src/tracking/__init__.py` is the supported import surface** — prefer `from src.tracking import TrackManager, BoundingBoxKalmanFilter` over reaching into internal modules unless you need implementation details.
- **ROI profiles are required** — `configs/roi.json` has no top-level `polygon`. Code must pick a profile name (`front_camera` or `rear_camera`). `MultiPolygonROI` rescales all zone vertices to the current frame size on every call to `update_frame_size()`.
- **Class list is 6, not 4** — syncing `configs/classes.yaml` with `configs/blindspot.yaml` is required when retraining.
- **ADAS bias (RULES §12)** — false negatives are worse than false positives; keep `conf_threshold` low, `max_misses` high, `min_hits` low.
- **Path handling** — every module resolves paths via `PROJECT_ROOT = Path(__file__).resolve().parents[N]` and `_resolve_path()`. Never hardcode absolute paths (RULES §3).
- **Track velocity source** — `Track.velocity` is populated from Kalman `(vx, vy)` for Phase 1 matching/tracking; `velocity_buffer` is maintained in parallel for later Phase 2 prediction work.
- **Video codec** hardcoded to `mp4v`; output containers must be `.mp4`.

## Default Hyperparameters

| Stage | Parameter | Default | Source |
|-------|-----------|---------|--------|
| Detector | `image_size` | `(640, 640)` | `src/detector.py` |
| Detector | `conf_threshold` | `0.25` | `src/detector.py` |
| Detector | `iou_threshold` | `0.45` (NMS) | `src/detector.py` |
| Detector | `max_det` | `300` | `src/detector.py` |
| Tracking | `iou_threshold` | `0.3` (Hungarian) | `TrackManager` |
| Tracking | `max_misses` | `5` | `TrackManager` |
| Tracking | `min_hits` | `2` | `TrackManager` |
| Kalman | `process_noise` / `measurement_noise` | `1.0` / `10.0` | `BoundingBoxKalmanFilter` |

Acceptable ranges and tuning guidance live in `RULES.md` §7 and `Plan.md`.

## Ongoing Work

See `Plan.md` for the tracking/motion-prediction roadmap:
- **Phase 1** — complete; remaining work is mostly documentation/tuning follow-up around tracking defaults.
- **Phase 2** — new `src/tracking/motion/` (VelocityBuffer, TrajectoryExtrapolator, MotionPredictor) with confidence scoring and predicted-trajectory visualization.

## Phase 2 — Motion Prediction (`src/tracking/motion/`)

| File | Purpose |
|------|---------|
| `perspective.py` | `PerspectiveTransform` — Inverse Perspective Mapping (IPM) to Bird's-Eye View (BEV). `pixel_to_bev()`, `bev_to_pixel()`, `estimate_distance()`. |
| `velocity_buffer.py` | `VelocityBuffer` — Rolling buffer of (position, timestamp) pairs. `get_velocity()`, `get_acceleration()`, `get_smoothed_velocity()` via least-squares regression. |
| `extrapolator.py` | `TrajectoryExtrapolator` — Quadratic extrapolation `x(t) = x0 + vx*t + 0.5*ax*t²`. `compute_confidence()` = weighted (tracking_quality + consistency + smoothness) × time-decay. |
| `predictor.py` | `MotionPredictor` — Wrapper combining the above. `update()` → `MotionPrediction`, `predict_trajectory()`, `should_alert()`. Motion validation: max velocity sanity (500 px/frame), 90° direction change, bbox size consistency. |

**Key dataclasses** (in `src/common/models.py`):
- `PredictedPoint(position, timestamp_s, confidence)`
- `MotionPrediction(track_id, trajectory, overall_confidence, alert_level, velocity_px_per_s, acceleration_px_per_s2)`

**Alert levels**: `overall_confidence < 0.4` → "low", `< 0.7` → "medium", `≥ 0.7` → "high" (only when `track.in_roi`).

**Velocity source priority**: buffer smoothed velocity (≥5 frames) > Kalman velocity.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **truck_blind_spot** (2477 symbols, 7892 relationships, 209 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/truck_blind_spot/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/truck_blind_spot/context` | Codebase overview, check index freshness |
| `gitnexus://repo/truck_blind_spot/clusters` | All functional areas |
| `gitnexus://repo/truck_blind_spot/processes` | All execution flows |
| `gitnexus://repo/truck_blind_spot/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
