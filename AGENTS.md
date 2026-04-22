# Truck Blind Spot Detection — Agent Guide

Hệ thống ADAS phát hiện vật thể trong vùng điểm mù xe tải theo thời gian thực, dùng YOLOv9 + Kalman tracking + motion prediction. **Đọc `RULES.md`** trước khi sửa code — đó là bộ quy tắc bắt buộc của project.

## Pipeline

```
Frame
  → YOLOv9Detector.predict()          src/detector.py        → List[Detection]
  → MultiPolygonROI.get_zone()        src/roi.py             → in_roi / zone_name / risk_level
  → TrackManager.update()             src/tracking/          → List[Track]
  → MotionPredictor.update()          src/tracking/motion/   → Dict[int, MotionPrediction]
  → BlindSpotVisualizer.draw()        src/visualize.py       → annotated frame
```

## Key Files

| File | Vai trò |
|------|---------|
| `app.py` | CLI entry point — vòng lặp video, phím `p/r/q`, ghi output, log ALERT |
| `src/pipeline.py` | `BlindSpotPipeline` — orchestrator, trả về `(frame, detections, tracks)` |
| `src/detector.py` | `YOLOv9Detector` — **nơi duy nhất** inject `yolov9/` vào `sys.path` |
| `src/roi.py` | `MultiPolygonROI` — multi-zone, per-zone `risk_level`, scale theo frame |
| `src/visualize.py` | `BlindSpotVisualizer` — vẽ zone, detection, track ID, velocity arrow, trajectory dots, alert label |
| `src/tracking/kalman_filter.py` | `BoundingBoxKalmanFilter` — state 8D `[cx,cy,w,h,vx,vy,vw,vh]` |
| `src/tracking/matching.py` | `match_tracks_detections` — Hungarian + IoU cost matrix |
| `src/tracking/track_manager.py` | `TrackManager` — lifecycle birth/update/death, buffer velocity ≥5 frames |
| `src/tracking/motion/velocity_buffer.py` | `VelocityBuffer` — rolling buffer, linear/quadratic regression |
| `src/tracking/motion/extrapolator.py` | `TrajectoryExtrapolator` — `x(t)=x0+vx·t+½ax·t²`, confidence scoring |
| `src/tracking/motion/perspective.py` | `PerspectiveTransform` — IPM/BEV homography |
| `src/tracking/motion/predictor.py` | `MotionPredictor` — wrapper, `update()→MotionPrediction`, motion validation, alert level |
| `src/common/models.py` | `Detection`, `Track`, `PredictedPoint`, `MotionPrediction`, `AlertEvent` |
| `src/common/enums.py` | `TrackStatus`, `AlertLevel`, `AlertType` |

## Invariants (KHÔNG được vi phạm)

- `src/tracking/` **không được** import từ `yolov9/` hoặc `src/detector.py`
- `yolov9/` là vendored upstream — **không sửa**
- Tests phải chạy được **không cần GPU** hoặc file `.pt`: `python -m pytest tests/ -v`
- Mọi path resolve qua `PROJECT_ROOT / path` — không hardcode absolute path
- Comments/docstrings viết bằng **tiếng Việt**

## Velocity Source Priority

```
Track.velocity  ←  buffer.get_smoothed_velocity()  (nếu len(buffer) ≥ 5)
                ←  kalman.get_velocity()            (fallback)
```

## Alert Levels

| `overall_confidence` | `alert_level` |
|---------------------|---------------|
| ≥ 0.7 và in_roi | `"high"` |
| ≥ 0.4 và in_roi | `"medium"` |
| > 0.0 và in_roi | `"low"` |
| không in_roi | `"none"` |

## Default Hyperparameters

| Tham số | Mặc định |
|---------|---------|
| `conf_threshold` | 0.25 |
| `iou_threshold` (NMS) | 0.45 |
| `track_iou_threshold` | 0.3 |
| `max_misses` | 5 |
| `min_hits` | 2 |
| `kalman process_noise` / `measurement_noise` | 1.0 / 10.0 |
| `prediction_horizons_s` | [0.5, 1.0, 2.0] |
| `alert_confidence_threshold` | 0.6 |

## Test & Run

```bash
python -m pytest tests/ -v                          # 59 tests, không cần GPU
python app.py                                       # demo mặc định
python app.py --source assets/videos/demo.mp4 \
  --output outputs/out.mp4 \
  --prediction-horizon 1.0 --alert-threshold 0.5
```

## Roadmap Status

- **Phase 0** ✅ Shared types (`src/common/`)
- **Phase 1** ✅ Kalman + Hungarian + TrackManager
- **Phase 2** ✅ MotionPredictor + trajectory visualization (còn: hyperparameter tuning)
- **Phase 3–4** Chưa bắt đầu (API, database, dashboard, mock data)

---

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **truck_blind_spot** (2257 symbols, 7165 relationships, 190 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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