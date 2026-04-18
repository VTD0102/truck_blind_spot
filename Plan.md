# Plan.md — Truck Blind Spot Tracking System

> **Mục tiêu**: Xây dựng pipeline tracking đầy đủ (Kalman + Hungarian + Track Manager) và hệ thống dự đoán chuyển động (Motion Prediction + Confidence Scoring) bên trên nền tảng YOLOv9 Detection đã có sẵn.

---

## Trạng Thái Hiện Tại (Baseline)

Codebase hiện có đã hoàn thiện các thành phần sau:

| File | Trạng thái | Ghi chú |
|------|-----------|---------|
| `src/detector.py` | ✅ Hoàn thiện | `YOLOv9Detector` + `Detection` dataclass |
| `src/roi.py` | ✅ Hoàn thiện | `MultiPolygonROI`, multi-zone, risk_level |
| `src/visualize.py` | ✅ Hoàn thiện | `BlindSpotVisualizer` |
| `src/pipeline.py` | ✅ Hoàn thiện | `BlindSpotPipeline` orchestrator |
| `src/tracking/types.py` | ✅ Hoàn thiện | `Track` dataclass, `FloatBBox`, `Velocity` |
| `src/tracking/kalman_filter.py` | ✅ Hoàn thiện | `BoundingBoxKalmanFilter` (8D state: cx,cy,w,h,vx,vy,vw,vh) |
| `src/tracking/matching.py` | ✅ Hoàn thiện | `match_tracks_detections` (Hungarian + IoU cost matrix) |
| `src/tracking/track_manager.py` | ✅ Hoàn thiện | `TrackManager` (birth/update/death lifecycle) |
| `tests/test_tracking_smoke.py` | 🔶 Partial | Smoke test cơ bản, chưa đầy đủ |

**Kết luận**: Toàn bộ Phase 1 đã được code sẵn. Nhiệm vụ còn lại là:
1. **Phase 1**: Polish, integrate Kalman với TrackManager, viết unit tests đầy đủ và docs.
2. **Phase 2**: Xây dựng `MotionPredictor` module hoàn toàn mới.

---

## Phase 1: Base Infrastructure

### 1.1 — Kalman Filter (DONE — Polish Only)

**File**: `src/tracking/kalman_filter.py`

**Hiện trạng**: `BoundingBoxKalmanFilter` đã hoàn thiện với 8D state vector `[cx, cy, w, h, vx, vy, vw, vh]`.

**Việc cần làm (polish)**:
- ✅ ~~Xác nhận `process_noise` và `measurement_noise` mặc định~~ — `1e-2` / `1e-1` đã có sẵn, configurable qua constructor
- ✅ ~~Kiến trúc matrix F, H, Q, R~~ — đã đúng, 8D state `[cx,cy,w,h,vx,vy,vw,vh]`, InitCovariance diagonal
- [ ] Thêm method `get_velocity() -> Velocity` để expose `(vx, vy)` từ state vector
- [ ] Thêm docstring mô tả đơn vị (pixels/frame) và giả thiết mô hình
- [ ] Kiểm tra edge case: `initiate()` với bbox rất nhỏ (w=0 hoặc h=0)

**Không cần thay đổi**: Kiến trúc matrix (F, H, Q, R) đã đúng.

---

### 1.2 — Hungarian Algorithm (DONE — Polish Only)

**File**: `src/tracking/matching.py`

**Hiện trạng**: `match_tracks_detections()` dùng `scipy.optimize.linear_sum_assignment` với IoU cost matrix. Đã hoàn thiện.

**Việc cần làm (polish)**:
- ✅ ~~`linear_sum_assignment` + IoU gating~~ — đã implement đúng với `scipy.optimize`
- ✅ ~~Edge case empty inputs~~ — đã handle `if not tracks`, `if not detections` rõ ràng
- [ ] Thêm optional `distance_metric` parameter để sau này có thể swap IoU với Mahalanobis distance
- [ ] Thêm unit test cho edge cases: perfect overlap, zero overlap
- [ ] Thêm docstring giải thích cost matrix construction

**Không cần thay đổi**: Logic `linear_sum_assignment` + IoU gating đã đúng.

---

### 1.3 — Track Manager (DONE — Cần tích hợp Kalman)

**File**: `src/tracking/track_manager.py`

**Hiện trạng**: `TrackManager` quản lý lifecycle (birth/update/death) với `iou_threshold`, `max_misses`, `min_hits`. Velocity được tính thủ công từ anchor point delta.

**Việc cần làm (integration)**:
- ✅ ~~`TrackManager` lifecycle (birth/update/death)~~ — đã có đầy đủ với `iou_threshold`, `max_misses`, `min_hits`
- ✅ ~~Track `trace` history~~ — `_append_trace()` đã trim đúng theo `max_trace_length`
- ✅ ~~`Track.velocity` cơ bản~~ — tính từ anchor delta trong `_update_track()` (tạm thời)
- [ ] Tích hợp `BoundingBoxKalmanFilter` vào `Track`: mỗi track sở hữu một instance Kalman filter riêng
- [ ] Thêm field `kalman: BoundingBoxKalmanFilter` vào `Track` dataclass trong `types.py`
- [ ] Trong `_create_track()`: gọi `track.kalman.initiate(detection.bbox)`
- [ ] Trong `_update_track()`: gọi `track.kalman.update(detection.bbox)` → dùng Kalman bbox
- [ ] Trong `_increment_track_ages()`: gọi `track.kalman.predict()` trước khi matching
- [ ] Trong matching: truyền Kalman-predicted bbox vào `match_tracks_detections()`
- [ ] Cập nhật `Track.velocity` từ `kalman.state[4,0]` và `kalman.state[5,0]`

**Ưu tiên**: Đây là task tích hợp quan trọng nhất của Phase 1.

**Sơ đồ tích hợp**:
```
Frame N:
  1. kalman.predict()  → predicted_bbox cho mỗi track hiện có
  2. match(predicted_bboxes, detections)  → Hungarian assignment
  3. kalman.update(matched_detection.bbox)  → corrected bbox
  4. track.bbox = kalman.get_bbox_xyxy()
  5. track.velocity = (kalman.state[4], kalman.state[5])
```

---

### 1.4 — Final Polish & Docs

**Việc cần làm**:
- [ ] Code review toàn bộ `src/tracking/` module
- [ ] ⚠️ File hiện tên là `init.py` (không phải `__init__.py`) — cần đổi tên để Python nhận ra package
- [ ] Cập nhật `init.py` → `__init__.py` và export đầy đủ: `Track`, `TrackManager`, `BoundingBoxKalmanFilter`, `match_tracks_detections`
- [ ] Viết/cập nhật `tests/test_tracking_smoke.py` với các test cases:
  - Kalman filter: `initiate → predict → update` cycle (có smoke test rồi nhưng chưa đầy đủ)
  - Hungarian: edge cases (no overlap)
  - TrackManager: track birth confirmation (min_hits), track death (max_misses)
  - Full integration: 3 detections → track creation → next frame match
- [ ] Update `CLAUDE.md` thêm section về `src/tracking/` module
- [ ] Chạy `python -m pytest tests/ -v` để validate

---

### 1.5 — Integration vào Pipeline

**File**: `src/pipeline.py` — cần update `BlindSpotPipeline`

**Việc cần làm** (pipeline.py hiện chưa import tracking module nào):
- [ ] Import `TrackManager` vào `pipeline.py`
- [ ] Thêm `TrackManager` instance vào `BlindSpotPipeline.__init__()`
- [ ] Cập nhật `process_frame()`: sau khi có `detections`, gọi `track_manager.update(detections)`
- [ ] `process_frame()` trả về `(annotated_frame, detections, active_tracks)`
- [ ] Cập nhật `BlindSpotVisualizer.draw()` để hiển thị track ID và velocity vector trên frame
- [ ] Cập nhật `app.py` CLI để pass track results

**Signature mới của `process_frame()`**:
```python
def process_frame(
    self, frame: np.ndarray
) -> Tuple[np.ndarray, List[Detection], List[Track]]:
    ...
    tracks = self.track_manager.update(detections)
    annotated_frame = self.visualizer.draw(frame, detections, tracks, self.roi.zones)
    return annotated_frame, detections, tracks
```

---

## Phase 2: Motion Prediction & Confidence

### 2.1 — Buffer + Velocity Calculation

**File mới**: `src/tracking/motion/velocity_buffer.py`

**Mục đích**: Duy trì rolling buffer lịch sử position để tính velocity và acceleration chính xác hơn Kalman state đơn thuần (dùng cho prediction dài hơn 1 frame).

**Class**: `VelocityBuffer`

```python
@dataclass
class VelocityBuffer:
    max_size: int = 10          # lưu 10 frames gần nhất
    timestamps: deque[float]    # epoch time mỗi frame
    positions: deque[Point]     # anchor_point mỗi frame

    def push(self, point: Point, timestamp: float) -> None
    def get_velocity(self) -> Optional[Velocity]        # pixels/second, dùng linear regression
    def get_acceleration(self) -> Optional[Velocity]    # pixels/second^2
    def get_smoothed_velocity(self, window: int = 3) -> Optional[Velocity]  # moving average
```

**Chi tiết implementation**:
- Dùng `collections.deque(maxlen=max_size)` cho timestamps và positions
- Velocity = linear regression (least squares) trên positions[-window:] — robust hơn finite difference
- Acceleration = rate of change of velocity qua các cặp consecutive windows
- Smoothing: exponential moving average với `alpha=0.7` để giảm noise
- Xử lý edge cases: buffer chưa đủ points (cần ≥ 2 để tính velocity, ≥ 3 cho acceleration)

**Tích hợp với `Track`**:
- Thêm field `velocity_buffer: VelocityBuffer` vào `Track` dataclass
- `TrackManager._update_track()` gọi `track.velocity_buffer.push(anchor_point, current_time)`
- Velocity từ buffer override velocity từ Kalman state khi buffer đủ dữ liệu (≥ 5 frames)

---

### 2.2 — Extrapolation + Confidence

**File mới**: `src/tracking/motion/extrapolator.py`

**Mục đích**: Dự đoán vị trí tương lai dựa trên velocity + acceleration, kèm confidence score.

**Class**: `TrajectoryExtrapolator`

```python
class TrajectoryExtrapolator:
    def extrapolate(
        self,
        current_position: Point,
        velocity: Velocity,          # pixels/second
        acceleration: Velocity,      # pixels/second^2
        dt_seconds: float,           # thời gian dự đoán phía trước
    ) -> Point:
        # x(t) = x0 + vx*t + 0.5*ax*t^2
        # y(t) = y0 + vy*t + 0.5*ay*t^2

    def compute_confidence(
        self,
        track: Track,
        prediction_horizon_s: float,
    ) -> float:                      # [0.0, 1.0]
```

**Confidence score formula** — tổng hợp từ 3 thành phần:
```
confidence = w1 * tracking_quality
           + w2 * detection_consistency
           + w3 * motion_smoothness

tracking_quality      = hits / (hits + misses)           # reward continuous tracking
detection_consistency = min(hits, min_hits_threshold) / min_hits_threshold
motion_smoothness     = 1 / (1 + velocity_variance)      # lower variance = higher confidence
```

- Áp dụng time-decay: `confidence *= exp(-lambda * dt)` — confidence giảm dần theo horizon
- Threshold alert: `confidence < 0.4` → LOW risk, `< 0.7` → MEDIUM, `>= 0.7` → HIGH (nếu in ROI)

---

### 2.3 — Full MotionPredictor

**File mới**: `src/tracking/motion/predictor.py`
**File mới**: `src/tracking/motion/__init__.py`

**Mục đích**: Wrapper tổng hợp kết hợp VelocityBuffer + TrajectoryExtrapolator thành một interface duy nhất.

**Class**: `MotionPredictor`

```python
class MotionPredictor:
    def __init__(
        self,
        prediction_horizons_s: List[float] = [0.5, 1.0, 2.0],
        fps: float = 30.0,
        alert_confidence_threshold: float = 0.6,
    ) -> None:

    def update(self, track: Track, timestamp: float) -> MotionPrediction:
        """Push latest data và return prediction."""

    def predict_trajectory(self, track: Track) -> List[PredictedPoint]:
        """Return vị trí dự đoán tại mỗi horizon."""

    def should_alert(self, track: Track, roi_zone: str) -> bool:
        """True nếu track có risk cao trong blind spot."""
```

**Dataclass output**:
```python
@dataclass
class PredictedPoint:
    position: Point
    timestamp_s: float      # seconds from now
    confidence: float

@dataclass
class MotionPrediction:
    track_id: int
    trajectory: List[PredictedPoint]
    overall_confidence: float
    alert_level: str        # "none", "low", "medium", "high"
    velocity_px_per_s: Velocity
    acceleration_px_per_s2: Velocity
```

**Motion validation**:
- Max velocity sanity check: > 500 px/frame → likely noise, giảm confidence
- Direction consistency check: sudden 90° direction change trong 2 frames → decreased confidence
- Bounding box size consistency: rapid size change → object may have been lost/reassigned

---

### 2.4 — Final Polish & Docs (Phase 2)

**Việc cần làm**:
- [ ] Validate predictions với video thực tế: vẽ predicted trajectory lên frame
- [ ] Tuning hyperparameters:
  - `max_size` của buffer: test 5 vs 10 vs 15 frames
  - `prediction_horizons_s`: kiểm tra 0.5s, 1.0s, 2.0s
  - `alert_confidence_threshold`: chỉnh từ 0.4 → 0.7 tùy false positive rate
- [ ] Viết unit tests cho `VelocityBuffer`, `TrajectoryExtrapolator`, `MotionPredictor`
  - Test với constant velocity motion → verify linear extrapolation
  - Test với accelerating motion → verify quadratic extrapolation
  - Test confidence decay với increasing prediction horizon
- [ ] Viết docstrings cho tất cả public methods
- [ ] Update `CLAUDE.md` với section Phase 2

---

### 2.5 — Integration (Phase 2 → Full Pipeline)

**Files cần update**: `src/pipeline.py`, `src/visualize.py`, `app.py`

**Việc cần làm**:
- [ ] Thêm `MotionPredictor` vào `BlindSpotPipeline.__init__()`
- [ ] Trong `process_frame()`:
  ```python
  current_time = time.time()
  tracks = self.track_manager.update(detections)
  predictions = {t.track_id: self.motion_predictor.update(t, current_time) for t in tracks}
  ```
- [ ] Cập nhật `BlindSpotVisualizer`:
  - Vẽ predicted trajectory points (dots hoặc dashed line) trên frame
  - Hiển thị confidence score và alert level bên cạnh track ID
  - Color coding: xanh = safe, vàng = medium risk, đỏ = high risk
- [ ] Alert generation logic trong `app.py`:
  - Log alert khi `prediction.alert_level in ["medium", "high"]` và `track.in_roi`
  - Optional: sound alert hoặc overlay warning banner
- [ ] Full pipeline benchmark:
  - Target: tổng latency ≤ 35ms/frame (≥ 28 FPS) trên CPU
  - Profile từng bước: detection, tracking, prediction, visualization
- [ ] Cập nhật CLI args trong `app.py`: `--prediction-horizon`, `--alert-threshold`

---

## Cấu Trúc Thư Mục Sau Khi Hoàn Thiện

```
src/
├── detector.py              ✅ (không đổi)
├── roi.py                   ✅ (không đổi)
├── visualize.py             🔶 update: hiển thị tracks, predictions
├── pipeline.py              🔶 update: tích hợp TrackManager + MotionPredictor
└── tracking/
    ├── __init__.py          🔶 verify exports
    ├── types.py             🔶 thêm kalman field + velocity_buffer field
    ├── kalman_filter.py     🔶 thêm get_velocity()
    ├── matching.py          ✅ (không đổi)
    ├── track_manager.py     🔶 tích hợp Kalman predict/update
    └── motion/              🆕 toàn bộ mới
        ├── __init__.py
        ├── velocity_buffer.py
        ├── extrapolator.py
        └── predictor.py

tests/
├── test_tracking_smoke.py   🔶 expand
├── test_kalman.py           🆕
├── test_matching.py         🆕
├── test_track_manager.py    🆕
└── test_motion_predictor.py 🆕
```

---

## Thứ Tự Thực Hiện Đề Xuất

```
1.3-integration  →  1.1-polish  →  1.2-polish  →  1.4-docs+tests  →  1.5-pipeline-integration
        ↓
2.1-velocity-buffer  →  2.2-extrapolation  →  2.3-full-predictor  →  2.4-docs+tests  →  2.5-full-integration
```

**Lý do**: Task 1.3 (Kalman integration vào TrackManager) là dependency của mọi thứ khác — phải làm trước.

---

## Testing Strategy

### Unit Tests (pytest)
```bash
python -m pytest tests/ -v
python -m pytest tests/test_kalman.py -v          # Kalman filter cycle
python -m pytest tests/test_matching.py -v        # Hungarian assignment
python -m pytest tests/test_track_manager.py -v  # lifecycle management
python -m pytest tests/test_motion_predictor.py -v  # prediction accuracy
```

### Integration Test (end-to-end)
```bash
# Chạy pipeline hoàn chỉnh với video test
python app.py --source assets/videos/demo.mp4 --show

# Record output để validate visually
python app.py --source assets/videos/demo.mp4 --output outputs/tracked_demo.mp4
```

### Performance Benchmark
```bash
python -c "
import time, cv2
# ... benchmark script đo latency từng stage
"
```

---

## Tuning Parameters Reference

| Parameter | Default | Range | Ảnh hưởng |
|-----------|---------|-------|-----------|
| `kalman.process_noise` | 1e-2 | 1e-4 – 1e-1 | Cao → trust detection hơn; thấp → trust prediction hơn |
| `kalman.measurement_noise` | 1e-1 | 1e-2 – 1.0 | Cao → smooth bbox; thấp → responsive to detections |
| `track_manager.iou_threshold` | 0.3 | 0.1 – 0.6 | Thấp → match aggressively; cao → only match high overlap |
| `track_manager.max_misses` | 5 | 2 – 15 | Cao → tracks persist longer through occlusion |
| `track_manager.min_hits` | 2 | 1 – 5 | Cao → fewer false tracks (need multiple detections to confirm) |
| `velocity_buffer.max_size` | 10 | 5 – 20 | Cao → smoother velocity but slower to react to changes |
| `motion.alert_threshold` | 0.6 | 0.4 – 0.8 | Thấp → more alerts (higher sensitivity, more false positives) |

---

## Key Design Decisions

1. **Kalman state 8D vs 6D**: Chọn 8D `[cx,cy,w,h,vx,vy,vw,vh]` để track cả kích thước bbox, giúp phân biệt xe đến gần (bbox to ra) vs xe đứng yên.

2. **IoU cost matrix vs Euclidean distance**: IoU được chọn vì nó không cần calibration camera-specific, phù hợp khi không có thông tin 3D depth.

3. **Velocity buffer song song Kalman**: Kalman velocity phù hợp cho matching (1 frame ahead), còn velocity buffer với rolling regression phù hợp hơn cho prediction dài hạn (0.5-2s) vì ít bị noise của Kalman state.

4. **Confidence time-decay**: Exponential decay `e^(-λt)` là lựa chọn tự nhiên — uncertainty tăng theo thời gian khi không có observation mới.

5. **Không dùng deep learning tracker (DeepSORT)**: Để tránh dependency phức tạp và giữ latency thấp. IoU-based matching đủ tốt cho xe cộ vì chúng có motion tương đối mượt và không overlap nhiều.
