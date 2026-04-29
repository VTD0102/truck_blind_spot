# Plan.md — Truck Blind Spot Tracking System

> **Mục tiêu**: Xây dựng pipeline tracking đầy đủ (Kalman + Hungarian + Track Manager) và hệ thống dự đoán chuyển động (Motion Prediction + Confidence Scoring) bên trên nền tảng YOLOv9 Detection đã có sẵn.

---

## Trạng Thái Hiện Tại (Baseline)

/
Codebase hiện có đã hoàn thiện các thành phần sau:

| File                            | Trạng thái    | Ghi chú                                                     |
| ------------------------------- | ------------- | ----------------------------------------------------------- |
| `src/detector.py`               | ✅ Hoàn thiện | `YOLOv9Detector` + `Detection` dataclass                    |
| `src/roi.py`                    | ✅ Hoàn thiện | `MultiPolygonROI`, multi-zone, risk_level                   |
| `src/visualize.py`              | ✅ Hoàn thiện | `BlindSpotVisualizer`                                       |
| `src/pipeline.py`               | ✅ Hoàn thiện | `BlindSpotPipeline` orchestrator                            |
| `src/tracking/types.py`         | ✅ Hoàn thiện | `Track` dataclass, `FloatBBox`, `Velocity`                  |
| `src/tracking/kalman_filter.py` | ✅ Hoàn thiện | `BoundingBoxKalmanFilter` (8D state: cx,cy,w,h,vx,vy,vw,vh) |
| `src/tracking/matching.py`      | ✅ Hoàn thiện | `match_tracks_detections` (Hungarian + IoU cost matrix)     |
| `src/tracking/track_manager.py` | ✅ Hoàn thiện | `TrackManager` (birth/update/death lifecycle)               |
| `tests/test_tracking_smoke.py`  | 🔶 Partial    | Smoke test cơ bản, chưa đầy đủ                              |

**Kết luận**: Toàn bộ Phase 1 đã được code sẵn. Nhiệm vụ còn lại là:

1. **Phase 1**: Polish, integrate Kalman với TrackManager, viết unit tests đầy đủ và docs.
2. **Phase 2**: Xây dựng `MotionPredictor` module hoàn toàn mới.
3. **Phase 3**: Tích hợp hệ thống (API, Database, Dashboard).

---

## Phase 0: Shared Types & Foundation

_Mục đích: Định nghĩa ngôn ngữ chung cho toàn bộ hệ thống để đảm bảo tính nhất quán giữa các module._

- [x] **Tạo TypeScript/Python type definitions cho**:
  - `Detection`: bounding box, class, confidence, anchor_point.
  - `Track`: ID, position, velocity, history.
  - `KalmanState`: state vector (8D), covariance.
  - `MotionPrediction`: predicted position, confidence, timestamp.
  - `AlertEvent`: type, severity, location, timestamp.
- [x] **Định nghĩa enums**: cho track status, alert levels.
- [x] **Tạo data transfer objects (DTOs)**.
- [x] **Tài liệu hóa schema** của tất cả các types.
- [x] **Ensure consistency** giữa các module.

---

## Phase 1: Base Infrastructure

### 1.1 — Kalman Filter (DONE — Polish Only)

**File**: `src/tracking/kalman_filter.py`

**Hiện trạng**: `BoundingBoxKalmanFilter` đã hoàn thiện với 8D state vector `[cx, cy, w, h, vx, vy, vw, vh]`.

**Việc cần làm (polish)**:

- [x] Xác nhận `process_noise` và `measurement_noise` mặc định bằng cách chạy với video thực tế
- [x] Thêm method `get_velocity() -> Velocity` để expose `(vx, vy)` từ state vector — hiện tại TrackManager tính velocity thủ công từ anchor points thay vì dùng Kalman state
- [x] Thêm docstring mô tả đơn vị (pixels/frame) và giả thiết mô hình — đã có docstring trong class
- [x] Kiểm tra edge case: `initiate()` với bbox rất nhỏ — `_clamp_state_size()` xử lý `w,h < 1e-6`

**Không cần thay đổi**: Kiến trúc matrix (F, H, Q, R) đã đúng.

---

### 1.2 — Hungarian Algorithm (DONE — Polish Only)

**File**: `src/tracking/matching.py`

**Hiện trạng**: `match_tracks_detections()` dùng `scipy.optimize.linear_sum_assignment` với IoU cost matrix. Đã hoàn thiện.

**Việc cần làm (polish)**:

- [x] Thêm optional `distance_metric` parameter để sau này có thể swap IoU với Mahalanobis distance (khi tích hợp Kalman prediction)
- [x] Thêm unit test cho edge cases: 0 tracks + N detections, N tracks + 0 detections, perfect overlap, zero overlap
- [x] Thêm docstring giải thích cost matrix construction — đã có trong `matching.py`

**Không cần thay đổi**: Logic `linear_sum_assignment` + IoU gating đã đúng.

---

### 1.3 — Track Manager (DONE — Cần tích hợp Kalman)

**File**: `src/tracking/track_manager.py`

**Hiện trạng**: `TrackManager` quản lý lifecycle (birth/update/death) với `iou_threshold`, `max_misses`, `min_hits`. Velocity được tính thủ công từ anchor point delta.

**Việc cần làm (integration)**:

- [x] Tích hợp `BoundingBoxKalmanFilter` vào `Track`: mỗi track sở hữu một instance Kalman filter riêng
- [x] Thêm field `kalman: BoundingBoxKalmanFilter` vào `Track` dataclass trong `types.py`
  - Dùng `field(default_factory=BoundingBoxKalmanFilter)` hoặc khởi tạo trong `TrackManager._create_track()`
- [x] Trong `_create_track()`: gọi `track.kalman.initiate(detection.bbox)`
- [x] Trong `_update_track()`: gọi `track.kalman.update(detection.bbox)` và dùng Kalman bbox thay vì raw detection bbox
- [x] Trong `_increment_track_ages()`: gọi `track.kalman.predict()` để dự đoán next bbox trước khi matching
- [x] Trong matching: truyền Kalman-predicted bbox thay vì current bbox vào `match_tracks_detections()`
- [x] Lấy `(vx, vy)` từ Kalman state thay vì tính thủ công từ anchor delta
- [x] Cập nhật `Track.velocity` từ `kalman.state[4,0]` và `kalman.state[5,0]`

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

- [x] Code review toàn bộ `src/tracking/` module
- [x] Đảm bảo `src/tracking/__init__.py` export đầy đủ — `types.py` hiện re-export từ `common`
- [x] ⚠️ File `src/tracking/init.py` vẫn còn tồn tại (tên sai, cần kiểm tra)
- [x] Viết/cập nhật `tests/test_tracking_smoke.py` với các test cases:
  - [x] Kalman filter: `initiate → predict → update` cycle
  - [x] Hungarian: edge cases (0 tracks, 0 detections, no overlap)
  - [x] TrackManager: track birth confirmation (min_hits), track death (max_misses)
  - [x] Full integration: 3 detections → track creation → next frame match
- [x] Update `CLAUDE.md` thêm section về `src/tracking/` module
- [x] Chạy `python -m pytest tests/ -v` để validate

---

### 1.5 — Integration vào Pipeline

**File**: `src/pipeline.py` — cần update `BlindSpotPipeline`

**Việc cần làm**:

- [x] Import `TrackManager` vào `pipeline.py` — hiện tại chưa có
- [x] Thêm `TrackManager` instance vào `BlindSpotPipeline.__init__()`
- [x] Cập nhật `process_frame()`: sau khi có `detections`, gọi `track_manager.update(detections)`
- [x] `process_frame()` trả về `(annotated_frame, detections, active_tracks)` hoặc embed track info vào detections
- [x] Cập nhật `BlindSpotVisualizer.draw()` để hiển thị track ID và velocity vector trên frame
- [x] Cập nhật `app.py` CLI để pass track results

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

> **Quyết định thiết kế**: Sử dụng **Relative Tracking** (Hướng 1) — coi xe tải đứng yên tuyệt đối, mọi chuyển động đo được trên pixel là vận tốc tương đối giữa đối tượng và xe tải. Không cần CAN bus, IMU hay GPS. Ego-motion compensation (Hướng 2) sẽ là upgrade tương lai nếu cần.

> **⚠️ Lưu ý quan trọng về Perspective Projection**: Camera là hình chiếu phối cảnh. Một xe chạy đều 40km/h ngoài đời, trên pixel sẽ có vận tốc **không hằng số** — càng gần camera, `vy(pixel)` tăng nhanh tạo "gia tốc ảo". Nếu ngoại suy tuyến tính `x(t) = x0 + vx*t` trên pixel gốc sẽ **sai lệch nghiêm trọng** cho prediction 1-2s. Do đó cần chuyển tọa độ sang BEV trước khi prediction.

### 2.0 — Coordinate Transform (IPM / Bird's-Eye View)

**File mới**: `src/tracking/motion/perspective.py`

**Mục đích**: Chuyển tọa độ pixel sang mặt phẳng BEV (nhìn từ trên xuống) để velocity và extrapolation có tỉ lệ đồng nhất (pixel ≈ mét).

**Nguyên lý**: Inverse Perspective Mapping (IPM) dùng Homography matrix `H` để warp điểm chạm đất (bottom-center của bbox) từ camera view sang top-down view.

```python
class PerspectiveTransform:
    def __init__(self, homography_matrix: np.ndarray) -> None:
        """H: 3x3 matrix, calibrated từ 4 điểm tương ứng trên mặt đất."""

    def pixel_to_bev(self, point: Point) -> Tuple[float, float]:
        """Chuyển tọa độ pixel (bottom-center) sang BEV (mét hoặc cm)."""

    def bev_to_pixel(self, bev_point: Tuple[float, float]) -> Point:
        """Chuyển ngược từ BEV sang pixel để vẽ lên frame."""

    @staticmethod
    def estimate_distance(bbox_height_px: float, focal_length: float, real_height_m: float) -> float:
        """Ước lượng khoảng cách Z bằng Pinhole model: Z = (f * H_real) / h_pixel."""
```

**Calibration**: Cần 4 cặp điểm tương ứng (pixel ↔ thực tế) trên mặt đường để tính `H`. Có thể:

- Đo thủ công từ video gốc (đánh dấu 4 điểm trên mặt đường đã biết khoảng cách)
- Hoặc ước lượng từ thông số camera (focal length, góc lắp đặt)

**Việc cần làm**:

- [x] Code `PerspectiveTransform` class với `pixel_to_bev` và `bev_to_pixel`
- [x] Thêm static method `estimate_distance` bằng Pinhole model
- [x] Thêm field `distance_m: Optional[float]` vào `Detection` và `Track` dataclass
- [x] Khởi tạo mock homography matrix dùng cho testing

---

### 2.1 — Buffer + Velocity Calculation

**File mới**: `src/tracking/motion/velocity_buffer.py`

**Mục đích**: Duy trì rolling buffer lịch sử position **trên BEV space** để tính velocity và acceleration chính xác.

**Class**: `VelocityBuffer`

```python
@dataclass
class VelocityBuffer:
    max_size: int = 10          # lưu 10 frames gần nhất
    timestamps: deque[float]    # epoch time mỗi frame
    positions: deque[Point]     # BEV-transformed positions (nếu có H) hoặc pixel fallback

    def push(self, point: Point, timestamp: float) -> None
    def get_velocity(self) -> Optional[Velocity]        # units/second (BEV: m/s, pixel: px/s)
    def get_acceleration(self) -> Optional[Velocity]    # units/second^2
    def get_smoothed_velocity(self, window: int = 3) -> Optional[Velocity]  # moving average
```

**Chi tiết implementation**:

- Dùng `collections.deque(maxlen=max_size)` cho timestamps và positions
- Velocity = linear regression (least squares) trên positions[-window:] — robust hơn finite difference
- Acceleration = rate of change of velocity qua các cặp consecutive windows
- Smoothing: exponential moving average với `alpha=0.7` để giảm noise
- Xử lý edge cases: buffer chưa đủ points (cần ≥ 2 để tính velocity, ≥ 3 cho acceleration)
- **Nếu có Homography**: push BEV-transformed point → velocity tính bằng m/s
- **Nếu không có Homography (fallback)**: push raw pixel → velocity tính bằng px/s (chấp nhận sai lệch)

**Việc cần làm**:

- [x] Code `VelocityBuffer` class sử dụng `deque`
- [x] Implement linear regression (`get_velocity`) và quadratic fit (`get_acceleration`)
- [x] Integrate `VelocityBuffer` vào `Track` dataclass
- [x] Cập nhật `TrackManager._update_track()` để tự động gọi `track.velocity_buffer.push(...)`
- [x] Thay thế Kalman velocity bằng buffer velocity nếu buffer đủ dữ liệu (≥ 5 frames)

---

### 2.2 — Extrapolation + Confidence

**File mới**: `src/tracking/motion/extrapolator.py`

**Mục đích**: Dự đoán vị trí tương lai **trên BEV space** dựa trên velocity + acceleration, kèm confidence score.

**Class**: `TrajectoryExtrapolator`

```python
class TrajectoryExtrapolator:
    def __init__(self, perspective_transform: Optional[PerspectiveTransform] = None):
        """Nếu có transform, extrapolate trên BEV rồi chuyển ngược về pixel."""

    def extrapolate(
        self,
        current_position: Point,     # BEV hoặc pixel
        velocity: Velocity,          # BEV: m/s, pixel: px/s
        acceleration: Velocity,      # BEV: m/s^2, pixel: px/s^2
        dt_seconds: float,           # thời gian dự đoán phía trước
    ) -> Point:
        # Trên BEV: x(t) = x0 + vx*t + 0.5*ax*t^2 ← CHÍNH XÁC vì tỉ lệ đồng nhất
        # Trên pixel gốc: công thức này chỉ là xấp xỉ (sai lệch tăng theo horizon)

    def compute_confidence(
        self,
        track: Track,
        prediction_horizon_s: float,
    ) -> float:                      # [0.0, 1.0]
```

> **Lưu ý**: Khi không có Homography (chưa calibrate camera), hệ thống vẫn chạy được nhưng prediction sẽ kém chính xác cho horizon > 0.5s. Đây là tradeoff có chủ đích cho MVP.

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

**Việc cần làm**:

- [x] Code `TrajectoryExtrapolator` class
- [x] Implement logic nội suy x(t) = x0 + vx*t + 0.5*ax\*t^2
- [x] Xây dựng hàm tính `confidence` với kết hợp tracking, consistency, smoothness, và decay
- [x] Cập nhật `extrapolate()` để gọi hàm warp trên `PerspectiveTransform` nếu được truyền vào (BEV transform)

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

**Việc cần làm**:

- [x] Code `MotionPredictor` wrapper
- [x] Định nghĩa `MotionPrediction` và `PredictedPoint` dataclass
- [x] Cài đặt `predict_trajectory()` nội suy quỹ đạo ở `0.5s, 1.0s, 2.0s`
- [x] Thêm logic motion validation (sanity check cho max velocity/direction)
- [x] Thêm custom alert generation rules logic cho ROI zone

---

### 2.4 — Final Polish & Docs (Phase 2)

**Việc cần làm**:

- [x] Validate predictions với video thực tế: vẽ predicted trajectory lên frame
- [x] Tuning hyperparameters (kết quả từ `tools/benchmark.py --skip-detection`):
  - `max_size` buffer: 5/10/15 frames → MAE tương đương (~15-19 px/s); giữ mặc định **10** (cân bằng latency và smoothing).
  - `prediction_horizons_s`: [0.5, 1.0, 2.0]s → extrapolation error < 0.3px với quadratic motion; giữ nguyên.
  - `alert_confidence_threshold`: 0.4→52%, 0.5→30%, 0.6→19%, 0.7→6% alert rate → **0.6** tối ưu cho ADAS (giảm false positive, vẫn đủ nhạy).
- [x] Viết unit tests cho `VelocityBuffer`, `TrajectoryExtrapolator`
- [x] Viết unit tests cho `MotionPredictor`
  - Test với constant velocity motion → verify linear extrapolation
  - Test với accelerating motion → verify quadratic extrapolation
  - Test confidence decay với increasing prediction horizon
- [x] Viết docstrings cho tất cả public methods trong VelocityBuffer và Extrapolator
- [x] Update `CLAUDE.md` với section Phase 2

---

### 2.5 — Integration (Phase 2 → Full Pipeline)

**Files cần update**: `src/pipeline.py`, `src/visualize.py`, `app.py`

**Việc cần làm**:

- [x] Thêm `MotionPredictor` vào `BlindSpotPipeline.__init__()`
- [x] Trong `process_frame()`:
  ```python
  current_time = time.time()
  tracks = self.track_manager.update(detections)
  predictions = {t.track_id: self.motion_predictor.update(t, current_time) for t in tracks}
  ```
- [x] Cập nhật `BlindSpotVisualizer`:
  - Vẽ predicted trajectory points (dots hoặc dashed line) trên frame
  - Hiển thị confidence score và alert level bên cạnh track ID
  - Color coding: xanh = safe, vàng = medium risk, đỏ = high risk
- [x] Alert generation logic trong `app.py`:
  - Log alert khi `prediction.alert_level in ["medium", "high"]` và `track.in_roi`
  - Optional: sound alert hoặc overlay warning banner
- [x] Full pipeline benchmark:
  - **Kết quả**: Detection (YOLOv9) chiếm ~80-90% latency trên CPU (~200-500ms/frame) → target 35ms/frame **không đạt được trên CPU**. Cần GPU (CUDA) để đạt ≥ 28 FPS.
  - Tracking + prediction + visualization mỗi bước < 2ms — đạt yêu cầu.
  - Script benchmark: `tools/benchmark.py` (chạy `python3 tools/benchmark.py --device 0` với GPU).
  - **Action**: Cập nhật target thành "≤ 35ms/frame trên GPU" trong Known Limitations.
- [x] Cập nhật CLI args trong `app.py`: `--prediction-horizon`, `--alert-threshold`

---

## Phase 3: System Integration (PRIORITY)

_Mục đích: Tích hợp toàn bộ hệ thống thành một pipeline hoàn chỉnh._

- [ ] **Kết nối YOLOv9 detector với tracking system**.
- [ ] **Pipeline flow**:
  - Input video frames
  - YOLOv9 inference
  - Hungarian matching
  - Kalman filter update
  - Track management
  - Motion prediction
  - Alert generation
- [ ] **API endpoints** để gửi video/frames.
- [ ] **Message queue** (nếu cần realtime processing).
- [ ] **Database integration** (lưu trữ tracks, alerts).
- [ ] **Dashboard backend** (serve data để visualization).
- [ ] **Error handling & logging**.
- [ ] **Performance monitoring**.
- [ ] **End-to-end testing**.

---

## Phase 4: Mock Data & Validation

_Mục đích: Tạo dữ liệu giả để testing và development._

- [ ] **Tạo mock camera frames** (video test data).
- [ ] **Mock YOLOv9 detection outputs**:
  - Simulated bounding boxes
  - Confidence scores
  - Class predictions
- [ ] **Mock vehicle trajectories**:
  - Different speeds (0-100 km/h)
  - Different paths (straight, turn, stop)
- [ ] **Mock scenarios**:
  - Vehicle enters blind spot
  - Vehicle exits blind spot
  - Multiple vehicles tracking
  - False positives/negatives
- [ ] **Dữ liệu CSV/JSON** cho historical testing.
- [ ] **Tạo unit test datasets**.

---

## Cấu Trúc Thư Mục Sau Khi Hoàn Thiện

```
src/
├── detector.py              ✅ (không đổi)
├── roi.py                   ✅ (không đổi)
├── visualize.py             🔶 update: hiển thị tracks, predictions
├── pipeline.py              🔶 update: tích hợp TrackManager + MotionPredictor
├── common/                  ✅ Phase 0 hoàn thiện
│   ├── models.py            ✅ Detection, Track, MotionPrediction, AlertEvent
│   ├── enums.py             ✅ TrackStatus, AlertLevel, AlertType
│   ├── dto.py               ✅ DTOs cho API
│   └── schemas.py           ✅ JSON schemas
└── tracking/
    ├── __init__.py          🔶 verify exports
    ├── types.py             🔶 thêm kalman field + velocity_buffer field
    ├── kalman_filter.py     🔶 thêm get_velocity()
    ├── matching.py          ✅ (không đổi)
    ├── track_manager.py     🔶 tích hợp Kalman predict/update
    └── motion/              🆕 toàn bộ mới
        ├── __init__.py
        ├── perspective.py   🆕 IPM / BEV transform
        ├── velocity_buffer.py
        ├── extrapolator.py
        └── predictor.py

tests/
├── test_tracking_smoke.py   🔶 expand
├── test_phase0_common_types.py ✅
├── test_kalman.py           🆕
├── test_matching.py         🆕
├── test_track_manager.py    🆕
├── test_perspective.py      🆕
├── test_velocity_buffer.py  ✅
├── test_extrapolator.py     ✅
└── test_motion_predictor.py 🆕
```

---

## Thứ Tự Thực Hiện Đề Xuất

```
1.3-integration  →  1.1-polish  →  1.2-polish  →  1.4-docs+tests  →  1.5-pipeline-integration
        ↓
Phase 0 (Shared Types)  →  Phase 3 (Integration)  →  Phase 4 (Mock Data)
        ↓
2.1-velocity-buffer  →  2.2-extrapolation  →  2.3-full-predictor  →  2.4-docs+tests  →  2.5-full-integration
```

**Lý do**: Task 1.3 (Kalman integration vào TrackManager) là dependency của mọi thứ khác — phải làm trước. Phase 0, 3, 4 có thể triển khai song song hoặc ngay sau khi có infrastructure cơ bản.

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

---

## Tuning Parameters Reference

| Parameter                     | Default | Range       | Ảnh hưởng                                                      |
| ----------------------------- | ------- | ----------- | -------------------------------------------------------------- |
| `kalman.process_noise`        | 1e-2    | 1e-4 – 1e-1 | Cao → trust detection hơn; thấp → trust prediction hơn         |
| `kalman.measurement_noise`    | 1e-1    | 1e-2 – 1.0  | Cao → smooth bbox; thấp → responsive to detections             |
| `track_manager.iou_threshold` | 0.3     | 0.1 – 0.6   | Thấp → match aggressively; cao → only match high overlap       |
| `track_manager.max_misses`    | 5       | 2 – 15      | Cao → tracks persist longer through occlusion                  |
| `track_manager.min_hits`      | 2       | 1 – 5       | Cao → fewer false tracks (need multiple detections to confirm) |
| `velocity_buffer.max_size`    | 10      | 5 – 20      | Cao → smoother velocity but slower to react to changes         |
| `motion.alert_threshold`      | 0.6     | 0.4 – 0.8   | Thấp → more alerts (higher sensitivity, more false positives)  |

---

## Key Design Decisions

1. **Relative Tracking (không cần CAN bus)**: Camera gắn cố định trên xe tải → ROI tĩnh trên pixel → vận tốc pixel = vận tốc tương đối. Ego-motion compensation là upgrade tương lai.

2. **Kalman state 8D vs 6D**: Chọn 8D `[cx,cy,w,h,vx,vy,vw,vh]` để track cả kích thước bbox, giúp phân biệt xe đến gần (bbox to ra) vs xe đứng yên.

3. **IoU cost matrix vs Euclidean distance**: IoU được chọn vì nó không cần calibration camera-specific, phù hợp khi không có thông tin 3D depth.

4. **Velocity buffer song song Kalman**: Kalman velocity phù hợp cho matching (1 frame ahead), còn velocity buffer với rolling regression phù hợp hơn cho prediction dài hạn (0.5-2s) vì ít bị noise của Kalman state.

5. **IPM/BEV trước Prediction**: Extrapolation tuyến tính `x(t) = x0 + vx*t` chỉ hợp lệ khi tỉ lệ pixel/mét đồng nhất. Camera perspective phá vỡ điều này → cần Homography transform sang BEV trước khi prediction. Fallback: chạy trên pixel gốc nếu chưa calibrate (chấp nhận sai lệch cho horizon > 0.5s).

6. **Distance estimation bằng Pinhole model**: `Z = (f * H_real) / h_pixel`. Không cần LiDAR/Radar. Thêm field `distance_m` vào `Track` để phục vụ alert logic.

7. **Confidence time-decay**: Exponential decay `e^(-λt)` là lựa chọn tự nhiên — uncertainty tăng theo thời gian khi không có observation mới.

8. **Không dùng deep learning tracker (DeepSORT)**: Để tránh dependency phức tạp và giữ latency thấp. IoU-based matching đủ tốt cho xe cộ vì chúng có motion tương đối mượt và không overlap nhiều.

---

## Phase 5: Ego-Motion Compensation (Future — Nâng cao)

_Mục đích: Xử lý chính xác các tình huống nguy hiểm nhất — xe tải rẽ phải/trái, phanh gấp, chuyển làn — nơi mà Relative Tracking (Phase 2) cho kết quả sai lệch._

> **Tại sao Phase này quan trọng?** Thống kê tai nạn cho thấy phần lớn va chạm điểm mù xảy ra khi xe tải **rẽ phải** — chính là lúc Relative Tracking bị sai nhiều nhất vì camera đang xoay. Phase 1-4 xử lý tốt khi đi thẳng (80% thời gian), Phase 5 xử lý 20% tình huống còn lại nhưng chiếm phần lớn rủi ro.

### Bài toán 2 luồng chuyển động

Khi xe tải rẽ phải:

```
Trên đời thực:                     Trên camera (pixel):
┌─────────────────┐                ┌─────────────────┐
│  Xe máy ĐỨNG YÊN│                │  Xe máy "LAO VÀO"│ ← gia tốc ảo
│  tại ngã tư      │                │  blind spot       │    do camera xoay
│                  │                │                   │
│  Xe tải RẼ PHẢI  │                │  ROI vẫn cố định  │
└─────────────────┘                └─────────────────┘
```

**Vấn đề**: System cảnh báo đúng (xe máy thực sự đang nguy hiểm) nhưng **dự đoán quỹ đạo sai** (vì pixel velocity thay đổi liên tục khi camera quay). Prediction 1-2s phía trước vô nghĩa.

### 5.1 — Phát hiện trạng thái xe tải (Ego-State Detection)

**Cách tiếp cận từ dễ → khó:**

| Phương pháp                  | Độ khó   | Mô tả                                                                                                    |
| ---------------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| **Background Optical Flow**  | ⭐⭐     | Tính optical flow của background (phần không có object). Nếu flow đồng nhất lớn → xe tải đang xoay/phanh |
| **Vanishing Point Tracking** | ⭐⭐⭐   | Theo dõi điểm hội tụ (vanishing point) trên frame. VP dịch chuyển = xe tải đang rẽ                       |
| **IMU Sensor**               | ⭐       | Gắn cảm biến gia tốc rẻ tiền (~50k VND). Cho biết trực tiếp gia tốc góc và gia tốc tuyến tính            |
| **CAN Bus**                  | ⭐⭐⭐⭐ | Đọc dữ liệu OBD-II của xe: vận tốc bánh, góc lái. Chính xác nhất nhưng cần hardware adapter              |

**Đề xuất cho MVP Phase 5**: Bắt đầu với **Background Optical Flow** (chỉ cần OpenCV, không cần hardware):

```python
class EgoMotionEstimator:
    def estimate_ego_rotation(self, prev_frame, curr_frame, object_masks) -> float:
        """Tính góc xoay của camera giữa 2 frame bằng background optical flow.
        - Mask ra vùng có object (từ YOLOv9 bboxes)
        - Tính optical flow trên phần background còn lại
        - Fit rotation model từ flow vectors
        Returns: estimated rotation angle (radians/frame)
        """

    def is_turning(self, threshold_rad: float = 0.01) -> bool:
        """True nếu xe tải đang rẽ (rotation > threshold)."""

    def compensate_velocity(self, pixel_velocity: Velocity, ego_rotation: float) -> Velocity:
        """Trừ đi thành phần vận tốc do camera xoay khỏi vận tốc pixel của object."""
```

### 5.2 — Bù trừ Ego-Motion cho Prediction

- [ ] Khi `is_turning() == True`: giảm `prediction_horizon` xuống 0.3s (thay vì 2s) vì prediction dài hạn không đáng tin cậy
- [ ] Trừ ego-rotation ra khỏi object velocity trước khi đưa vào VelocityBuffer
- [ ] Tăng alert sensitivity khi đang rẽ (vì đây là lúc nguy hiểm nhất)
- [ ] Hiển thị trạng thái "TURNING" trên dashboard

### 5.3 — Dynamic ROI (Khi xe rẽ)

Khi xe tải rẽ phải, vùng blind spot thực tế **mở rộng** (quét qua diện tích lớn hơn):

- [ ] Mở rộng ROI polygon tạm thời khi phát hiện xe đang rẽ
- [ ] Hoặc thêm "sweep zone" — vùng mà thân xe sẽ quét qua trong lúc rẽ
- [ ] Tính toán sweep zone dựa trên bán kính rẽ ước lượng từ ego_rotation

### 5.4 — Multi-Sensor Fusion (Optional — Cần Hardware)

- [ ] Tích hợp IMU sensor qua USB/Serial → real-time ego acceleration
- [ ] Tích hợp GPS module → ego velocity chính xác
- [ ] Fuse IMU + Camera optical flow bằng Extended Kalman Filter
- [ ] Tích hợp CAN Bus adapter (OBD-II) → steering angle, wheel speed

---

## Tổng Kết Roadmap Theo Mức Độ

```
Phase 0-4 (MVP):
  ✅ Detection + Tracking + Basic Prediction
  ✅ Hoạt động tốt khi xe tải đi thẳng (80% thời gian)
  ⚠️ Prediction sai khi rẽ, nhưng detection vẫn cảnh báo đúng (an toàn)

Phase 5 (Advanced):
  🔶 Ego-motion compensation
  🔶 Prediction chính xác cả khi rẽ phải/trái
  🔶 Dynamic ROI mở rộng khi xe rẽ
  🔶 Multi-sensor fusion (IMU/GPS/CAN)
```

---

## Known Limitations (Phase 1-4)

1. **CPU không đủ nhanh cho realtime**: YOLOv9 inference chiếm ~200-500ms/frame trên CPU → hệ thống yêu cầu GPU (CUDA) để đạt target ≤ 35ms/frame. Các bước tracking/prediction/visualization đều < 2ms. → Dùng `--device 0` khi chạy thực tế.

2. **Ego-motion không được bù trừ**: Khi xe tải rẽ gấp/phanh gấp, prediction sai lệch. Detection vẫn hoạt động (cảnh báo an toàn) nhưng quỹ đạo dự đoán không chính xác. → **Giải quyết ở Phase 5**.

3. **Homography cố định**: Nếu camera bị rung hoặc thay đổi góc, ma trận H sẽ sai → cần recalibrate. Giải pháp: auto-calibration bằng vanishing point detection.

4. **Không có 3D depth thực sự**: Distance estimation bằng pinhole model chỉ là xấp xỉ, phụ thuộc vào chiều cao thực của đối tượng (giả định cố định).

5. **Turning scenarios là quan trọng nhất nhưng khó nhất**: Phần lớn tai nạn xảy ra khi rẽ, nhưng đây cũng là lúc prediction kém chính xác nhất. Phase 5 được thiết kế để giải quyết điểm yếu này theo hướng incremental (từ software-only đến multi-sensor).
