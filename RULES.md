# RULES.md — Project Rules for Truck Blind Spot Detection

> Tài liệu này mô tả các quy tắc, convention và hướng dẫn bắt buộc khi phát triển project **Truck Blind Spot Detection**. Mọi agent (Claude Code, Gemini, v.v.) và developer PHẢI tuân thủ khi đóng góp code.

---

## 1. Ngôn Ngữ & Phong Cách Code

### 1.1 Python Version & Type Hints
- **Python 3.10+** bắt buộc (dùng `from __future__ import annotations` ở đầu mỗi file)
- **Tất cả** function và method phải có **type hints** đầy đủ cho parameters và return type
- Dùng `Tuple`, `List`, `Optional`, `Dict` từ `typing` module (tương thích Python 3.9)
- Dùng `dataclass` cho data containers, **không** dùng plain dict hoặc NamedTuple

```python
# ✅ ĐÚNG
def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:

# ❌ SAI — thiếu type hints
def process_frame(self, frame):
```

### 1.2 Comments & Docstrings
- **Comments bằng tiếng Việt** — đây là convention của project
- **Docstrings bằng tiếng Việt hoặc tiếng Anh** — tùy context, nhưng phải nhất quán trong cùng 1 file
- Chỉ comment logic phức tạp, KHÔNG comment code hiển nhiên
- Mỗi class và public method phải có docstring

```python
# ✅
# Lấy vị trí tâm/đáy của bbox để kiểm tra trong ROI
detection.anchor_point = self.roi.get_reference_point(detection.bbox)

# ❌ — không comment code hiển nhiên
# Gán giá trị True
detection.in_roi = True
```

### 1.3 Naming Convention
- **Classes**: `PascalCase` — `BlindSpotPipeline`, `TrackManager`, `BoundingBoxKalmanFilter`
- **Functions/methods**: `snake_case` — `process_frame`, `match_tracks_detections`
- **Constants**: `UPPER_SNAKE_CASE` — `PROJECT_ROOT`, `IMAGE_EXTENSIONS`
- **Private methods**: prefix `_` — `_resolve_path`, `_create_track`, `_prune_dead_tracks`
- **Type aliases**: `PascalCase` — `FloatBBox`, `Velocity`, `Point`

---

## 2. Kiến Trúc & Module Structure

### 2.1 Phân Tách Module
```
src/
├── detector.py          # YOLOv9 detection ONLY — không mix tracking logic vào đây
├── roi.py               # ROI geometry ONLY
├── visualize.py         # Rendering annotations ONLY
├── pipeline.py          # Orchestrator — kết nối tất cả module
└── tracking/            # Tracking & prediction — TÁCH BIỆT khỏi YOLOv9
    ├── __init__.py
    ├── types.py          # Dataclasses dùng chung (Track, FloatBBox, ...)
    ├── kalman_filter.py  # Kalman filter thuần (không phụ thuộc detector)
    ├── matching.py       # Hungarian assignment (không phụ thuộc detector)
    ├── track_manager.py  # Lifecycle management
    └── motion/           # Motion prediction (Phase 2)
```

**Nguyên tắc bắt buộc**:
- `src/tracking/` **KHÔNG ĐƯỢC import** bất cứ thứ gì từ `yolov9/` hoặc `src/detector.py`
- `src/tracking/` chỉ nhận input thông qua **dataclass** (`Detection`, `Track`) — không nhận raw tensors
- `src/detector.py` là nơi **DUY NHẤT** inject `yolov9/` vào `sys.path`
- `pipeline.py` là **DUY NHẤT** nơi kết nối detector ↔ tracking ↔ visualizer

### 2.2 Dependency Direction (Một chiều)
```
detector.py ←── pipeline.py ──→ tracking/
     ↑                              ↑
     │                              │
  yolov9/                       (standalone)
```
- `tracking/` phải chạy được **hoàn toàn độc lập** mà không cần YOLOv9
- Test tracking module phải pass mà **không cần GPU hay model weights**

### 2.3 Data Flow Bắt Buộc
```
Frame → Detector.predict() → List[Detection]
      → ROI.check()        → Detection.in_roi, .zone_name, .risk_level
      → TrackManager.update() → List[Track]
      → MotionPredictor.update() → List[MotionPrediction]
      → Visualizer.draw() → Annotated Frame
```
Mỗi bước nhận **output của bước trước** dưới dạng dataclass, KHÔNG truy cập internal state.

---

## 3. Path Handling

- **LUÔN** dùng `pathlib.Path`, KHÔNG dùng `os.path`
- Dùng `PROJECT_ROOT = Path(__file__).resolve().parents[N]` để xác định root
- Dùng `_resolve_path()` static method để convert relative → absolute
- **KHÔNG** hardcode absolute paths trong code

```python
# ✅ ĐÚNG
PROJECT_ROOT = Path(__file__).resolve().parents[1]
weights = PROJECT_ROOT / "weights" / "best_small.pt"

# ❌ SAI
weights = "/home/taitu/Documents/GitHub/truck_blind_spot/weights/best_small.pt"
```

---

## 4. Error Handling

- Dùng **`RuntimeError`** cho lỗi runtime (video không mở được, model load fail)
- Dùng **`FileNotFoundError`** cho file/path không tồn tại
- Dùng **`ValueError`** cho input không hợp lệ (empty frame, invalid bbox)
- **Luôn** cung cấp error message rõ ràng bằng tiếng Việt
- **Luôn** validate input ở đầu function trước khi xử lý

```python
if frame is None or frame.size == 0:
    raise ValueError("Khung hình đầu vào rỗng (empty).")
```

---

## 5. YOLOv9 Integration Rules

- **KHÔNG** sửa bất kỳ file nào trong `yolov9/` — đây là vendored code
- Nếu cần utility từ YOLOv9, import qua `detector.py` hoặc copy logic ra module riêng
- Import YOLOv9 chỉ qua pattern `sys.path.append()` đã có sẵn trong `detector.py`
- Model weights file: `weights/best_small.pt` — KHÔNG commit file .pt vào git (đã có .gitignore)

---

## 6. Tracking Module Rules

### 6.1 Track Lifecycle
```
BIRTH:      Detection không match với track nào → tạo Track mới
            Track mới có is_confirmed=False cho đến khi hits >= min_hits
CONFIRMED:  hits >= min_hits → is_confirmed=True → hiển thị trên UI
UPDATE:     Match thành công → cập nhật bbox, confidence, hits++, misses=0
MISS:       Không match ở frame hiện tại → misses++
DEATH:      misses > max_misses → xóa track
```

### 6.2 Kalman Filter
- State vector 8D: `[cx, cy, w, h, vx, vy, vw, vh]` — KHÔNG thay đổi kích thước state
- Dùng linear constant-velocity model — adequate cho vehicle tracking ở 25-30 FPS
- Bbox format nội bộ: `(cx, cy, w, h)` — convert từ/đến `(x1, y1, x2, y2)` ở boundary
- `process_noise` và `measurement_noise` phải là configurable parameters

### 6.3 Hungarian Assignment
- Cost matrix = `1 - IoU_matrix` (IoU cost, KHÔNG dùng Euclidean distance)
- Tối thiểu IoU threshold 0.3 để được coi là match hợp lệ
- Dùng `scipy.optimize.linear_sum_assignment` — KHÔNG tự implement Hungarian

### 6.4 Velocity & Motion
- Velocity đơn vị: **pixels/frame** (từ Kalman) hoặc **pixels/second** (từ VelocityBuffer)
- Phải ghi rõ đơn vị trong docstring
- Motion prediction horizon: 0.5s → 2.0s maximum
- Confidence score: [0.0, 1.0] — 0 = không tin cậy, 1 = rất tin cậy

---

## 7. Configuration Management

- Config files đặt trong `configs/` folder
- Dùng `.json` cho geometry configs (ROI vertices)
- Dùng `.yaml` cho class mappings
- **KHÔNG** hardcode hyperparameters — luôn nhận qua constructor params với default values
- Tất cả thresholds phải có giá trị default hợp lý:

| Parameter | Default | Acceptable Range |
|-----------|---------|-----------------|
| `conf_threshold` | 0.25 | 0.1 – 0.8 |
| `iou_threshold` | 0.45 | 0.2 – 0.7 |
| `track_iou_threshold` | 0.3 | 0.1 – 0.6 |
| `max_misses` | 5 | 2 – 15 |
| `min_hits` | 2 | 1 – 5 |
| `process_noise` | 1e-2 | 1e-4 – 1e-1 |
| `measurement_noise` | 1e-1 | 1e-2 – 1.0 |

---

## 8. Testing Rules

### 8.1 Test Structure
- Tất cả tests đặt trong `tests/` folder ở project root
- File naming: `test_<module_name>.py`
- Dùng `pytest` framework — KHÔNG dùng `unittest`
- Chạy: `python -m pytest tests/ -v`

### 8.2 Test Requirements
- Tracking tests **KHÔNG cần GPU, KHÔNG cần model weights, KHÔNG cần video file**
- Dùng mock data (synthetic bboxes, fake detections) cho unit tests
- Mỗi module tracking phải có ít nhất:
  - Happy path test
  - Edge case tests (empty input, single element, no match)
  - Numerical correctness test (Kalman predict → known results)

### 8.3 Imports Trong Tests
```python
# ✅ Import trực tiếp — tests/ nằm ngoài src/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tracking.kalman_filter import BoundingBoxKalmanFilter
from tracking.track_manager import TrackManager
```

---

## 9. Performance Constraints

- **Target FPS**: ≥ 25 FPS trên GPU, ≥ 10 FPS trên CPU
- Detection là bottleneck chính (~80% thời gian) — tracking code phải chạy nhanh
- Tracking + prediction: **≤ 5ms/frame** (excluding detection)
- Tránh Python for-loops trên large arrays — dùng NumPy vectorized operations
- `deque(maxlen=N)` cho buffers, KHÔNG dùng `list` rồi cắt thủ công

---

## 10. Git & Commit Rules

- **KHÔNG commit**: model weights (`.pt`), video files, `__pycache__/`, `venv/`, `.DS_Store`
- Commit message format: `[module] mô tả ngắn gọn` — ví dụ: `[tracking] integrate Kalman into TrackManager`
- Một commit nên focus vào **một thay đổi logic** — không gộp nhiều feature

---

## 11. Visualization Rules

- Bounding box color: **Xanh lá** = safe (ngoài ROI), **Đỏ** = danger (trong blind spot)
- Track ID hiển thị bên cạnh bbox khi track `is_confirmed=True`
- Predicted trajectory vẽ bằng **đường nét đứt (dashed line)** với màu vàng
- FPS counter luôn hiển thị ở **góc trên trái**
- Alert/warning text hiển thị ở **góc trên phải** khi có object trong blind spot
- Font: `cv2.FONT_HERSHEY_SIMPLEX` — consistent toàn project

---

## 12. Safety & ADAS Constraints

- Hệ thống này là **cảnh báo (warning)**, KHÔNG phải **tự động phanh (autonomous braking)**
- **FALSE NEGATIVE nguy hiểm hơn FALSE POSITIVE**: thà cảnh báo dư còn hơn bỏ sót
- Vì vậy:
  - `conf_threshold` nên giữ thấp (0.25) để không bỏ sót detection
  - `max_misses` nên giữ cao (5-10) để track không chết quá nhanh khi bị occlude tạm thời
  - `min_hits` nên giữ thấp (1-2) để cảnh báo sớm
- Alert **PHẢI** được kích hoạt khi: object `in_roi=True` AND `is_confirmed=True`
- Logging: in ra console mỗi alert event với timestamp, track_id, class_name, risk_level

---

## 13. Checklist Trước Khi Commit Feature Mới

- [ ] Type hints đầy đủ cho tất cả public functions
- [ ] Docstring cho mỗi class và public method
- [ ] Ít nhất 1 unit test cho happy path
- [ ] Code chạy mà KHÔNG cần GPU (fallback CPU)
- [ ] KHÔNG import trực tiếp từ `yolov9/` (trừ `detector.py`)
- [ ] KHÔNG hardcode absolute paths
- [ ] Default values hợp lý cho mọi parameter
- [ ] Error messages bằng tiếng Việt
- [ ] `python -m pytest tests/ -v` pass tất cả
