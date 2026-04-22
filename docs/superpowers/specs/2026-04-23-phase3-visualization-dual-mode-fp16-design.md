# Phase 3 Design — Visualization Compact + Dual Mode + FP16

**Date:** 2026-04-23  
**Scope:** Cải thiện demo visualization, thêm 2 chế độ chạy (realtime/offline), tối ưu FP16 trên GPU.  
**Files thay đổi:** `src/detector.py`, `src/visualize.py`, `src/pipeline.py`, `app.py`

---

## 1. Dual Mode (Realtime vs Offline)

### Mô tả

Hai chế độ chạy được chọn qua CLI arg `--mode {realtime,offline}`.

| | Realtime Mode | Offline Mode |
|---|---|---|
| Mặc định khi | `--show` hoặc source là webcam | `--output` không có `--show` |
| FP16 | Tự bật nếu CUDA có mặt | FP32 (mặc định) |
| Frame drop | Có — skip frame cũ khi xử lý không kịp | Không — xử lý 100% frames |
| Visualization | Compact (font nhỏ, label ngắn) | Full (label đầy đủ, trajectory) |
| Mục tiêu FPS | ≥25 FPS live display | Xuất file nhanh nhất |

### Frame dropping (Realtime)

Dùng `time.perf_counter()` để đo thời gian xử lý mỗi frame. Nếu processing time > `1/fps_target`, bỏ qua frame tiếp theo (gọi `cap.grab()` thay vì `cap.read()`). Giữ display smooth.

### CLI

```bash
# Realtime (mặc định khi --show)
python app.py --source assets/videos/demo.mp4 --show

# Realtime explicit
python app.py --source 0 --mode realtime --show

# Offline
python app.py --source assets/videos/demo.mp4 --mode offline --output outputs/result.mp4
```

---

## 2. FP16 Half-Precision Inference

### Thay đổi `src/detector.py`

Thêm `half: bool = False` vào `YOLOv9Detector.__init__()`:

```python
def __init__(self, ..., half: bool = False) -> None:
    ...
    self.half = half and str(device) != 'cpu'
    if self.half:
        self.model.half()
```

Khi inference:
```python
img_tensor = img_tensor.half() if self.half else img_tensor.float()
```

Fallback: nếu FP16 raise exception (CPU không hỗ trợ), tự động về FP32 với `logging.warning`.

### Auto-enable trong Pipeline

`BlindSpotPipeline.__init__()` nhận `mode: str = "realtime"`. Khi `mode == "realtime"` và device có CUDA: tự set `half=True` cho detector.

---

## 3. Visualization Compact

### Font & Label

| Thuộc tính | Hiện tại | Sau |
|------------|---------|-----|
| `font_scale` | `0.6` | `0.42` |
| Label detection | `left_blind_spot [high] \| car 0.92` | `L.BLIND car #3 [H]` |
| Label track ID | Riêng biệt dưới bbox | Gộp vào label detection |
| Confidence số | Hiển thị `conf:0.42` | Bỏ — chỉ giữ letter code |
| Trajectory label | `conf:0.42 [medium]` mỗi track | Bỏ hẳn — chỉ giữ màu dot |

**Zone name rút gọn** (mapping trong `BlindSpotVisualizer`):
- `left_blind_spot` → `L.BLIND`
- `right_blind_spot` → `R.BLIND`
- `forward_danger_zone` → `FORWARD`
- Tên khác: lấy 8 ký tự đầu viết hoa

**Alert letter code:**
- `high` → `[H]`
- `medium` → `[M]`
- `low` → `[L]`
- `none` → không hiện code

**Realtime mode**: dùng label compact.  
**Offline mode**: giữ label đầy đủ như hiện tại (truyền `compact=True/False` vào `draw()`).

### Flash Effect

`BlindSpotVisualizer` thêm field `_flash_counter: int = 0`.

Logic trong `draw()`:
1. Tính `max_alert = max alert_level` của tất cả tracks trong frame (ưu tiên high > medium > low > none)
2. Nếu `max_alert in ("high", "medium")`: `_flash_counter += 1`, else reset về 0
3. Flash pattern:
   - HIGH: `4 frames ON / 4 frames OFF` (`_flash_counter % 8 < 4`)
   - MEDIUM: `8 frames ON / 8 frames OFF` (`_flash_counter % 16 < 8`)
4. Khi flash ON:
   - Phủ overlay màu lên toàn frame: HIGH → đỏ `(0,0,255)` alpha `0.25`, MEDIUM → cam `(0,100,255)` alpha `0.15`
   - Vẽ banner **"⚠ NGUY HIEM"** (HIGH) hoặc **"CANH BAO"** (MEDIUM) căn giữa phía trên frame, font lớn `1.2`, màu trắng nền đỏ/cam

**Thứ tự vẽ:** zones → detections → tracks → predictions → flash overlay → banner → FPS counter

Flash state reset khi `reset()` được gọi (nếu có) hoặc tự tắt sau 60 frames không có alert.

---

## 4. Error Handling & Logging

- Dùng Python `logging` module thay vì `print()` trong `app.py`
- Log level: `INFO` cho frame processing, `WARNING` cho FP16 fallback, `ERROR` cho video open fail
- Alert events: `logging.warning(f"[ALERT] Track {id}: {level} | zone={zone}")` — giữ format hiện tại

---

## 5. Testing

- `tests/test_visualizer_flash.py` — unit test flash counter state machine:
  - No alert → counter stays 0
  - HIGH alert N frames → counter increments, flash ON at correct frames
  - Alert clears → counter resets after 60 frames
- Existing 59 tests phải pass không thay đổi
- Manual validation: chạy `python app.py --source assets/videos/demo.mp4 --show` và quan sát flash

---

## 6. Files Thay Đổi

| File | Thay đổi |
|------|---------|
| `src/detector.py` | Thêm `half: bool` param, FP16 inference path |
| `src/visualize.py` | `font_scale`, compact labels, zone shortening, flash state + overlay |
| `src/pipeline.py` | Thêm `mode` param, auto FP16, `compact` flag cho visualizer |
| `app.py` | `--mode` CLI arg, frame drop logic, logging module |
| `tests/test_visualizer_flash.py` | Unit tests flash state machine |

---

## 7. Non-Goals (Không làm trong Phase này)

- Backend API / database / message queue
- Web dashboard
- Batch inference
- Model quantization (INT8, TensorRT)
- Ego-motion compensation (Phase 5)
