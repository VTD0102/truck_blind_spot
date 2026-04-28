# Truck Blind Spot Detection

Hệ thống ADAS phát hiện và theo dõi vật thể trong vùng điểm mù xe tải theo thời gian thực, sử dụng **YOLOv9** + **Kalman Tracking** + **Motion Prediction**.

## Tổng Quan

Pipeline xử lý mỗi frame theo 4 bước:

```
Frame → YOLOv9 Detection → Multi-zone ROI → Kalman Tracking → Motion Prediction → Visualization
```

**Tính năng chính:**
- Phát hiện 6 class đối tượng: `person, bike, motor, car, truck, bus`
- Multi-zone ROI với mức độ nguy hiểm riêng cho từng vùng (`risk_level`)
- Tracking liên tục qua các frame bằng Kalman Filter + Hungarian Algorithm
- Dự đoán quỹ đạo chuyển động tại 0.5s / 1.0s / 2.0s phía trước
- Cảnh báo có độ tin cậy (confidence-based alert) với 4 mức: none / low / medium / high
- Hỗ trợ 2 profile camera: `front_camera`, `rear_camera`

## Cài Đặt

```bash
git clone https://github.com/VTD0102/truck_blind_spot.git
cd truck_blind_spot
python3 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Yêu cầu:** Python 3.10+, PyTorch ≥ 2.0, OpenCV ≥ 4.8

## Cách Chạy

### 1. Kiểm tra file cần thiết

Demo mặc định dùng các file sau:

| Loại | Mặc định |
|------|----------|
| Model weights | `weights/best_roiv2.pt` |
| Video input | `assets/videos/demo4.mp4` |
| ROI config | `configs/roi.json` |
| ROI profile | `front_camera` |
| Class config | `configs/classes.yaml` |

Nếu dùng file khác, truyền đường dẫn tương đối từ project root qua CLI.

### 2. Chạy demo mặc định

```bash
python app.py
```

`app.py` mở cửa sổ OpenCV để hiển thị kết quả realtime, vì vậy cần chạy trong môi trường có GUI/display. Phím điều khiển:

| Phím | Chức năng |
|------|-----------|
| `p` | Tạm dừng / tiếp tục |
| `r` | Chạy lại video từ đầu |
| `q` | Thoát |

### 3. Chạy với input/output khác

```bash
# Chạy video khác và lưu kết quả
python app.py --source assets/videos/demo.mp4 --output outputs/result.mp4

# Chạy webcam
python app.py --source 0

# Lặp lại video file khi đọc đến cuối
python app.py --loop

# Chọn profile ROI khác
python app.py --roi-profile rear_camera

# Dùng weights / ROI / classes config khác
python app.py \
  --weights weights/best_6k.pt \
  --roi configs/roi.json \
  --classes-config configs/classes.yaml
```

### 4. Chọn thiết bị và điều chỉnh ngưỡng

```bash
# CPU
python app.py --device cpu

# GPU CUDA đầu tiên
python app.py --device 0

# Điều chỉnh ngưỡng detection
python app.py --conf-thres 0.3 --iou-thres 0.5

# Điều chỉnh motion prediction
python app.py --prediction-horizon 1.0 --alert-threshold 0.5
```

### 5. Chạy tests

```bash
python -m pytest tests/ -v
```

Tests không cần GPU, weights `.pt`, hoặc video thật.

## Cấu Trúc Project

```
truck_blind_spot/
├── app.py                       # CLI entry point chính
├── requirements.txt
│
├── src/
│   ├── detector.py              # YOLOv9Detector — inference, NMS, scale_boxes
│   ├── roi.py                   # MultiPolygonROI — multi-zone, risk_level
│   ├── visualize.py             # BlindSpotVisualizer — vẽ zone, track, trajectory
│   ├── pipeline.py              # BlindSpotPipeline — orchestrator
│   ├── roi_evaluation.py        # Công cụ đánh giá recall theo zone
│   ├── common/
│   │   ├── models.py            # Detection, Track, MotionPrediction, AlertEvent
│   │   └── enums.py             # TrackStatus, AlertLevel, AlertType
│   └── tracking/
│       ├── kalman_filter.py     # BoundingBoxKalmanFilter (state 8D)
│       ├── matching.py          # match_tracks_detections (Hungarian + IoU)
│       ├── track_manager.py     # TrackManager (birth / update / death)
│       └── motion/
│           ├── velocity_buffer.py   # VelocityBuffer — rolling regression
│           ├── extrapolator.py      # TrajectoryExtrapolator — quadratic extrapolation
│           ├── perspective.py       # PerspectiveTransform — IPM / BEV
│           └── predictor.py         # MotionPredictor — wrapper, alert logic
│
├── configs/
│   ├── classes.yaml             # 6 classes: person, bike, motor, car, truck, bus
│   ├── roi.json                 # Multi-zone ROI (front_camera / rear_camera)
│   └── blindspot.yaml           # Dataset config cho training/eval
│
├── weights/
│   ├── best_roiv2.pt            # Checkpoint chính (mặc định cho app.py)
│   ├── best_6k.pt
│   └── best_pilot_4k5.pt
│
├── assets/videos/               # Video test
├── outputs/                     # Kết quả xuất ra
├── tests/                       # pytest — 59 tests, không cần GPU
├── CLAUDE.md                    # Hướng dẫn cho Claude Code
├── AGENTS.md                    # Hướng dẫn cho AI agents
├── RULES.md                     # Quy tắc bắt buộc của project
└── Plan.md                      # Roadmap chi tiết theo phase
```

## Pipeline Chi Tiết

### 1. Detection (`src/detector.py`)
`YOLOv9Detector` load model qua `DetectMultiBackend`, chạy letterbox preprocess → inference → NMS → `scale_boxes`. Trả về `List[Detection]` với `bbox (x1,y1,x2,y2)`, `confidence`, `class_id`, `class_name`.

### 2. ROI (`src/roi.py`)
`MultiPolygonROI` đọc `configs/roi.json`, scale polygon theo kích thước frame thực tế. Mỗi zone có `name`, `risk_level`, `color`. Kiểm tra `anchor_point` (bottom-center của bbox) để xác định `in_roi` và `zone_name`.

### 3. Tracking (`src/tracking/`)
- **Kalman Filter**: state vector 8D `[cx, cy, w, h, vx, vy, vw, vh]` — track cả vị trí lẫn kích thước bbox
- **Hungarian Algorithm**: gán detection vào track dùng IoU cost matrix
- **TrackManager**: quản lý vòng đời track (TENTATIVE → CONFIRMED → LOST), tự động xóa track chết
- **Velocity**: dùng Kalman velocity cho frame đầu, chuyển sang buffer smoothed velocity khi có ≥ 5 frames

### 4. Motion Prediction (`src/tracking/motion/`)
- **VelocityBuffer**: rolling buffer 10 frames, tính velocity/acceleration bằng least-squares regression
- **TrajectoryExtrapolator**: `x(t) = x0 + vx·t + ½·ax·t²`, confidence = tracking_quality × consistency × smoothness × exp(-λt)
- **MotionPredictor**: dự đoán vị trí tại `[0.5s, 1.0s, 2.0s]`, motion validation (max speed, direction change, bbox size), alert level

### 5. Visualization (`src/visualize.py`)
Vẽ: zone polygon (alpha overlay) · detection bbox + label · track ID + velocity arrow · trajectory dots (màu theo alert level) · confidence/alert label · cảnh báo banner.

**Color coding:**
| Alert level | Màu |
|-------------|-----|
| high | Đỏ |
| medium | Cam |
| low | Vàng |
| none / ngoài ROI | Xanh lá |

## Cấu Hình

### Hyperparameters mặc định

| Tham số | Mặc định | CLI arg |
|---------|---------|---------|
| Confidence threshold | 0.25 | `--conf-thres` |
| NMS IoU threshold | 0.45 | `--iou-thres` |
| Track IoU threshold | 0.3 | — |
| Max misses trước khi xóa track | 5 | — |
| Min hits để xác nhận track | 2 | — |
| Prediction horizon | 1.0s | `--prediction-horizon` |
| Alert confidence threshold | 0.6 | `--alert-threshold` |

### ROI Config (`configs/roi.json`)

Schema: `profiles.{front_camera,rear_camera}.zones[].{name, risk_level, color, polygon}`

Polygon tự động scale theo kích thước frame thực tế — không cần chỉnh thủ công khi đổi resolution.

## Tests

```bash
python -m pytest tests/ -v          # 59 tests, không cần GPU hoặc weights
```

## Đánh Giá ROI Recall

```bash
python -m src.roi_evaluation \
  --weights weights/best_6k.pt \
  --data configs/blindspot.yaml \
  --roi configs/roi.json \
  --roi-profile front_camera \
  --split val --conf-thres 0.25 \
  --output-dir outputs/roi_eval
```

## Roadmap

| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| 0 | Shared types, enums, DTOs | ✅ Hoàn thành |
| 1 | Kalman Filter + Hungarian + TrackManager | ✅ Hoàn thành |
| 2 | MotionPredictor + trajectory visualization | ✅ Hoàn thành |
| 3 | API endpoints, message queue, database | Chưa bắt đầu |
| 4 | Mock data & validation | Chưa bắt đầu |
| 5 | Ego-motion compensation (advanced) | Chưa bắt đầu |

Chi tiết từng phase xem `Plan.md`.
