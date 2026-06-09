# Truck Blind Spot Detection — Câu lệnh chạy

Tài liệu này tổng hợp tất cả câu lệnh thường dùng trong dự án.

---

## Cài đặt môi trường

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Chạy app chính (`app.py`)

### CPU (mặc định)

```bash
python3 app.py
```

### Apple Silicon GPU — MPS (Phase 1)

Dùng cho macOS M1/M2/M3. Không cần export trước.

```bash
python3 app.py --device mps
```

### Apple Neural Engine — CoreML (Phase 2)

Cần export model một lần trước:

```bash
# Bước 1: Export (chạy một lần)
python3 tools/export_coreml.py --weights weights/best_roiv2.pt

# Bước 2: Chạy app với CoreML backend
python3 app.py --backend coreml
```

### Tùy chọn thêm

```bash
# Đổi ROI profile (camera sau xe)
python3 app.py --roi-profile rear_camera

# Nguồn video / webcam
python3 app.py --source assets/videos/demo4.mp4
python3 app.py --source 0                       # webcam index 0

# Lặp lại video
python3 app.py --loop

# Lưu kết quả ra file
python3 app.py --output outputs/result.mp4

# Thay đổi ngưỡng detection
python3 app.py --conf-thres 0.3 --iou-thres 0.5

# Bật frame skipping khi FPS thấp (Kalman predict-only cho frame bị skip)
python3 app.py --device mps --enable-frame-skip

# Bật tự động giảm resolution khi FPS < target
python3 app.py --device mps --enable-adaptive-scale

# Đặt target FPS (default 25)
python3 app.py --device mps --target-fps 30

# Ghi log cảnh báo ra CSV
python3 app.py --device mps --alert-log outputs/alerts.csv

# Chạy không hiển thị cửa sổ (headless)
python3 app.py --no-display --output outputs/result.mp4

# Kết hợp MPS + tất cả optimizations
python3 app.py --device mps --enable-frame-skip --enable-adaptive-scale --target-fps 30
```

Phím tắt trong cửa sổ: `p` pause/resume · `r` restart · `q` thoát.

---

## Pipeline CLI (`src/pipeline.py`)

Chạy inference trên ảnh hoặc video, không cần cửa sổ `app.py`.

```bash
# Ảnh
python3 src/pipeline.py --source path/to/image.jpg --show

# Video
python3 src/pipeline.py --source assets/videos/demo4.mp4 --output outputs/out.mp4 --show
```

---

## Export CoreML

```bash
python3 tools/export_coreml.py --weights weights/best_roiv2.pt
# Output: weights/best_roiv2.onnx + weights/best_roiv2.mlpackage/
```

---

## Đánh giá ROI recall (`src/roi_evaluation.py`)

```bash
python -m src.roi_evaluation \
  --weights weights/best_roiv2.pt \
  --data configs/blindspot.yaml \
  --roi configs/roi.json \
  --roi-profile front_camera \
  --split val --conf-thres 0.25 --iou-match 0.5 \
  --output-dir outputs/roi_eval
```

---

## Tests

```bash
# Toàn bộ test suite (không cần GPU hay weights)
python -m pytest tests/ -v --ignore=tests/test_demo_notebook.py

# Chỉ tracking tests
python -m pytest tests/test_tracking_smoke.py -v

# Chỉ backend tests (MPS + CoreML)
python -m pytest tests/test_detector_backends.py -v
```
