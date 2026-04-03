# 🚗 Truck Blind Spot Detection

Hệ thống phát hiện vật thể trong vùng điểm mù xe tải sử dụng **YOLOv9** - một giải pháp AI cho an toàn giao thông.

## 📋 Tổng Quan

Hệ thống này tự động phát hiện và cảnh báo khi có vật thể (người, xe đạp, ô tô, mô tô) xuất hiện trong vùng điểm mù của xe tải:

- **Phát hiện đối tượng**: Sử dụng model YOLOv9-Small được fine-tune trên bộ dữ liệu xe cộ
- **Xác định vùng nguy hiểm**: Kiểm tra xem đối tượng có nằm trong vùng ROI (Region of Interest) - điểm mù không?
- **Cảnh báo realtime**: Hiển thị bbox đỏ và nhãn "BLIND SPOT" khi phát hiện vật thể nguy hiểm
- **4 class đối tượng**: Person, Bicycle, Car, Motorcycle

### Ứng dụng

- 🚛 Hỗ trợ an toàn lái xe xe tải (ADAS - Advanced Driver Assistance System)
- 📹 Tích hợp vào hệ thống camera giám sát
- 🎓 Đồ án AI, Computer Vision cho sinh viên

## 🏗️ Cấu Trúc Project

```
truck_blind_spot/
├── README.md                    # Tài liệu này
├── app.py                       # Script chạy demo video chính
├── requirements.txt             # Danh sách package Python
│
├── configs/                     # Cấu hình hệ thống
│   ├── classes.yaml             # Định nghĩa 4 class đối tượng
│   └── roi.json                 # Polygon ROI (điểm mù) cho frame 1280x720
│
├── src/                         # Source code chính
│   ├── detector.py              # YOLOv9Detector - load model, inference, NMS
│   ├── roi.py                   # PolygonROI - kiểm tra điểm trong polygon
│   ├── visualize.py             # BlindSpotVisualizer - vẽ bbox, ROI, warning
│   └── pipeline.py              # BlindSpotPipeline - điều phối toàn bộ pipeline
│
├── weights/                     # Model weights
│   └── best_small.pt            # YOLOv9-Small trained (mAP@0.5=0.70)
│
├── assets/videos/               # Video demo
│   ├── demo.mp4
│   ├── demo2.mp4
│   └── ...
│
├── report/                      # Báo cáo chi tiết
│   ├── dataset_report.md        # Chuẩn bị dataset
│   ├── training_report.md       # Kết quả huấn luyện model
│   └── inference_report.md      # Chi tiết inference & hiệu năng
│
└── yolov9/                      # YOLOv9 source code (upstream)
    ├── models/
    ├── utils/
    ├── train.py
    ├── detect.py
    └── ...
```

## 🔧 Yêu Cầu

- **Python**: 3.8 trở lên
- **PyTorch**: >= 2.0 (với CUDA 11.8+ nếu chạy GPU)
- **OpenCV**: >= 4.8.0
- **CUDA** (optional): Để tăng tốc độ inference trên GPU

### Thông số chi tiết

Xem file `requirements.txt`:
- PyTorch, torchvision
- OpenCV, NumPy, PIL
- Matplotlib, Pandas, Seaborn (cho phân tích)
- PyYAML (cấu hình)

## 📦 Cài Đặt

### 1. Clone repository

```bash
git clone https://github.com/VTD0102/truck_blind_spot.git
cd truck_blind_spot
```

### 2. Tạo virtual environment (optional nhưng recommended)

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# hoặc
venv\Scripts\activate       # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Chạy Demo

### Chạy demo video đơn giản

```bash
python3 app.py
```

Chương trình sẽ:
- Load model YOLOv9-Small từ `weights/best_small.pt`
- Đọc video từ `assets/videos/demo.mp4`
- Xử lý từng frame và hiển thị realtime
- Hiển thị FPS ở góc trên trái

### Điều khiển

| Phím | Chức năng |
|------|----------|
| `p`  | Tạm dừng / Tiếp tục (Pause/Resume) |
| `q`  | Thoát chương trình |

### Sử dụng CLI (tùy chọn)

Pipeline hỗ trợ CLI để xử lý ảnh, video, hoặc camera:

```bash
python3 src/pipeline.py --source assets/videos/demo.mp4 --output output.mp4
python3 src/pipeline.py --source 0  # Sử dụng webcam
```

## 🧠 Pipeline Hoạt Động

Các bước xử lý cho mỗi frame:

```
┌─────────────────┐
│  Input Frame    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│   YOLOv9Detector         │
│ - Preprocess frame       │
│ - Run inference          │
│ - NMS filtering          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   Detection Results      │
│ [bbox, conf, class_id]   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│     PolygonROI Check     │
│ - Get bbox check point   │
│ - Test point in polygon  │
│ - Mark: Safe/Blind Spot  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  BlindSpotVisualizer     │
│ - Draw ROI polygon       │
│ - Draw bbox (green/red)  │
│ - Add labels & warning   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Output Frame + FPS      │
└──────────────────────────┘
```

### Chi tiết các thành phần

| Module | Chức năng |
|--------|----------|
| **src/detector.py** | Tải model YOLOv9, tiền/hậu xử lý, inference, NMS |
| **src/roi.py** | Đọc config ROI từ JSON, kiểm tra điểm có trong polygon |
| **src/visualize.py** | Vẽ ROI overlay, bbox, nhãn, cảnh báo lên frame |
| **src/pipeline.py** | Kết nối detector + ROI + visualizer thành pipeline hoàn chỉnh |

## 📊 Cấu Hình

### Model Config

- **Model**: YOLOv9-Small
- **Input Size**: 640x640
- **Confidence Threshold**: 0.25 (mặc định)
- **NMS Threshold**: 0.45 (mặc định)
- **Weight**: `weights/best_small.pt` (20.3 MB)

### Classes Config (`configs/classes.yaml`)

```yaml
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
```

### ROI Config (`configs/roi.json`)

```json
{
  "image_size": { "w": 1280, "h": 720 },
  "polygon": [[900, 150], [1270, 250], [1270, 710], [800, 710], [760, 520]],
  "check_point": "bbox_bottom_center"
}
```

- **polygon**: 5 đỉnh định nghĩa vùng điểm mù (bên phải)
- **check_point**: Điểm dùng để kiểm tra (dưới giữa bbox thích hợp cho xe cộ)

## 📈 Hiệu Năng Model

Model được huấn luyện trên **Google Colab Pro (NVIDIA H100)** trong 100 epochs:

| Metric | YOLOv9-Small | YOLOv9-Tiny |
|--------|--------------|------------|
| **mAP@0.5** | 0.70 | 0.69 |
| **Model Size** | 20.3 MB | ~15 MB |
| **Inference Speed** | ~30 FPS | ~50 FPS |
| **Status** | ✅ Chính | (Thay thế) |

## 🎯 Kết Quả Hiển Thị

Trên video output:

- **Bbox xanh**: Đối tượng phát hiện nhưng ở ngoài vùng điểm mù (an toàn)
- **Bbox đỏ**: Đối tượng nằm trong vùng điểm mù (⚠️ BLIND SPOT)
- **ROI polygon**: Đường viền xác định vùng điểm mù
- **FPS counter**: Hiệu suất xử lý realtime

## 📝 Báo Cáo Chi Tiết

Các báo cáo chi tiết nằm trong thư mục `report/`:

- **dataset_report.md**: Chuẩn bị dataset, xử lý dữ liệu
- **training_report.md**: Quá trình huấn luyện, kết quả, so sánh model
- **inference_report.md**: Cấu trúc pipeline, hướng tối ưu hóa

## 🔮 Hướng Phát Triển

Các cải tiến tiềm năng:

- ✅ Hỗ trợ GPU acceleration (CUDA/TensorRT)
- ✅ Sử dụng half precision (FP16) để tăng tốc
- ✅ Video writer để lưu kết quả
- ✅ Logging và benchmark FPS
- ✅ Web interface / REST API
- ✅ Đa ROI (các vùng điểm mù khác nhau)

## 📄 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

## 👥 Tác Giả

Project được phát triển bởi Taitu và team VTD0102.

## 📧 Liên Hệ & Hỗ Trợ

Nếu có câu hỏi hoặc phát hiện issue:
- Tạo GitHub Issue
- Hoặc liên hệ trực tiếp qua email/Slack

---

**Happy Detection! 🎯**
