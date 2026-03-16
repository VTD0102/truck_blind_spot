# Blind Spot Inference Report

## 1. Mục tiêu

Xây dựng module suy luận (inference) cho bài toán phát hiện vật thể trong vùng điểm mù xe tải bằng **YOLOv9**, sử dụng model đã huấn luyện sẵn với weight:

- `weights/best_small.pt`

Hệ thống có nhiệm vụ:

- phát hiện các đối tượng trong frame video
- kiểm tra đối tượng có nằm trong vùng ROI điểm mù hay không
- hiển thị cảnh báo `BLIND SPOT`
- hỗ trợ demo realtime bằng video có sẵn

---

## 2. Cấu trúc project phục vụ inference

```text
project/
├── app.py
├── requirements.txt
├── configs/
│   ├── classes.yaml
│   └── roi.json
├── report/
│   ├── dataset_report.md
│   ├── training_report.md
│   └── inference_report.md
├── src/
│   ├── detector.py
│   ├── roi.py
│   ├── visualize.py
│   └── pipeline.py
├── videos/
│   └── test_video.mp4
├── weights/
│   └── best_small.pt
└── yolov9/
```

Ý nghĩa các module:

- `src/detector.py`: load YOLOv9 model, tiền xử lý ảnh, suy luận, hậu xử lý kết quả
- `src/roi.py`: đọc polygon ROI từ file JSON và kiểm tra điểm của bounding box bằng `cv2.pointPolygonTest`
- `src/visualize.py`: vẽ ROI, bounding box, nhãn class, confidence, cảnh báo blind spot
- `src/pipeline.py`: kết nối detector + ROI + visualize thành luồng xử lý hoàn chỉnh cho từng frame
- `app.py`: ứng dụng demo video bằng OpenCV

---

## 3. Thiết kế pipeline inference

Luồng xử lý của hệ thống:

1. Đọc từng frame từ video bằng OpenCV.
2. Gửi frame vào `BlindSpotPipeline`.
3. `YOLOv9Detector` thực hiện:
   - resize ảnh theo input size
   - chạy model YOLOv9
   - non-max suppression
   - trả về danh sách đối tượng gồm `bbox`, `confidence`, `class_id`, `class_name`
4. `PolygonROI` xác định điểm kiểm tra của bounding box.
5. Kiểm tra điểm này có nằm trong polygon ROI hay không.
6. `BlindSpotVisualizer` vẽ:
   - ROI polygon
   - bounding box
   - class label
   - confidence
   - cảnh báo `BLIND SPOT` nếu đối tượng nằm trong ROI
7. `app.py` hiển thị frame kết quả và FPS theo thời gian thực.

---

## 4. Cấu hình sử dụng

### Model

- **Model:** YOLOv9
- **Weight:** `weights/best_small.pt`

### ROI

ROI được lưu trong file `configs/roi.json` dưới dạng polygon.

Ví dụ:

```json
{
  "image_size": { "w": 1280, "h": 720 },
  "polygon": [[900, 150], [1270, 250], [1270, 710], [800, 710], [760, 520]],
  "check_point": "bbox_bottom_center"
}
```

Trong đó:

- `polygon`: danh sách các đỉnh của vùng điểm mù
- `check_point`: điểm dùng để kiểm tra đối tượng có thuộc ROI hay không
- giá trị hiện tại là `bbox_bottom_center`, phù hợp cho bài toán xe cộ vì phần đáy bbox gần với vị trí tiếp xúc mặt đường hơn

### Classes

Ví dụ `configs/classes.yaml`:

```yaml
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
```

---

## 5. Chức năng demo video

File `app.py` dùng để chạy demo với video có sẵn:

- đọc video từ `videos/test_video.mp4`
- xử lý lần lượt từng frame
- hiển thị realtime bằng OpenCV
- nhấn `q` để thoát
- nhấn `p` để pause / resume
- chương trình tự dừng khi video kết thúc

Thông tin hiển thị trên frame:

- bounding box
- class label
- confidence score
- vùng ROI
- cảnh báo `BLIND SPOT`
- FPS

---

## 6. Tối ưu cho realtime

Một số điểm tối ưu đã áp dụng trong phần demo:

- dùng pipeline theo từng frame, không tạo lại model mỗi lần
- model được load một lần khi khởi tạo chương trình
- xử lý trực tiếp trên video stream bằng OpenCV
- hiển thị FPS để theo dõi hiệu năng
- sử dụng `best_small.pt` để cân bằng giữa tốc độ và độ chính xác

Trong tương lai có thể tối ưu thêm:

- chạy GPU với CUDA
- dùng half precision (FP16)
- giảm `imgsz` nếu cần tăng tốc
- thêm video writer để lưu kết quả
- bổ sung logging và benchmark FPS trung bình

---

## 7. Kết quả đầu ra mong muốn

Sau khi chạy demo, hệ thống sẽ:

- phát hiện đúng các đối tượng đã train
- đánh dấu các đối tượng nằm trong vùng điểm mù
- hiển thị cảnh báo trực quan trên video
- hỗ trợ minh họa trực tiếp cho đồ án AI

Ví dụ trạng thái hiển thị:

- đối tượng ngoài ROI: bbox màu xanh
- đối tượng trong ROI: bbox màu đỏ và gắn nhãn `BLIND SPOT`

---

## 8. Cách chạy

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Chạy demo

```bash
python3 app.py
```

---

## 9. Đánh giá

Phần inference đã hoàn thiện theo hướng tách module rõ ràng, dễ bảo trì và dễ mở rộng:

- có thể thay ROI mà không sửa logic detector
- có thể thay model weight mới mà không ảnh hưởng app
- có thể tái sử dụng `pipeline.py` cho ảnh, video hoặc camera
- phù hợp để phát triển tiếp thành hệ thống giám sát thời gian thực

Đây là phần quan trọng để kết nối giữa **kết quả huấn luyện model** và **ứng dụng demo thực tế** của bài toán phát hiện điểm mù.
