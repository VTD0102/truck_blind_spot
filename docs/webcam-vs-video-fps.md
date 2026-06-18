# Tại sao Webcam + CoreML cho FPS cao hơn Video File + CoreML?

## Kết quả đo thực tế

| Nguồn đầu vào | FPS hiển thị | FPS thực tế người dùng thấy |
|---|---|---|
| File `.mp4` (demo4.mp4) | 53–60 FPS | 53–60 FPS |
| Webcam | 70–80 FPS | 30 FPS (giới hạn phần cứng) |

---

## 3 nguyên nhân

### 1. CPU contention — H.264 decode (nguyên nhân chính)

Mỗi lần `cap.read()` đọc một frame từ file `.mp4`, CPU phải **decode H.264** để tái tạo ảnh gốc từ dữ liệu nén (~3–5ms/frame). Trong khi đó, CoreML vẫn dùng CPU cho các bước pre/post-processing (letterbox, NMS, visualization). Hai tác vụ tranh chấp CPU → tổng thời gian xử lý mỗi frame tăng lên.

Webcam truyền **MJPG** (nén từng frame độc lập, không có delta giữa các frame) → decode nhanh hơn nhiều, ít tranh chấp CPU hơn.

### 2. Frame pacing — webcam tự tạo "breathing room"

Webcam phần cứng giới hạn ở 30 FPS → `cap.read()` tự nhiên **block ~33ms** chờ frame tiếp theo từ camera. Trong 33ms đó, Neural Engine được "nghỉ", cache warm → inference frame kế nhanh hơn.

File video thì `cap.read()` không chờ — app đọc hết tốc độ liên tục, Neural Engine không có thời gian reset giữa các frame.

### 3. FPS đo được là throughput, không phải realtime

```python
t0 = time.perf_counter()
annotated = pipeline.process_frame(frame)   # chỉ đo phần này
elapsed = time.perf_counter() - t0
fps = 1.0 / elapsed                         # bỏ qua thời gian cap.read()
```

Khi webcam block 33ms chờ frame mới, `elapsed` của `process_frame()` nhỏ → FPS số học cao. Con số 70–80 FPS là **khả năng tối đa của pipeline**, không phải FPS thực người dùng thấy vì camera chỉ cấp 30 frame/giây.

---

## Kết luận

Con số **53–60 FPS với video file phản ánh thực tế hơn** vì app đọc frame liên tục không nghỉ, không có thời gian chờ phần cứng. Đây là con số nên dùng khi đánh giá hiệu năng CoreML backend.
