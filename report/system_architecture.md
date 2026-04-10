# Kiến trúc hệ thống — Truck Blind Spot Detection

> Tài liệu phân tích toàn bộ vùng ROI, luồng xử lý, chức năng từng thành phần  
> trong hệ thống cảnh báo điểm mù xe tải sử dụng YOLOv9.

---

## 1. Hệ tọa độ ảnh

Tất cả tọa độ trong `configs/roi.json` dùng hệ tọa độ **pixel tuyệt đối** trên ảnh 1280×720.

```
(0,0) ─────────────────────────────── (1280,0)
  │                                        │
  │   x tăng →                             │
  │   y tăng ↓                             │
  │                                        │
(0,720) ────────────────────────────(1280,720)
```

**Đọc một điểm `[x, y]`:**

| Ký hiệu | Ý nghĩa |
|---|---|
| `x` | Khoảng cách từ **cạnh trái** sang phải (0 = cạnh trái, 1280 = cạnh phải) |
| `y` | Khoảng cách từ **cạnh trên** xuống dưới (0 = cạnh trên, 720 = cạnh dưới) |

**Ví dụ cụ thể:**

| Điểm | Vị trí trên ảnh |
|---|---|
| `[0, 360]` | Cạnh trái, chính giữa chiều cao (360/720 = 50%) |
| `[640, 0]` | Chính giữa chiều ngang, cạnh trên |
| `[1279, 719]` | Góc dưới-phải (gần cạnh phải, gần cạnh dưới) |
| `[220, 610]` | Gần cạnh dưới (610/720 ≈ 85%), lệch trái nhẹ |
| `[430, 430]` | Vùng trung tâm (y=430 ≈ 60% chiều cao) |

---

## 2. Phân tích chi tiết các vùng ROI

### 2.1 Camera trước (`front_camera`)

**Bối cảnh:** Camera gắn phía trước cabin xe tải, nhìn về phía trước và hai bên. Ảnh 1280×720 hiển thị mặt đường trước mặt, với phần dưới là đầu xe tải.

```
Sơ đồ vùng ROI — front_camera (1280×720)
┌──────────────────────────────────────────────────────────────────┐ y=0
│                         [khu vực xa, an toàn]                   │
│  [220,330]                                     [1060,330]        │
│    ╱───────────────────────────────────────────────╲            │ y≈330
│   ╱  left_blind_spot         forward_danger_zone    ╲           │
│  [0,360]  [420,520]──[430,430]────────[850,430]──[980,520]       │ y≈360–520
│  │  (MEDIUM)  ╲    ╱                              ╱  (WARNING)  │
│  │             ╲  ╱  [300,520]        [980,520]  ╱  right_blind │
│  │         [300,719]──[380,719]     [900,719]──[980,719]        │ y≈520
│  └──────────────────────────────────────────────────────────────┤
│  [0,719]   [220,610]═══════════════════════[1060,610]  [1279,719]│ y≈610–719
│            ╚══════ near_cabin_zone (HIGH) ═══════════╝          │
└──────────────────────────────────────────────────────────────────┘ y=720
x=0                      x=640                               x=1280
```

---

#### Zone 1: `left_blind_spot` — MEDIUM risk

```json
"polygon": [[0,360],[220,330],[420,520],[300,719],[0,719]]
```

**Hình dạng và vị trí:**
```
(0,360) ──→── (220,330)
   │               ↘
   │           (420,520)
   │               ↓
(0,719) ←──── (300,719)
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[0, 360]` | Cạnh trái, giữa chiều cao | Điểm bắt đầu vùng nhìn sang trái |
| `[220, 330]` | Gần trung tâm trái, hơi cao | Góc trong của vùng điểm mù trái (nội cảnh) |
| `[420, 520]` | Phần tư trái dưới | Mở rộng vào giữa ảnh ở vùng gần |
| `[300, 719]` | Đáy ảnh, lệch trái | Chân vùng điểm mù sát đầu xe |
| `[0, 719]` | Góc dưới trái | Đóng vùng về cạnh trái |

**Ý nghĩa thực tế:** Vùng bên trái xe tải nhìn từ camera trước — nơi xe đạp, xe máy, người đi bộ có thể bị khuất sau gương chiếu hậu hoặc góc chết của cabin. Mức MEDIUM vì camera trước vẫn nhìn được một phần.

---

#### Zone 2: `forward_danger_zone` — WARNING risk

```json
"polygon": [[300,520],[430,430],[850,430],[980,520],[900,719],[380,719]]
```

**Hình dạng và vị trí:**
```
        (430,430) ──────── (850,430)      ← y=430, ngang tầm trung
       ╱                          ╲
(300,520)                      (980,520)  ← y=520
      ╲                          ╱
   (380,719) ──────────── (900,719)       ← đáy ảnh, vùng trước đầu xe
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[430, 430]` | Trung tâm trái, 60% chiều cao | Góc trái trên vùng nguy hiểm phía trước |
| `[850, 430]` | Trung tâm phải, 60% chiều cao | Góc phải trên vùng nguy hiểm phía trước |
| `[980, 520]` | Phần tư phải dưới | Cạnh phải vùng gần |
| `[900, 719]` | Đáy phải | Góc phải sát đầu xe |
| `[380, 719]` | Đáy trái | Góc trái sát đầu xe |
| `[300, 520]` | Phần tư trái dưới | Cạnh trái vùng gần |

**Ý nghĩa thực tế:** Vùng thẳng phía trước xe, khu vực ngay trước đầu xe tải — nơi người hoặc vật thể có thể bị cán nếu xe tiến lên. Mức WARNING (thấp hơn HIGH) vì camera trước nhìn rõ nhất vùng này, chủ yếu cảnh báo khi có vật thể quá gần.

---

#### Zone 3: `right_blind_spot` — MEDIUM risk

```json
"polygon": [[1060,330],[1279,360],[1279,719],[980,719],[860,520]]
```

**Hình dạng và vị trí:**
```
(1060,330) ─→─ (1279,360)
                    │
               (1279,719) ← cạnh phải đáy
                   ╱
            (980,719)
           ╱
      (860,520)
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[1060, 330]` | Gần cạnh phải, 46% chiều cao | Điểm bắt đầu vùng mù phải, ngang tầm gương |
| `[1279, 360]` | Cạnh phải, giữa chiều cao | Điểm trên cùng cạnh phải |
| `[1279, 719]` | Góc dưới phải | Đóng vùng về cạnh phải |
| `[980, 719]` | Đáy ảnh, 77% chiều ngang | Chân vùng điểm mù phải |
| `[860, 520]` | Phần tư phải, vùng gần | Điểm vào trong nhất của vùng mù phải |

**Ý nghĩa thực tế:** Đối xứng với `left_blind_spot`, vùng bên phải xe tải nhìn từ camera trước. Điểm mù phải thường nguy hiểm hơn ở Việt Nam (làn xe máy chạy phía phải).

---

#### Zone 4: `near_cabin_zone` — HIGH risk

```json
"polygon": [[220,610],[1060,610],[1279,719],[0,719]]
```

**Hình dạng và vị trí:**
```
(220,610) ─────────────────────── (1060,610)   ← y=610, ~85% chiều cao
╱                                              ╲
(0,719) ──────────────────────────────── (1279,719)  ← đáy toàn bộ
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[220, 610]` | Gần đáy, lệch trái | Góc trái dải nguy hiểm cao |
| `[1060, 610]` | Gần đáy, lệch phải | Góc phải dải nguy hiểm cao |
| `[1279, 719]` | Góc dưới phải | Rộng ra toàn bộ đáy ảnh phải |
| `[0, 719]` | Góc dưới trái | Rộng ra toàn bộ đáy ảnh trái |

**Ý nghĩa thực tế:** Dải sát đầu xe nhất — vật thể ở đây đang **ngay trước bánh xe hoặc gầm xe**. Đây là vùng nguy hiểm nhất (HIGH) vì khoảng cách phản ứng gần như bằng không. Zone này nằm ở cùng chiều ngang với các zone khác nhưng ở sát đáy → khi `get_zone()` gặp overlap, HIGH sẽ được ưu tiên.

---

### 2.2 Camera sau (`rear_camera`)

**Bối cảnh:** Camera gắn phía sau thùng xe tải, nhìn ra phía sau. Phần dưới ảnh là đuôi xe, phần trên là không gian phía sau.

```
Sơ đồ vùng ROI — rear_camera (1280×720)
┌──────────────────────────────────────────────────────────────────┐ y=0
│                    [khu vực xa phía sau]                         │
│ [0,180]──[240,180]                    [1040,180]──[1279,180]     │ y=180
│ │  rear_left_blind     [320,300]──[960,300]    rear_right_blind  │
│ │  (MEDIUM)           ╱              ╲         (MEDIUM)          │
│ [360,420]            ╱  rear_danger   ╲        [920,420]        │ y≈300–420
│  ╲                  ╱   zone (HIGH)    ╲               ╱        │
│   [260,719]        [200,719]──────────[1080,719]  [1020,719]    │ y=719
└──────────────────────────────────────────────────────────────────┘
x=0                      x=640                               x=1280
```

---

#### Zone 5: `rear_left_blind_spot` — MEDIUM risk

```json
"polygon": [[0,180],[240,180],[360,420],[260,719],[0,719]]
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[0, 180]` | Cạnh trái, 25% chiều cao | Điểm trên cùng bên trái |
| `[240, 180]` | Gần trái, 25% chiều cao | Giới hạn trong vùng mù trái |
| `[360, 420]` | Phần tư trái, 58% chiều cao | Vùng mù mở rộng xuống dưới |
| `[260, 719]` | Đáy, lệch trái | Chân vùng mù trái |
| `[0, 719]` | Góc dưới trái | Đóng về cạnh trái |

**Ý nghĩa thực tế:** Điểm mù trái khi lùi xe — xe máy/người đi bộ đến từ phía sau bên trái không nhìn thấy qua gương.

---

#### Zone 6: `rear_danger_zone` — HIGH risk

```json
"polygon": [[320,300],[960,300],[1080,719],[200,719]]
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[320, 300]` | Phần tư trái, 42% chiều cao | Góc trên trái vùng nguy hiểm trung tâm |
| `[960, 300]` | Phần tư phải, 42% chiều cao | Góc trên phải vùng nguy hiểm trung tâm |
| `[1080, 719]` | Đáy, lệch phải | Mở rộng xuống đáy phải |
| `[200, 719]` | Đáy, lệch trái | Mở rộng xuống đáy trái |

**Ý nghĩa thực tế:** Vùng thẳng phía sau đuôi xe — nguy hiểm nhất khi lùi xe. Hình thang mở rộng xuống đáy vì vật thể gần đuôi xe chiếm nhiều pixel ở dưới (phối cảnh camera).

---

#### Zone 7: `rear_right_blind_spot` — MEDIUM risk

```json
"polygon": [[1040,180],[1279,180],[1279,719],[1020,719],[920,420]]
```

| Điểm | Vị trí | Ý nghĩa |
|---|---|---|
| `[1040, 180]` | Gần phải, 25% chiều cao | Điểm trên vùng mù phải |
| `[1279, 180]` | Cạnh phải, 25% chiều cao | Cạnh trên phải |
| `[1279, 719]` | Góc dưới phải | Cạnh phải đáy |
| `[1020, 719]` | Đáy, gần phải | Chân vùng mù phải |
| `[920, 420]` | Phần tư phải, 58% chiều cao | Giới hạn trong vùng mù phải |

**Ý nghĩa thực tế:** Đối xứng với `rear_left_blind_spot` — góc chết bên phải khi lùi.

---

### 2.3 Xử lý vùng chồng lấn (Overlap)

Các zone có thể chồng lên nhau (ví dụ `near_cabin_zone` nằm dưới cả `left_blind_spot` và `right_blind_spot`). Hàm `get_zone()` trong [src/roi.py](../src/roi.py) xử lý overlap bằng cách **ưu tiên risk_level cao nhất:**

```python
priority = {"HIGH": 3, "MEDIUM": 2, "WARNING": 1, "LOW": 0}
matched_zones.sort(key=lambda z: priority.get(z.risk_level.upper(), 0), reverse=True)
return matched_zones[0]  # trả về zone nguy hiểm nhất
```

Ví dụ: một người đứng ở góc dưới-trái ảnh front_camera sẽ khớp cả `left_blind_spot` (MEDIUM) và `near_cabin_zone` (HIGH) → hệ thống báo **HIGH**.

---

## 3. Kiến trúc tổng thể hệ thống

```
configs/
├── roi.json          ← Định nghĩa vùng nguy hiểm (đa giác, màu, risk_level)
└── classes.yaml      ← Tên 6 class: person, bike, motor, car, truck, bus

weights/
└── best_small.pt     ← Weights YOLOv9-S đã train

src/
├── detector.py       ← Load model, chạy inference, trả về List[Detection]
├── roi.py            ← Load ROI config, kiểm tra điểm tham chiếu nằm trong zone nào
├── visualize.py      ← Vẽ zone, bbox, label, banner cảnh báo lên frame
└── pipeline.py       ← Điều phối toàn bộ: detector → roi → visualize

app.py                ← Entry point demo: đọc video/webcam, hiển thị realtime
```

---

## 4. Chi tiết từng thành phần

### 4.1 `detector.py` — Phát hiện vật thể

**Class `Detection` (dataclass):**
```python
@dataclass
class Detection:
    bbox: (x1, y1, x2, y2)   # tọa độ pixel bounding box
    confidence: float          # độ tin cậy (0.0–1.0)
    class_id: int              # 0=person, 1=bike, 2=motor, 3=car, 4=truck, 5=bus
    class_name: str            # tên class
    in_roi: bool               # có nằm trong zone nguy hiểm không
    anchor_point: (x, y)       # điểm chân bbox dùng để kiểm tra zone
    zone_name: str | None      # tên zone đang ở (nếu có)
    risk_level: str | None     # mức nguy hiểm: HIGH / MEDIUM / WARNING
```

**Class `YOLOv9Detector`:**

| Bước | Hàm | Mô tả |
|---|---|---|
| 1 | `__init__` | Load model `.pt`, warmup GPU |
| 2 | `predict(frame)` | Tiền xử lý ảnh (letterbox → tensor) |
| 3 | — | Chạy inference `torch.inference_mode()` |
| 4 | `_unwrap_predictions` | Giải nén output YOLOv9 (có thể là list/tuple lồng nhau) |
| 5 | — | NMS lọc bbox trùng |
| 6 | — | Scale bbox về kích thước ảnh gốc |
| 7 | — | Trả về `List[Detection]` |

---

### 4.2 `roi.py` — Kiểm tra vùng nguy hiểm

**Class `ROIZone`:**
- Lưu tên, polygon (np.array), màu, risk_level
- `contains_point(x, y)` → dùng `cv2.pointPolygonTest` để kiểm tra điểm có trong đa giác không

**Class `MultiPolygonROI`:**

| Hàm | Mô tả |
|---|---|
| `__init__(path, profile)` | Load JSON, chọn profile (`front_camera` / `rear_camera`), tạo danh sách `ROIZone` |
| `get_reference_point(bbox)` | Tính điểm tham chiếu từ bbox (mặc định: đáy giữa = chân vật thể) |
| `get_zone(point)` | Kiểm tra điểm khớp zone nào, trả về zone có risk_level cao nhất |
| `contains_bbox(bbox)` | Shorthand: `get_zone(get_reference_point(bbox)) is not None` |

---

### 4.3 `visualize.py` — Vẽ kết quả

**Class `BlindSpotVisualizer`:**

| Hàm | Mô tả |
|---|---|
| `draw(frame, detections, roi_zones)` | Hàm chính: gọi `_draw_roi_zones` → vẽ bbox/label → vẽ banner |
| `_draw_roi_zones(frame, zones)` | **3 bước:** (1) fillPoly vào overlay, (2) addWeighted blend nhẹ, (3) vẽ viền+label sau blend để rõ nét |
| `_resolve_detection_color(detection, zones)` | Xanh lá nếu an toàn; màu của zone nếu trong ROI |
| `_draw_banner(frame, text)` | Banner đỏ góc trên-trái: "BLIND SPOT ALERT: N" |
| `_draw_label(frame, text, pos, color)` | Nhãn có nền màu, chữ trắng |

**Quy tắc màu sắc bbox:**
- Object **ngoài ROI** → xanh lá `(0, 200, 0)`
- Object **trong zone** → màu riêng của zone đó (lấy từ `roi.json`)

---

### 4.4 `pipeline.py` — Điều phối trung tâm

**Class `BlindSpotPipeline`:**

```
BlindSpotPipeline.__init__
    ├── YOLOv9Detector(weights, classes, device, ...)
    ├── MultiPolygonROI(roi_config, profile)
    └── BlindSpotVisualizer()
```

| Hàm | Mô tả |
|---|---|
| `process_frame(frame)` | Xử lý 1 frame: detect → gán zone → visualize → trả về (annotated_frame, detections) |
| `process_image(path)` | Đọc ảnh → `process_frame` → lưu output nếu có |
| `run_video(source)` | Vòng lặp video: đọc frame → `process_frame` → hiển thị/lưu |

---

## 5. Luồng xử lý đầy đủ

### 5.1 Luồng mỗi frame (realtime)

```
─────────────────────────────────────────────────────────────────
                     MỘT FRAME TỪ CAMERA / VIDEO
─────────────────────────────────────────────────────────────────
                              │
                              ▼
              ┌───────────────────────────┐
              │     YOLOv9Detector        │
              │  letterbox → tensor       │
              │  inference (GPU)          │
              │  NMS → scale bbox         │
              └─────────────┬─────────────┘
                            │ List[Detection]
                            │ (bbox, confidence, class_id, class_name)
                            ▼
              ┌───────────────────────────┐
              │      MultiPolygonROI      │
              │                           │
              │  get_reference_point()    │  → điểm chân bbox (bottom_center)
              │  get_zone()               │  → khớp zone nào? ưu tiên HIGH
              │                           │
              │  detection.in_roi  ←──────┤  True / False
              │  detection.zone_name ←────┤  "near_cabin_zone" / None / ...
              │  detection.risk_level ←───┤  "HIGH" / "MEDIUM" / None / ...
              └─────────────┬─────────────┘
                            │ List[Detection] (đã enriched)
                            ▼
              ┌───────────────────────────┐
              │    BlindSpotVisualizer    │
              │                           │
              │  fillPoly (alpha 0.18)    │  → tô nhẹ vùng ROI
              │  addWeighted blend        │
              │  polylines + zone labels  │  → viền và tên zone rõ nét
              │                           │
              │  Mỗi detection:           │
              │    rectangle (màu zone)   │  → bbox màu theo zone
              │    label text             │  → "near_cabin_zone [HIGH] | person 0.87"
              │    circle (anchor point)  │  → chấm tròn tại chân bbox
              │                           │
              │  Nếu có in_roi > 0:       │
              │    banner đỏ góc trên     │  → "BLIND SPOT ALERT: 2"
              └─────────────┬─────────────┘
                            │ annotated_frame (np.ndarray)
                            ▼
                    Hiển thị / Lưu file
─────────────────────────────────────────────────────────────────
```

### 5.2 Luồng khởi động `app.py`

```
python3 app.py [--roi-profile front_camera|rear_camera] [--source video.mp4]
      │
      ▼
parse_args()                  ← đọc --source, --weights, --roi-profile, ...
      │
      ▼
BlindSpotPipeline(            ← khởi tạo toàn bộ hệ thống
    weights, roi, profile,
    classes, device, conf, iou
)
      │
      ├── Load YOLOv9 model (GPU warmup)
      ├── Load ROI profile từ configs/roi.json
      └── Khởi tạo Visualizer
      │
      ▼
open_capture(source)          ← mở VideoCapture (file hoặc webcam)
      │
      ▼
┌─────────────────────────────────────┐
│         VÒNG LẶP CHÍNH              │
│                                     │
│  cap.read() → frame                 │
│       │                             │
│       ▼                             │
│  pipeline.process_frame(frame)      │  ← detector + roi + visualize
│       │                             │
│       ▼                             │
│  draw_overlay(fps, status)          │  ← vẽ FPS + RUNNING/PAUSED
│       │                             │
│       ▼                             │
│  cv2.imshow(window, frame)          │
│       │                             │
│  Phím tắt:                          │
│    [p] → pause/resume               │
│    [r] → restart video              │
│    [q] → thoát                      │
└─────────────────────────────────────┘
      │
      ▼
cap.release() + writer.release()
cv2.destroyAllWindows()
```

---

## 6. Lệnh chạy nhanh

```bash
# Camera trước, video demo mặc định
python3 app.py

# Camera sau
python3 app.py --roi-profile rear_camera

# Video khác + lưu output
python3 app.py --source assets/videos/demo.mp4 \
               --roi-profile front_camera \
               --output output/result.mp4 \
               --loop

# Webcam realtime (camera trước)
python3 app.py --source 0 --roi-profile front_camera

# Từ pipeline.py trực tiếp (ảnh đơn)
python3 src/pipeline.py --source path/to/image.jpg \
                        --roi-profile front_camera \
                        --show

# Chỉnh confidence threshold
python3 app.py --conf-thres 0.35 --iou-thres 0.45
```

---

## 7. Tóm tắt các zone và mức cảnh báo

| Zone | Profile | Risk | Màu (BGR) | Vị trí thực tế |
|---|---|---|---|---|
| `left_blind_spot` | front | MEDIUM | `(255,140,0)` cam | Bên trái cabin, góc chết trái |
| `forward_danger_zone` | front | WARNING | `(0,165,255)` vàng | Thẳng trước đầu xe |
| `right_blind_spot` | front | MEDIUM | `(255,140,0)` cam | Bên phải cabin, góc chết phải |
| `near_cabin_zone` | front | **HIGH** | `(0,0,255)` đỏ | Sát đầu xe — khoảng cách tối thiểu |
| `rear_left_blind_spot` | rear | MEDIUM | `(128,0,255)` tím | Góc chết trái khi lùi |
| `rear_danger_zone` | rear | **HIGH** | `(255,0,0)` xanh | Thẳng phía sau đuôi xe |
| `rear_right_blind_spot` | rear | MEDIUM | `(128,0,255)` tím | Góc chết phải khi lùi |
