# Dataset Report — Truck Blind Spot Detection

> Tài liệu mô tả cấu trúc dữ liệu, quy trình chuẩn hóa và lịch sử phát triển dataset  
> cho dự án **Hệ thống cảnh báo điểm mù xe tải** sử dụng YOLOv9.

---

## 1. Cấu trúc thư mục `yolov9/data/`

```
yolov9/data/
├── blindspot.yaml              ← cấu hình dataset chính của dự án
├── coco.yaml                   ← cấu hình COCO (tham khảo, không dùng train)
├── hyps/
│   ├── hyp.blindspot.yaml      ← hyperparameters train dành riêng cho dự án
│   └── hyp.scratch-high.yaml   ← hyperparameters gốc của YOLOv9 (tham khảo)
├── images/                     ← tiện ích xử lý ảnh của framework YOLOv9
└── datasets/
    ├── bdd100k_ultralytics.yaml ← cấu hình dataset gốc BDD100K (lưu tham khảo)
    ├── train/
    │   ├── images/             ← ảnh training (7.000 ảnh)
    │   ├── labels/             ← nhãn YOLO hiện tại, đã chuẩn hóa 6 class
    │   └── labels_origin/      ← bản sao gốc trước khi chuẩn hóa (backup)
    ├── valid/
    │   ├── images/             ← ảnh validation (1.000 ảnh)
    │   ├── labels/             ← nhãn YOLO hiện tại, đã chuẩn hóa 6 class
    │   └── labels_origin/      ← bản sao gốc trước khi chuẩn hóa (backup)
    └── test/
        ├── images/             ← ảnh test (2.000 ảnh)
        ├── labels/             ← nhãn YOLO (hiện rỗng, chưa annotate)
        └── labels_origin/      ← bản sao gốc (backup)
```

---

### 1.1 Thư mục `datasets/`

#### `train/`, `valid/`, `test/`

Mỗi split chia thành hai thư mục con:

| Thư mục | Mô tả |
|---|---|
| `images/` | Ảnh đầu vào định dạng `.jpg` hoặc `.png`, là dữ liệu thô để model học/đánh giá |
| `labels/` | File `.txt` tương ứng mỗi ảnh, chứa nhãn YOLO đã chuẩn hóa theo 6 class của dự án |
| `labels_origin/` | Bản sao **nguyên bản** của `labels/` trước khi lọc và remap — dùng để tra cứu hoặc phục hồi nếu có sai sót |

**Định dạng file label (YOLO):** Mỗi dòng trong `.txt` là một object:
```
<class_id> <center_x> <center_y> <width> <height>
```
Tất cả tọa độ được chuẩn hóa về khoảng `[0, 1]` so với kích thước ảnh.

**Phân bố dữ liệu:**

| Split | Số ảnh | Tỉ lệ | Mục đích |
|---|---|---|---|
| train | 7.000 | 70% | Huấn luyện model |
| valid | 1.000 | 10% | Đánh giá trong quá trình train |
| test  | 2.000 | 20% | Kiểm tra độc lập sau khi train xong |

---

#### `bdd100k_ultralytics.yaml`

File cấu hình dataset **BDD100K gốc** (10 class) — được lưu lại để tham khảo nguồn gốc dữ liệu, **không dùng** để train dự án. Nội dung 10 class gốc:

```
0: bike  |  1: bus  |  2: car  |  3: motor  |  4: person
5: rider  |  6: traffic light  |  7: traffic sign  |  8: train  |  9: truck
```

---

### 1.2 File `blindspot.yaml`

File cấu hình dataset **chính thức** của dự án. Được trỏ vào khi chạy train, val, detect.

```yaml
path: data/datasets

train: train/images
val:   valid/images
test:  test/images

nc: 6

names:
  0: person
  1: bike
  2: motor
  3: car
  4: truck
  5: bus
```

**6 class được giữ lại** là những đối tượng thực sự xuất hiện trong vùng điểm mù xe tải và có nguy cơ gây tai nạn: người đi bộ, xe đạp, xe máy, ô tô con, xe tải, xe buýt.

---

### 1.3 File `hyps/hyp.blindspot.yaml`

Hyperparameters tùy chỉnh cho bài toán phát hiện điểm mù. Các điểm khác biệt so với cấu hình gốc:

| Nhóm | Thông số quan trọng | Giá trị | Lý do |
|---|---|---|---|
| Màu sắc | `hsv_v` | 0.5 | Giả lập ban ngày/đêm, kính ướt |
| Màu sắc | `hsv_s` | 0.7 | Nắng chói vs mưa/sương mù |
| Màu sắc | `hsv_h` | 0.02 | Đèn vàng/xanh đô thị ban đêm |
| Hình học | `scale` | 0.5 | Xe/người ở gần lẫn xa |
| Hình học | `translate` | 0.2 | Object ở rìa khung hình |
| Hình học | `fliplr` | 0.0 | Giữ đúng chiều lưu thông |
| Augmentation | `mosaic` | 1.0 | Dataset dồi dào, bật toàn bộ |
| Augmentation | `mixup` | 0.1 | Tăng robustness khi objects chồng nhau |

---

## 2. Thư mục `yolov9/standardize_dataset/`

Chứa toàn bộ script chuẩn hóa dataset — từ backup, lọc class, remap ID đến kiểm tra chất lượng.

```
standardize_dataset/
├── filter_labels.py        ← Bước 1: xóa các class không dùng
├── remap_labels.py         ← Bước 2: remap class ID theo blindspot.yaml
├── analyze_dataset.py      ← Kiểm tra: đếm phân bố class sau chuẩn hóa
├── check_dataset.py        ← Kiểm tra: xem ảnh + bbox trực quan
└── check_images/
    ├── check_duplicate_images.py   ← Kiểm tra ảnh trùng lặp
    └── check_error_images.py       ← Kiểm tra ảnh bị lỗi/corrupt
```

---

### 2.1 Mô tả từng file

#### `filter_labels.py`
**Mục đích:** Xóa tất cả các dòng label thuộc class không sử dụng trong dự án.

Các class bị xóa khỏi dataset gốc BDD100K:

| ID gốc | Class | Lý do loại bỏ |
|---|---|---|
| 5 | rider | Trùng lặp ngữ nghĩa với person + bike/motor |
| 6 | traffic light | Không phải đối tượng nguy hiểm trong vùng điểm mù |
| 7 | traffic sign | Không phải đối tượng nguy hiểm trong vùng điểm mù |
| 8 | train | Không xuất hiện trong môi trường điểm mù xe tải đô thị |

Sau bước này, mỗi file label chỉ còn các dòng thuộc 6 class: `bike, bus, car, motor, person, truck`.

---

#### `remap_labels.py`
**Mục đích:** Đổi class ID từ schema BDD100K gốc sang schema `blindspot.yaml` mới.

Bảng remap:

| ID gốc (BDD100K) | Class | → | ID mới (blindspot) | Class |
|---|---|---|---|---|
| 4 | person | → | 0 | person |
| 0 | bike | → | 1 | bike |
| 3 | motor | → | 2 | motor |
| 2 | car | → | 3 | car |
| 9 | truck | → | 4 | truck |
| 1 | bus | → | 5 | bus |

> **Quan trọng:** Phải chạy `filter_labels.py` **trước** rồi mới chạy `remap_labels.py`.  
> Nếu đảo ngược thứ tự, các ID cần xóa (5, 6, 7, 8) sẽ không khớp với bảng remap và gây lỗi.

---

#### `analyze_dataset.py`
**Mục đích:** Thống kê phân bố số lượng labels theo class trong từng split (train, val).

Output mẫu:
```
[train] 7000 ảnh | 120.000 labels tổng cộng:
  ID   Class          Labels   Tỉ lệ
  ----------------------------------------
  0    person          28.000   23.3%
  1    bike             8.000    6.7%
  2    motor           35.000   29.2%
  3    car             32.000   26.7%
  4    truck            9.000    7.5%
  5    bus              8.000    6.7%
  ----------------------------------------
  Tổng                120.000  100.0%
```

Dùng để xác nhận dataset đã được remap đúng và kiểm tra cân bằng dữ liệu trước khi train.

---

#### `check_dataset.py`
**Mục đích:** Hiển thị trực quan từng ảnh kèm bounding box và tên class để kiểm tra bằng mắt.

- Mỗi class được tô màu riêng biệt (xanh lá, cam, đỏ, tím, hồng...)
- Nhãn tên class hiển thị ngay trên bbox
- Hỗ trợ cả `train` và `val` (đổi biến `SPLIT` ở đầu file)
- Điều khiển: `SPACE/Enter` → ảnh tiếp, `q` → thoát

---

#### `check_images/check_duplicate_images.py`
**Mục đích:** Phát hiện ảnh trùng lặp nội dung bằng cách so sánh MD5 hash.

- Scan toàn bộ 3 splits: `train`, `valid`, `test`
- In ra các cặp ảnh trùng để xem xét xóa hoặc giữ lại

---

#### `check_images/check_error_images.py`
**Mục đích:** Phát hiện ảnh bị corrupt/lỗi không đọc được bằng OpenCV.

- Scan toàn bộ 3 splits: `train`, `valid`, `test`
- In ra danh sách ảnh lỗi để xóa hoặc thay thế

---

### 2.2 Thứ tự chạy để chuẩn hóa dataset hoàn chỉnh

```
Lưu ý tất cả các bước này đều đã được thực hiện và dataset hiện tại là dataset đã hoàn chỉnh.
Bước 0 (một lần duy nhất — đã thực hiện):
  cp -r train/labels  train/labels_origin
  cp -r valid/labels  valid/labels_origin
  cp -r test/labels   test/labels_origin

Bước 1 — Lọc class không dùng:
  python yolov9/standardize_dataset/filter_labels.py

Bước 2 — Remap class ID:
  python yolov9/standardize_dataset/remap_labels.py

Bước 3 — Kiểm tra chất lượng:
  python yolov9/standardize_dataset/check_images/check_error_images.py
  python yolov9/standardize_dataset/check_images/check_duplicate_images.py
  python yolov9/standardize_dataset/analyze_dataset.py

Bước 4 — Kiểm tra trực quan:
  python yolov9/standardize_dataset/check_dataset.py
```

> Nếu phát hiện sai sót sau Bước 2, có thể phục hồi bằng cách copy lại từ `labels_origin/` và chạy lại từ Bước 1.

---

## 3. Lịch sử phát triển Dataset

### v1 — Dataset gốc BDD100K (Raw)

**Nguồn:** Berkeley DeepDrive BDD100K — một trong những dataset giao thông lớn nhất thế giới, thu thập từ camera dashcam trên xe hơi tại các thành phố Mỹ.

**Đặc điểm:**
- 10 class: `bike, bus, car, motor, person, rider, traffic light, traffic sign, train, truck`
- Đa dạng điều kiện: ban ngày, ban đêm, mưa, nắng, sương mù, đường cao tốc, đô thị
- Định dạng đã chuyển sang YOLO bởi công cụ Ultralytics

**Hạn chế ở giai đoạn này:**
- Có nhiều class không phù hợp với bài toán điểm mù xe tải
- Class ID không khớp với schema mong muốn của dự án
- Chưa tách riêng các class ưu tiên (person đứng đầu, không phải bike)

---

### v2 — Lọc và Remap Class (Chuẩn hóa)

**Thời điểm:** Tháng 4/2026

**Những gì đã làm:**

1. **Backup toàn bộ labels gốc** thành `labels_origin/` trong cả 3 splits (train/valid/test) — bảo toàn dữ liệu gốc để có thể phục hồi bất kỳ lúc nào.

2. **Xóa 4 class không dùng** bằng `filter_labels.py`:
   - `rider` (ID 5): Đối tượng này về mặt ngữ nghĩa thực chất là người đang ngồi trên xe máy/xe đạp — trùng lặp với sự kết hợp `person + motor/bike`. Việc giữ lại gây nhiễu cho model.
   - `traffic light` (ID 6): Đèn tín hiệu không di chuyển, không gây nguy hiểm trực tiếp trong vùng điểm mù.
   - `traffic sign` (ID 7): Tương tự đèn tín hiệu — vật thể tĩnh, không phải mục tiêu cảnh báo.
   - `train` (ID 8): Tàu hỏa không xuất hiện trong kịch bản điểm mù xe tải đô thị/quốc lộ thông thường.

3. **Remap class ID** bằng `remap_labels.py` — sắp xếp lại thứ tự ưu tiên theo mức độ nguy hiểm và tần suất xuất hiện trong vùng điểm mù:
   - `0: person` — người đi bộ: nguy hiểm nhất, cần được ưu tiên phát hiện
   - `1: bike` — xe đạp: dễ bị khuất tầm nhìn, di chuyển chậm
   - `2: motor` — xe máy: phổ biến nhất trên đường Việt Nam
   - `3: car` — ô tô con: thường xuyên vào điểm mù khi vượt
   - `4: truck` — xe tải: nguy cơ va chạm nghiêm trọng
   - `5: bus` — xe buýt: kích thước lớn, khó xử lý khi vào điểm mù

**Kết quả sau chuẩn hóa:**
- Dataset sạch với đúng 6 class cần thiết
- Class ID nhất quán với `blindspot.yaml`, `configs/classes.yaml` và toàn bộ pipeline xử lý
- Dữ liệu gốc được bảo toàn hoàn toàn trong `labels_origin/`

---

### Tổng quan Dataset hiện tại

**Quy mô:** ~10.000 ảnh thực tế từ camera dashcam đường phố

| Split | Ảnh | Labels (ước tính) |
|---|---|---|
| train | 7.000 | ~100.000+ |
| valid | 1.000 | ~15.000+ |
| test  | 2.000 | (chưa annotate) |

**Độ đa dạng — Thời gian trong ngày:**
- Ban ngày ánh sáng tự nhiên đầy đủ
- Hoàng hôn / bình minh (ánh sáng thấp, tương phản cao)
- Ban đêm với đèn đường, đèn xe
- Điều kiện ánh sáng ngược (backlight, glare)

**Độ đa dạng — Thời tiết:**
- Trời nắng trong: màu sắc bão hòa cao, bóng đổ rõ
- Trời흐리/âm u: độ tương phản thấp, màu sắc nhạt
- Mưa nhỏ: mặt đường phản chiếu, kính ướt làm mờ ảnh
- Sương mờ nhẹ: giảm visibility, object xa bị mờ

**Độ đa dạng — Môi trường:**
- Đường đô thị đông đúc (nhiều loại phương tiện đan xen)
- Đường cao tốc / quốc lộ (tốc độ cao, khoảng cách lớn)
- Giao lộ, vòng xoay (nhiều hướng tiếp cận)
- Điều kiện ban đêm đô thị (đèn đường vàng, đèn neon)

**Hyperparameter augmentation** (trong `hyp.blindspot.yaml`) được tinh chỉnh phù hợp với độ đa dạng trên:
- `hsv_v: 0.5` — giả lập tất cả mức độ sáng từ ban ngày đến ban đêm
- `hsv_s: 0.7` — giả lập từ trời nắng chói đến mưa/sương mù
- `mosaic: 1.0` — ghép 4 ảnh khác nhau, tăng tối đa sự đa dạng scene mỗi iteration

---

## 4. Hướng dẫn train với dataset này

```bash
# Chạy từ thư mục yolov9/
python train_dual.py \
  --data data/blindspot.yaml \
  --cfg models/detect/yolov9-c.yaml \
  --weights yolov9-c.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 100 \
  --batch 16 \
  --img 640 \
  --name blindspot_v1
```

> Tham số `--data` phải trỏ đến `data/blindspot.yaml` (6 class).  
> Không dùng `bdd100k_ultralytics.yaml` (10 class gốc) để train.
