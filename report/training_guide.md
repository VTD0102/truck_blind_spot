# Hướng dẫn Train Model — Truck Blind Spot Detection

> Tài liệu này tập trung vào **chiến lược huấn luyện**, **chỉ số cần đạt** và **tiêu chuẩn xuất sắc**  
> cho dự án phát hiện điểm mù xe tải sử dụng YOLOv9.  
>
> **Bối cảnh quan trọng:** Dataset hiện tại 10.000 ảnh là **tập kiểm nghiệm pilot** trích từ  
> bộ dữ liệu đầy đủ ~100.000 ảnh. Mục tiêu của giai đoạn này là **xác nhận pipeline hoạt động đúng**  
> và **tìm ra cấu hình tốt nhất** trước khi train toàn bộ dữ liệu.

---

## 1. Định hướng ưu tiên: Realtime hay Độ chính xác?

Đây là câu hỏi quan trọng nhất cần trả lời trước khi quyết định bất kỳ thông số nào.

### Bài toán này đòi hỏi **cả hai — nhưng theo thứ tự ưu tiên rõ ràng**

| Ưu tiên | Yêu cầu | Lý do |
|---|---|---|
| **#1 — Tốc độ realtime** | ≥ 20 FPS trên thiết bị nhúng | Camera điểm mù phải phản hồi tức thì khi xe chuyển làn. Cảnh báo trễ 1–2 giây = tai nạn |
| **#2 — Recall cao cho person/bike** | Recall ≥ 0.80 với người & xe đạp | Bỏ sót người đi bộ hoặc xe đạp ở điểm mù là nguy hiểm tính mạng — **sai lầm không thể chấp nhận** |
| **#3 — Precision tổng thể** | Precision ≥ 0.70 | Cảnh báo nhầm gây phiền nhưng vẫn tốt hơn bỏ sót. Tuy nhiên quá nhiều false alarm khiến tài xế bỏ qua |

> **Kết luận:** Ưu tiên **Recall > Precision** với class `person` và `bike`.  
> Ưu tiên **tốc độ inference** trên **độ chính xác tuyệt đối** với các class còn lại.  
> Model vừa-nhỏ (YOLOv9-S, YOLOv9-C) phù hợp hơn model lớn (YOLOv9-E).

---

## 2. Batch Size và Epoch

### 2.1 Batch Size

Batch size ảnh hưởng đến **tốc độ train, độ ổn định gradient và khả năng tổng quát hóa**.

| GPU VRAM | Batch Size khuyến nghị | Ghi chú |
|---|---|---|
| 8 GB (RTX 3070/4060) | 8–16 | Dùng `--accumulate` nếu cần batch lớn hơn |
| 16–24 GB (RTX 3090/4090, A10) | 16–32 | Phù hợp cho 10K dataset |
| 40–80 GB (A100, H100) | 32–64 | **Môi trường hiện tại (Colab Pro H100)** |

**Quy tắc:**
- Batch quá nhỏ (< 8): gradient nhiễu, model học không ổn định
- Batch quá lớn (> 64 với 10K ảnh): ít bước update mỗi epoch, model hội tụ chậm
- **Khuyến nghị cho 10K dataset trên H100: `batch=32`**
- **Khuyến nghị cho 100K dataset trên H100: `batch=64`**

### 2.2 Epoch

Số epoch phụ thuộc vào **kích thước dataset** và **mục tiêu giai đoạn**.

#### Giai đoạn Pilot (10K ảnh — hiện tại)

| Mục tiêu | Epochs | Lý do |
|---|---|---|
| Kiểm tra pipeline (sanity check) | 10–20 | Chỉ cần xác nhận loss giảm, không overfit ngay |
| So sánh model variant (t/s/c) | 50 | Đủ để phân biệt hiệu năng giữa các model |
| Tìm hyperparameter tốt nhất | 100 | **Đây là mục tiêu chính của pilot** |

> Với 10K ảnh, model thường **bắt đầu overfit sau epoch 80–120**.  
> Dừng sớm (early stopping) nếu `val/mAP50` không tăng sau 20 epoch liên tiếp.

#### Giai đoạn Full Train (100K ảnh — sắp tới)

| Mục tiêu | Epochs | Ghi chú |
|---|---|---|
| Train từ đầu (scratch) | 200–300 | Dataset lớn cần nhiều epoch để hội tụ |
| Fine-tune từ pilot weights | 100–150 | Khởi đầu từ điểm đã tốt, hội tụ nhanh hơn |
| Fine-tune từ pretrained YOLOv9 | 100 | Khuyến nghị — tiết kiệm thời gian và tài nguyên |

**Lưu ý quan trọng:** Với 100K ảnh, mỗi epoch mất gấp ~10 lần thời gian so với 10K. Ưu tiên `fine-tune` từ weights tốt nhất của pilot thay vì train từ đầu.

---

## 3. Các chỉ số cần theo dõi và ngưỡng đạt chuẩn

### 3.1 Chỉ số chính (Primary Metrics)

#### `mAP@0.5` — Mean Average Precision tại IoU 0.5

Chỉ số **quan trọng nhất** — đo độ chính xác tổng thể của model.

| Mức | Giá trị | Đánh giá |
|---|---|---|
| Không đạt | < 0.55 | Model chưa học được, cần kiểm tra lại data/config |
| Cơ bản | 0.55 – 0.65 | Hoạt động được nhưng còn nhiều sai sót |
| **Đạt chuẩn** | **0.65 – 0.75** | **Chấp nhận được cho pilot 10K ảnh** |
| Tốt | 0.75 – 0.82 | Mục tiêu với 100K ảnh |
| **Xuất sắc** | **≥ 0.82** | **Mục tiêu cuối cùng của dự án** |

> Kết quả pilot hiện tại: YOLOv9-S đạt **mAP@0.5 = 0.70** — nằm ở ngưỡng **đạt chuẩn**, hoàn toàn phù hợp cho giai đoạn kiểm nghiệm 10K ảnh.

---

#### `mAP@0.5:0.95` — COCO Standard (nghiêm ngặt hơn)

Đo độ chính xác trung bình trên nhiều ngưỡng IoU từ 0.5 đến 0.95 (bước 0.05).

| Mức | Giá trị | Đánh giá |
|---|---|---|
| Không đạt | < 0.30 | |
| Cơ bản | 0.30 – 0.40 | |
| **Đạt chuẩn** | **0.40 – 0.50** | **Mục tiêu pilot** |
| Xuất sắc | **≥ 0.55** | **Mục tiêu 100K dataset** |

---

#### `Precision` và `Recall` — Theo từng class

> Đây là chỉ số **quan trọng hơn mAP** cho bài toán an toàn.

**Công thức nhớ nhanh:**
- **Precision** = Trong số tất cả cảnh báo phát ra, bao nhiêu % là đúng → đo false alarm
- **Recall** = Trong số tất cả vật thể thật, bao nhiêu % được phát hiện → đo bỏ sót

| Class | Precision tối thiểu | Recall tối thiểu | Lý do |
|---|---|---|---|
| `person` | 0.70 | **0.82** | Bỏ sót người = nguy hiểm tính mạng |
| `bike` | 0.65 | **0.78** | Xe đạp nhỏ, dễ khuất — ưu tiên recall |
| `motor` | 0.70 | 0.75 | Phổ biến, phải phát hiện tốt |
| `car` | 0.72 | 0.72 | Cân bằng — xe hơi dễ phát hiện |
| `truck` | 0.70 | 0.70 | Kích thước lớn, không khó phát hiện |
| `bus` | 0.68 | 0.70 | Ít xuất hiện hơn trong điểm mù |

---

#### Loss Functions — Theo dõi trong quá trình train

| Loss | Ý nghĩa | Dấu hiệu tốt |
|---|---|---|
| `box_loss` | Sai số tọa độ bbox | Giảm đều, cuối train < 0.04 |
| `cls_loss` | Sai số phân loại class | Giảm đều, cuối train < 0.02 |
| `dfl_loss` | Distribution Focal Loss — tinh chỉnh bbox | Giảm đều, ổn định |
| `val/box_loss` | Box loss trên validation | **Không được tăng** sau epoch 50 |
| `val/cls_loss` | Cls loss trên validation | **Không được tăng** — dấu hiệu overfit |

> **Cảnh báo overfit:** Nếu `train loss` tiếp tục giảm nhưng `val loss` bắt đầu tăng → dừng train ngay, lấy `best.pt` (checkpoint tốt nhất).

---

#### `FPS` — Tốc độ inference (Realtime)

| Thiết bị | FPS tối thiểu | FPS mục tiêu | Ghi chú |
|---|---|---|---|
| GPU server (RTX 3090+) | 60 FPS | 100+ FPS | Kiểm thử, không deploy thực tế |
| Jetson Orin / AGX | 25 FPS | 40+ FPS | Thiết bị nhúng trên xe |
| Jetson Nano / Xavier NX | 15 FPS | 25+ FPS | Thiết bị nhúng cấp thấp hơn |
| CPU-only (fallback) | 5 FPS | 10 FPS | Không khuyến nghị cho production |

> Nếu FPS không đạt trên thiết bị mục tiêu → **giảm kích thước model** (dùng YOLOv9-T hoặc YOLOv9-S)  
> trước khi cân nhắc giảm `--img` xuống 416.

---

### 3.2 Chỉ số phụ (Secondary Metrics)

| Chỉ số | Ý nghĩa | Mục tiêu |
|---|---|---|
| `F1-score` | Trung hòa giữa Precision và Recall | ≥ 0.72 tổng thể |
| `val/fitness` | Điểm tổng hợp YOLOv9 (0.1×mAP50 + 0.9×mAP50:95) | Tối đa hóa |
| Model size (MB) | Kích thước file `.pt` | ≤ 50 MB cho deploy nhúng |
| Latency (ms/frame) | Thời gian xử lý mỗi khung hình | ≤ 50ms (= 20 FPS) |

---

## 4. Chiến lược train để đạt mức Xuất sắc

### 4.1 Giai đoạn 1 — Pilot 10K (Đang thực hiện)

**Mục tiêu:** Tìm ra model variant và hyperparameter tốt nhất, không phải đạt độ chính xác cao nhất.

```bash
# So sánh nhanh 3 variants trong 50 epochs
python train_dual.py --data data/blindspot.yaml \
  --cfg models/detect/yolov9-t.yaml --weights yolov9-t.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 50 --batch 32 --img 640 --name pilot_tiny

python train_dual.py --data data/blindspot.yaml \
  --cfg models/detect/yolov9-s.yaml --weights yolov9-s.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 50 --batch 32 --img 640 --name pilot_small

python train_dual.py --data data/blindspot.yaml \
  --cfg models/detect/yolov9-c.yaml --weights yolov9-c.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 50 --batch 32 --img 640 --name pilot_compact
```

**Câu hỏi cần trả lời sau giai đoạn này:**
- [ ] Model nào cho FPS tốt nhất trên thiết bị mục tiêu?
- [ ] Model nào cho Recall tốt nhất với class `person` và `bike`?
- [ ] Loss có hội tụ ổn định không? Có overfit sớm không?
- [ ] Augmentation hiện tại (`hyp.blindspot.yaml`) có phù hợp không?

---

### 4.2 Giai đoạn 2 — Full Train 100K (Sắp tới)

**Mục tiêu:** Đạt mAP@0.5 ≥ 0.82, Recall person ≥ 0.82, chạy realtime ≥ 20 FPS trên thiết bị nhúng.

#### Bước 2.1 — Fine-tune từ pilot weights tốt nhất

```bash
# Lấy best.pt từ model variant tốt nhất ở Giai đoạn 1
python train_dual.py --data data/blindspot.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights runs/train/pilot_small/weights/best.pt \   # khởi đầu từ pilot
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 150 --batch 64 --img 640 \
  --name fulldata_v1
```

#### Bước 2.2 — Theo dõi và can thiệp sớm

| Epoch | Điểm kiểm tra | Hành động nếu cần |
|---|---|---|
| 10–20 | Loss phải giảm ổn định | Nếu loss không giảm → kiểm tra lại data pipeline |
| 50 | mAP@0.5 ≥ 0.70 | Nếu chưa đạt → tăng epochs, giảm lr, kiểm tra augmentation |
| 100 | mAP@0.5 ≥ 0.78 | Nếu chưa đạt → cân nhắc dùng model lớn hơn (YOLOv9-C) |
| Kết thúc | mAP@0.5 ≥ 0.82 | Nếu chưa đạt → fine-tune thêm với lr nhỏ hơn |

#### Bước 2.3 — Fine-tune tập trung vào class khó (nếu cần)

Nếu sau full train, `person` hoặc `bike` vẫn thấp:

```bash
# Train thêm với loss weight tăng cho class nhạy cảm
# Chỉnh cls: 0.5 -> 0.8 trong hyp.blindspot.yaml
# Tăng dữ liệu có person/bike bằng cách oversample
```

---

### 4.3 Checklist đạt mức Xuất sắc

Một model được coi là **xuất sắc** khi đáp ứng **tất cả** các tiêu chí sau:

#### Độ chính xác
- [ ] `mAP@0.5` ≥ 0.82 trên validation set
- [ ] `mAP@0.5:0.95` ≥ 0.55 trên validation set
- [ ] `Recall (person)` ≥ 0.82
- [ ] `Recall (bike)` ≥ 0.78
- [ ] `Precision` tổng thể ≥ 0.75 (không quá nhiều false alarm)

#### Tốc độ
- [ ] Inference ≥ 20 FPS trên thiết bị nhúng mục tiêu
- [ ] Latency ≤ 50ms mỗi frame ở độ phân giải 640×640

#### Độ bền (Robustness)
- [ ] Không sụt mAP quá 10% khi test trên ảnh ban đêm
- [ ] Không sụt mAP quá 10% khi test trên ảnh mưa/sương
- [ ] Model size ≤ 50 MB (deploy được trên thiết bị nhúng)

#### Ổn định khi train
- [ ] `val loss` không tăng trước epoch 80 (không overfit sớm)
- [ ] Chênh lệch `train mAP` và `val mAP` ≤ 0.08 (không overfit nặng)

---

## 5. Lộ trình từ Pilot đến Production

```
[Hiện tại]
10K ảnh (pilot)
  │
  ├── Mục tiêu: mAP ≥ 0.70, xác nhận pipeline OK
  ├── Kết quả đã đạt: YOLOv9-S mAP = 0.70 ✓
  └── Bài học: chọn YOLOv9-S làm base model
         │
         ▼
[Sắp tới — Giai đoạn 2]
100K ảnh (full dataset)
  │
  ├── Fine-tune từ pilot_small/best.pt
  ├── Epochs: 150, Batch: 64, GPU: H100
  ├── Mục tiêu: mAP ≥ 0.82, Recall(person) ≥ 0.82
  └── Thời gian ước tính: 3–5 giờ trên H100
         │
         ▼
[Sau full train — Giai đoạn 3]
Đánh giá & Tối ưu hóa
  │
  ├── Benchmark FPS trên thiết bị nhúng
  ├── Test trên video ban đêm, mưa (edge cases)
  ├── Export sang TensorRT / ONNX nếu cần tăng FPS
  └── Deploy lên hệ thống cảnh báo thực tế
```

---

## 6. Lệnh train tham khảo

### Pilot 10K — Tìm model tốt nhất
```bash
python train_dual.py \
  --data data/blindspot.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights yolov9-s.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 100 \
  --batch 32 \
  --img 640 \
  --patience 20 \
  --name pilot_small_v1
```

### Full 100K — Fine-tune từ best pilot weights
```bash
python train_dual.py \
  --data data/blindspot.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights runs/train/pilot_small_v1/weights/best.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 150 \
  --batch 64 \
  --img 640 \
  --patience 30 \
  --name fulldata_v1
```

### Validation sau train
```bash
python val_dual.py \
  --data data/blindspot.yaml \
  --weights runs/train/fulldata_v1/weights/best.pt \
  --img 640 \
  --batch 32
```

> **`--patience`**: Số epoch không cải thiện trước khi early stop.  
> Đặt `--patience 20` cho pilot (dataset nhỏ, overfit nhanh hơn).  
> Đặt `--patience 30` cho full dataset.

---

## 7. Tóm tắt nhanh

| Câu hỏi | Trả lời |
|---|---|
| Ưu tiên gì? | **Realtime trước, Recall(person/bike) thứ hai, mAP tổng thể thứ ba** |
| Batch size tốt nhất (H100)? | **32** cho 10K / **64** cho 100K |
| Epochs cho pilot 10K? | **100** (early stop nếu val không tăng sau 20 epoch) |
| Epochs cho full 100K? | **150** fine-tune từ pilot weights |
| mAP@0.5 cần đạt? | **≥ 0.70** (pilot) / **≥ 0.82** (full) |
| Recall person cần đạt? | **≥ 0.82** (không thương lượng) |
| Model nên dùng? | **YOLOv9-S** — cân bằng tốt nhất giữa tốc độ và độ chính xác |
| Khi nào dừng train? | Khi `val/mAP50` không tăng sau 20–30 epoch liên tiếp |
