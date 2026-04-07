# Hướng dẫn Training — Truck Blind Spot Detection với YOLOv9

> Tài liệu này mô tả **chiến lược huấn luyện**, **ý nghĩa thông số**, **mục tiêu cần đạt**,
> và **cách đánh giá đúng cho bài toán blind spot detection** dùng YOLOv9.
>
> Dự án này **không phải object detection thông thường**.
> Model không chỉ cần phát hiện đúng vật thể, mà còn phải:
>
> * chạy **realtime**
> * ưu tiên **ít bỏ sót** ở các vùng nguy hiểm
> * hoạt động tốt trên pipeline có **ROI nhiều vùng**
> * đủ ổn định để tiến tới deploy trên video thực tế

---

# 1. Mục tiêu cốt lõi của bài toán

Bài toán của dự án là:

> Phát hiện vật thể nguy hiểm quanh xe tải trong các vùng điểm mù và vùng gần cabin,
> sau đó cảnh báo theo mức độ rủi ro.

Vì vậy, model tốt không chỉ là model có `mAP` cao.
Model tốt phải đồng thời đáp ứng:

1. **Phát hiện được người/xe máy đủ tốt**
2. **Ít bỏ sót ở vùng nguy hiểm**
3. **Chạy đủ nhanh cho video thời gian thực**
4. **Dùng được với pipeline ROI hiện tại**

---

# 2. Thứ tự ưu tiên đúng của dự án

Trong bài toán này, thứ tự ưu tiên nên là:

| Mức ưu tiên | Tiêu chí                         | Vì sao                                                            |
| ----------- | -------------------------------- | ----------------------------------------------------------------- |
| **#1**      | **Recall cao ở vùng nguy hiểm**  | Bỏ sót người/xe máy trong blind spot có thể gây tai nạn trực tiếp |
| **#2**      | **Realtime / latency thấp**      | Hệ thống cảnh báo chậm sẽ mất ý nghĩa thực tế                     |
| **#3**      | **Recall theo class quan trọng** | `person`, `motor`, `bike` là các class nhạy cảm nhất              |
| **#4**      | **mAP tổng thể**                 | Có giá trị, nhưng không đủ để kết luận model “an toàn”            |
| **#5**      | **Precision tổng thể**           | Báo nhầm gây phiền, nhưng thường vẫn đỡ nguy hiểm hơn bỏ sót      |

## Kết luận

Với project này:

* **Recall quan trọng hơn Precision**
* **ROI-aware recall quan trọng hơn Recall tổng**
* **Realtime quan trọng hơn việc cố đẩy mAP lên quá cao bằng model nặng**

---

# 3. Giai đoạn hiện tại: Pilot 10K ảnh là để làm gì?

Dataset hiện tại 10.000 ảnh chỉ là **pilot subset** của bộ đầy đủ khoảng 100.000 ảnh.

Mục tiêu của pilot **không phải** đạt độ chính xác tối đa cuối cùng.
Mục tiêu thật sự của pilot là:

* xác nhận pipeline train/val/infer chạy đúng
* tìm model variant phù hợp nhất
* chọn `img`, `batch`, `epochs`, `hyp` hợp lý
* xem class nào yếu
* xem ROI pipeline có hoạt động đúng không
* tạo ra checkpoint tốt để fine-tune ở giai đoạn full data

## Kết quả mong đợi của pilot

Sau pilot, bạn phải trả lời được:

* YOLOv9-T, S hay C phù hợp nhất?
* `img=640` có đủ chưa?
* model có overfit sớm không?
* recall của `person`, `motor`, `bike` có đủ chưa?
* inference speed có đủ nhanh cho demo và deploy không?
* metric theo vùng nguy hiểm có chấp nhận được không?

---

# 4. Giải thích chi tiết các thông số train quan trọng

---

## 4.1 `--data`

Ví dụ:

```bash
--data data/blindspot.yaml
```

Đây là file mô tả dataset, thường gồm:

* đường dẫn train set
* đường dẫn val set
* số lượng class
* tên class

Ví dụ:

```yaml
train: ...
val: ...
nc: 6
names: [person, bike, motor, car, truck, bus]
```

### Vai trò

File này nói cho YOLO biết:

* phải train trên tập nào
* validate trên tập nào
* label id `0`, `1`, `2` là class gì

### Nếu sai sẽ xảy ra gì?

* train nhầm dữ liệu
* tên class bị lệch
* metric vô nghĩa
* infer ra label sai

---

## 4.2 `--cfg`

Ví dụ:

```bash
--cfg models/detect/yolov9-s.yaml
```

Đây là kiến trúc model.

Các lựa chọn thường dùng:

* `yolov9-t`: tiny
* `yolov9-s`: small
* `yolov9-c`: compact / lớn hơn

### Ý nghĩa

Model càng nhỏ:

* nhanh hơn
* nhẹ hơn
* dễ deploy
* accuracy thường thấp hơn chút

Model càng lớn:

* accuracy có thể cao hơn
* chậm hơn
* tốn VRAM hơn
* khó đạt realtime hơn

### Khuyến nghị cho project này

* **YOLOv9-S** là lựa chọn base tốt nhất
* `YOLOv9-T` dùng làm baseline tốc độ
* `YOLOv9-C` dùng để kiểm tra xem accuracy tăng thêm có đáng đổi lấy tốc độ không

---

## 4.3 `--weights`

Ví dụ:

```bash
--weights yolov9-s.pt
```

hoặc:

```bash
--weights runs/train/pilot_small_v1/weights/best.pt
```

Đây là checkpoint khởi tạo.

### Có 3 kiểu dùng

#### A. Pretrained weights

Dùng model đã học từ trước trên dataset lớn.

Ưu điểm:

* hội tụ nhanh hơn
* cần ít dữ liệu hơn
* thường cho kết quả tốt hơn train từ đầu

#### B. Fine-tune từ pilot weights

Dùng `best.pt` từ pilot 10K để train tiếp trên 100K.

Ưu điểm:

* tiết kiệm thời gian
* tận dụng cấu hình tốt nhất đã tìm ra
* phù hợp nhất với dự án của bạn

#### C. Train từ đầu

Chỉ dùng khi:

* pretrained không phù hợp
* data quá khác biệt
* muốn nghiên cứu sâu hơn

Trong project này, **không nên ưu tiên train từ scratch** ở giai đoạn production.

---

## 4.4 `--img`

Ví dụ:

```bash
--img 640
```

Đây là kích thước ảnh đầu vào cho model.

Ảnh gốc có thể là:

* 1280×720
* 1920×1080

nhưng khi vào model sẽ được resize/letterbox về kích thước train, ví dụ `640×640`.

### `img` ảnh hưởng gì?

* ảnh lớn hơn:

  * dễ thấy vật nhỏ hơn
  * bbox chính xác hơn
  * chậm hơn
  * tốn VRAM hơn

* ảnh nhỏ hơn:

  * nhanh hơn
  * nhẹ hơn
  * dễ bỏ sót người/xe máy nhỏ hoặc ở xa

### Với dự án blind spot

`img=640` là điểm cân bằng tốt vì:

* đủ chi tiết cho `person`, `motor`, `bike`
* vẫn còn cơ hội đạt realtime
* phù hợp với YOLOv9-S

### Khi nào nên thử thêm?

Nên thử:

* `512`
* `640`
* `768`

Mục tiêu là xem:

* `512` có tăng FPS đáng kể không
* `768` có thực sự tăng recall đủ nhiều để đáng trả giá tốc độ không

### Khuyến nghị

* **Pilot:** `img=640`
* chỉ thử `512/768` như thí nghiệm bổ sung

---

## 4.5 `--batch`

Ví dụ:

```bash
--batch 32
```

Batch size = số ảnh được xử lý trước mỗi lần cập nhật trọng số.

### Ý nghĩa trực quan

Nếu dataset có 10.000 ảnh, batch 32 thì mỗi epoch có khoảng:

```text
10000 / 32 ≈ 313 steps
```

### Batch nhỏ

Ví dụ `8`:

* update thường xuyên hơn
* gradient nhiễu hơn
* dễ dao động

### Batch lớn

Ví dụ `64`:

* gradient ổn định hơn
* train nhanh hơn trên GPU mạnh
* nhưng ít step mỗi epoch hơn
* có thể không tối ưu cho pilot nhỏ

### Khuyến nghị cho dự án

* **Pilot 10K trên H100:** `batch=32`
* **Full 100K trên H100:** `batch=64`

### Vì sao không đẩy quá lớn ở pilot?

Pilot cần:

* nhiều lần thử
* so sánh model/config
* quan sát hội tụ rõ ràng

`batch=32` là cân bằng đẹp.

---

## 4.6 `--epochs`

Ví dụ:

```bash
--epochs 100
```

Một epoch = model đi qua toàn bộ train set 1 lần.

### Ý nghĩa

* ít epoch quá → model chưa học đủ
* nhiều epoch quá → dễ overfit

### Overfit là gì?

Model học quá kỹ train set:

* train loss giảm rất đẹp
* nhưng val loss tăng hoặc val metric đứng im / giảm

### Với pilot 10K

* 10–20 epoch: sanity check
* 50 epoch: đủ so model variant
* 100 epoch: đủ để chọn checkpoint tốt và quan sát overfit

### Với full 100K

* 100–150 epoch khi fine-tune là hợp lý
* 200–300 epoch chỉ nên tính tới nếu train từ scratch

### Khuyến nghị

* **Pilot:** `epochs=100`
* **Full:** `epochs=150`

---

## 4.7 `--patience`

Ví dụ:

```bash
--patience 20
```

Đây là early stopping.

### Ý nghĩa

Nếu metric validation không cải thiện trong 20 epoch liên tiếp, train dừng sớm.

### Vì sao cần?

* tránh train lãng phí
* tránh overfit kéo dài
* giữ checkpoint tốt nhất

### Khuyến nghị

* **Pilot 10K:** `patience=20`
* **Full 100K:** `patience=30`

---

## 4.8 `--hyp`

Ví dụ:

```bash
--hyp data/hyps/hyp.blindspot.yaml
```

Đây là file chứa hyperparameters:

* learning rate
* momentum
* weight decay
* augmentation
* loss weights

### Vì sao rất quan trọng?

Cùng một model, cùng một dataset, nhưng `hyp` khác nhau có thể làm:

* hội tụ nhanh hoặc chậm
* recall tăng hoặc giảm
* precision tăng hoặc giảm
* model overfit hoặc ổn định hơn

---

# 5. Giải thích các hyperparameter quan trọng trong `hyp`

Tùy repo YOLOv9, file `hyp` có thể hơi khác, nhưng bạn cần hiểu các nhóm sau.

---

## 5.1 Learning rate (`lr0`, `lrf`)

### `lr0`

Learning rate ban đầu.

Nếu quá lớn:

* loss rung mạnh
* metric không ổn định
* dễ train hỏng

Nếu quá nhỏ:

* train chậm
* metric tăng rất chậm

### `lrf`

Learning rate cuối cùng theo scheduler.

### Ý nghĩa thực tế

Learning rate quyết định:

> model thay đổi trọng số mạnh hay nhẹ sau mỗi batch

### Với pilot

Nếu thấy:

* loss không giảm
* mAP đứng im lâu
* training rất chậm

thì learning rate là thứ cần kiểm tra đầu tiên.

---

## 5.2 Momentum

Giúp update gradient mượt hơn.

Hiểu đơn giản:

* tránh việc model đổi hướng quá mạnh giữa các batch
* giúp quá trình học ổn định hơn

---

## 5.3 Weight decay

Regularization để chống overfit.

Nó phạt các trọng số quá lớn, giúp model tổng quát hóa tốt hơn.

Với dataset 10K, weight decay khá quan trọng vì nguy cơ overfit cao hơn full dataset.

---

## 5.4 Augmentation

Thường gồm:

* HSV
* scale
* translate
* flip
* mosaic
* mixup
* perspective

### Với dự án blind spot

Augmentation cần **vừa đủ**, không nên quá mạnh.

Vì bài toán của bạn có:

* vật thể ở mép ảnh
* vật thể sát cabin
* góc nhìn camera cố định
* hình học phối cảnh quan trọng

Nếu augmentation quá mạnh:

* méo bố cục camera
* làm model học sai phân bố thực tế
* ROI thực tế không còn tương thích tốt

### Khuyến nghị

* mosaic: giữ mức vừa phải
* hsv: dùng nhẹ
* perspective: không quá mạnh
* mixup: cân nhắc, không lạm dụng
* tránh augment quá cực đoan làm méo cảnh giao thông

---

# 6. Các tầng metric cần hiểu đúng

Đây là phần quan trọng nhất đối với project của bạn.

Model có nhiều “tầng đánh giá”, và mỗi tầng trả lời một câu hỏi khác nhau.

---

# Tầng 1 — Metric train/val tổng quát của YOLO

Đây là các metric bạn thấy trong:

* `results.csv`
* `results.png`
* console train

Bao gồm:

* `box_loss`
* `cls_loss`
* `dfl_loss`
* `precision`
* `recall`
* `mAP50`
* `mAP50-95`

## Tầng này trả lời câu hỏi gì?

> Model tổng thể đang học tốt hay không?

## Nhưng tầng này **không đủ**

Vì nó không cho bạn biết rõ:

* class nào yếu
* vùng nào yếu
* model có bỏ sót ở near cabin hay không

---

## 6.1 `box_loss`

Sai số định vị bounding box.

### Giảm tốt nghĩa là gì?

BBox model dự đoán ngày càng gần bbox thật hơn.

### Dấu hiệu tốt

* giảm đều theo epoch
* val box loss không tăng sớm

---

## 6.2 `cls_loss`

Sai số phân loại class.

### Giảm tốt nghĩa là gì?

Model ngày càng phân biệt đúng:

* person
* motor
* bike
* car
* truck
* bus

---

## 6.3 `dfl_loss`

Distribution Focal Loss, giúp định vị bbox tốt hơn.

Bạn không cần tối ưu nó riêng, chỉ cần thấy nó:

* giảm đều
* không dao động bất thường

---

## 6.4 `Precision`

Trong tất cả dự đoán model đưa ra, bao nhiêu là đúng.

### Precision thấp nghĩa là gì?

Model báo nhầm nhiều:

* cảnh báo thừa
* dễ gây khó chịu cho tài xế

---

## 6.5 `Recall`

Trong tất cả object thật có trong ảnh, model phát hiện được bao nhiêu.

### Recall thấp nghĩa là gì?

Model bỏ sót nhiều:

* có người mà không báo
* có xe máy mà không detect

### Trong project này

Recall quan trọng hơn precision.

---

## 6.6 `mAP@0.5`

Đây là chỉ số tổng hợp detection quan trọng nhất ở tầng tổng quát.

### Ý nghĩa

Model có:

* phát hiện đúng không
* bbox đủ khớp không
* confidence có hợp lý không

### Mức mục tiêu

* **Pilot 10K:** `mAP50 >= 0.70`
* **Full 100K:** `mAP50 >= 0.82`

---

## 6.7 `mAP@0.5:0.95`

Khắt khe hơn `mAP50`.

Nếu chỉ số này thấp nhưng `mAP50` khá:

* model tìm được object
* nhưng bbox chưa thật chính xác

### Mức mục tiêu

* **Pilot:** `>= 0.40`
* **Full:** `>= 0.55`

---

# Tầng 2 — Metric theo class

Tầng này trả lời câu hỏi:

> Model mạnh/yếu ở class nào?

Ví dụ:

* `Recall_person`
* `Recall_motor`
* `Recall_bike`
* `Precision_person`
* `mAP50_motor`

Đây là tầng cực quan trọng với dự án của bạn, vì:

* `person`, `motor`, `bike` là class nhạy cảm nhất
* recall tổng đẹp không đồng nghĩa recall của class quan trọng đẹp

## Ví dụ nguy hiểm

Model có:

* `Recall tổng = 0.82`

nhưng bóc ra:

* `Recall_person = 0.71`
* `Recall_motor = 0.76`
* `Recall_car = 0.92`

Thì model này **không đạt**, dù recall tổng nhìn đẹp.

---

## Mục tiêu theo class

| Class    | Precision tối thiểu | Recall tối thiểu | Ghi chú                                                                          |
| -------- | ------------------- | ---------------- | -------------------------------------------------------------------------------- |
| `person` | 0.70                | **0.82**         | quan trọng nhất                                                                  |
| `bike`   | 0.65                | **0.78**         | vật nhỏ, dễ khuất                                                                |
| `motor`  | 0.70                | **0.78**         | nên coi gần ngang `bike`, thậm chí quan trọng hơn nếu data Việt Nam nhiều xe máy |
| `car`    | 0.72                | 0.72             | cân bằng                                                                         |
| `truck`  | 0.70                | 0.70             | tương đối dễ                                                                     |
| `bus`    | 0.68                | 0.70             | ít mẫu hơn                                                                       |

## Gợi ý thực tế

Nếu dataset của bạn nhiều xe máy hơn xe đạp, nên coi:

* `person`
* `motor`
* `bike`

là 3 class trọng tâm.

---

# Tầng 3 — Metric theo vùng ROI

Đây là tầng quan trọng nhất để bài toán của bạn trở thành **blind spot detection** thật sự, thay vì chỉ là detection thông thường.

Tầng này trả lời:

> Model detect tốt đến đâu trong từng vùng nguy hiểm?

Ví dụ:

* `Recall_near_cabin_zone`
* `Recall_left_blind_spot`
* `Recall_right_blind_spot`
* `Recall_forward_danger_zone`
* `Recall_rear_danger_zone`

YOLO mặc định **không tính tầng này cho bạn**.
Bạn phải tự tính bằng pipeline ROI hiện tại.

---

## Vì sao phải có tầng 3?

Vì hậu quả bỏ sót ở mỗi vùng không giống nhau.

### `near_cabin_zone`

Đây là vùng nguy hiểm nhất:

* vật ở rất gần đầu cabin
* thời gian phản ứng rất ngắn

=> recall ở vùng này phải cao nhất

### `left_blind_spot` / `right_blind_spot`

Đây là vùng mù khi rẽ/chuyển làn

=> cũng phải rất cao

### `forward_danger_zone`

Cũng nguy hiểm, nhưng:

* camera nhìn rõ hơn
* tài xế có thể tự quan sát phần nào

=> có thể thấp hơn near cabin một chút

---

## Mục tiêu theo ROI zone

| ROI Zone              | Recall mục tiêu pilot | Recall mục tiêu full | Ghi chú                 |
| --------------------- | --------------------- | -------------------- | ----------------------- |
| `near_cabin_zone`     | **≥ 0.85**            | **≥ 0.90**           | vùng nguy hiểm nhất     |
| `left_blind_spot`     | ≥ 0.80                | ≥ 0.85               | cực quan trọng          |
| `right_blind_spot`    | ≥ 0.80                | ≥ 0.85               | cực quan trọng          |
| `forward_danger_zone` | ≥ 0.78                | ≥ 0.82               | quan trọng nhưng dễ hơn |
| `rear_danger_zone`    | ≥ 0.82                | ≥ 0.88               | nếu dùng camera sau     |

---

## Ví dụ tại sao ROI metric quan trọng

Giả sử model có:

* `Recall tổng = 0.84`
* `mAP50 = 0.78`

Nhìn có vẻ đẹp.

Nhưng theo vùng:

* `Recall_forward = 0.91`
* `Recall_left = 0.79`
* `Recall_right = 0.77`
* `Recall_near_cabin = 0.66`

Thì model này **không an toàn**.

Lý do:

* nó mạnh ở vùng dễ
* yếu ở vùng nguy hiểm nhất

---

# 7. Cách hiểu đúng các file metric bạn sẽ thấy

---

## 7.1 `results.csv`

Thường có:

* epoch
* train losses
* val losses
* precision
* recall
* mAP50
* mAP50-95

### File này dùng để làm gì?

* xem model có học không
* xem loss có giảm không
* xem overfit có xảy ra không
* xem mAP tổng tăng đến đâu

### Không dùng file này để làm gì?

* không đủ để kết luận recall theo class
* không đủ để kết luận recall theo ROI zone

---

## 7.2 `results.png`

Giúp nhìn nhanh:

* loss curve
* precision curve
* recall curve
* mAP curve

### Dùng để trả lời:

* model hội tụ chưa?
* overfit sớm không?
* mAP còn tăng nữa không?

---

## 7.3 `confusion_matrix.png`

Giúp biết:

* class nào bị nhầm với class nào

Ví dụ:

* `motor` hay bị nhầm thành `bike`
* `truck` hay bị nhầm thành `bus`

---

## 7.4 `val_dual.py`

Đây là nơi nên dùng để lấy:

* metric tốt nhất sau train
* kết quả validation sạch hơn
* metric theo class nếu repo hỗ trợ in ra

---

# 8. Khi nào model được coi là “đạt chuẩn”?

## 8.1 Chuẩn cho Pilot 10K

Model được coi là pass pilot nếu thỏa phần lớn các điều kiện sau:

### Tầng 1

* `mAP50 >= 0.70`
* `mAP50-95 >= 0.40`
* loss giảm ổn định
* không overfit quá sớm trước epoch 50

### Tầng 2

* `Recall_person >= 0.80`
* `Recall_motor >= 0.75`
* `Recall_bike >= 0.75`

### Tầng 3

* `Recall_near_cabin >= 0.85`
* `Recall_left/right_blind_spot >= 0.80`
* `Recall_in_ROI` tổng thể >= 0.82

### Runtime

* infer video chạy ổn định
* không crash
* FPS đủ cho demo và benchmark tiếp

---

## 8.2 Chuẩn cho Full 100K

Model được coi là đủ mạnh để tiến tới production khi đạt:

### Tầng 1

* `mAP50 >= 0.82`
* `mAP50-95 >= 0.55`
* `Precision tổng >= 0.75`

### Tầng 2

* `Recall_person >= 0.82`
* `Recall_motor >= 0.80`
* `Recall_bike >= 0.78`

### Tầng 3

* `Recall_near_cabin >= 0.90`
* `Recall_left/right_blind_spot >= 0.85`
* `Recall_forward_danger_zone >= 0.82`

### Runtime

* latency <= 50 ms/frame
* realtime >= 20 FPS trên thiết bị mục tiêu

---

# 9. Chiến lược training tối ưu cho dự án này

---

## Giai đoạn 1 — Sanity check

Mục tiêu:

* xác nhận data đúng
* loss giảm
* pipeline train/val chạy ổn

### Khuyến nghị

* model: `YOLOv9-S`
* epochs: `10–20`
* batch: `32`
* img: `640`

Ví dụ:

```bash
python train_dual.py \
  --data data/blindspot.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --weights yolov9-s.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 20 \
  --batch 32 \
  --img 640 \
  --name sanity_small
```

---

## Giai đoạn 2 — So sánh model variant

Mục tiêu:

* chọn model phù hợp nhất giữa T / S / C

### Khuyến nghị

* mỗi model train 50 epochs
* giữ cùng `img=640`, `batch=32`

```bash
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

### Sau bước này cần chọn theo gì?

Không chọn chỉ theo `mAP`.

Phải so cùng lúc:

* mAP50
* recall person / motor / bike
* FPS
* model size
* độ ổn định video infer
* tiềm năng đạt ROI recall tốt

---

## Giai đoạn 3 — Train pilot sâu hơn với model tốt nhất

Sau khi chọn model tốt nhất, train 100 epochs để lấy checkpoint chuẩn pilot.

### Khuyến nghị

* model: thường là `YOLOv9-S`
* epochs: `100`
* batch: `32`
* patience: `20`

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

---

## Giai đoạn 4 — Fine-tune trên full 100K

Dùng `best.pt` từ pilot để train tiếp.

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

---

# 10. Cách đánh giá model sau train

Sau khi train xong, đừng chỉ nhìn `results.csv`.

Bạn nên đánh giá theo 3 tầng.

---

## 10.1 Tầng 1 — Global metrics

Dùng:

* `results.csv`
* `results.png`
* `val_dual.py`

Mục tiêu:

* xem hội tụ
* xem mAP tổng
* xem precision/recall tổng

---

## 10.2 Tầng 2 — Per-class metrics

Dùng:

* output của `val_dual.py`
* confusion matrix
* PR curves nếu repo hỗ trợ

Mục tiêu:

* biết class nào yếu
* biết class nào cần oversample / augment thêm

---

## 10.3 Tầng 3 — ROI-aware metrics

Dùng pipeline ROI hiện tại của bạn để tự tính:

* GT bbox nào nằm trong zone nào
* prediction nào match GT nào
* từ đó tính recall theo zone

### Tầng này phải tự làm

YOLO không tự cho bạn metric theo zone.

Nhưng project của bạn đã có:

* `zone_name`
* `risk_level`
* `in_roi`
* `bbox_bottom_center`

nên hoàn toàn có thể xây bộ đánh giá riêng.

---

# 11. Các lỗi thường gặp khi train project này

## 11.1 Chỉ nhìn mAP tổng

Sai vì:

* mAP tổng đẹp nhưng vùng nguy hiểm có thể rất yếu

## 11.2 Chỉ nhìn recall tổng

Sai vì:

* recall tổng có thể bị kéo lên bởi `car`, `truck`
* trong khi `person` hoặc `motor` vẫn kém

## 11.3 Chỉ train model lớn hơn để tăng accuracy

Sai vì:

* deploy không đạt realtime
* hệ thống cảnh báo mất tính thực tế

## 11.4 Augmentation quá mạnh

Sai vì:

* phá bố cục camera
* làm giảm tính phù hợp với ROI thật

## 11.5 Không benchmark video thật

Sai vì:

* metric ảnh tốt chưa chắc video ổn
* có thể bbox nhấp nháy, miss object ngắn hạn

---

# 12. Checklist chuẩn cho dự án này

## A. Đạt chuẩn pilot

* [ ] train chạy ổn định
* [ ] `mAP50 >= 0.70`
* [ ] `mAP50-95 >= 0.40`
* [ ] `Recall_person >= 0.80`
* [ ] `Recall_motor/bike >= 0.75`
* [ ] `Recall_near_cabin >= 0.85`
* [ ] infer video ổn định
* [ ] ROI pipeline hoạt động đúng

## B. Đạt chuẩn full dataset

* [ ] `mAP50 >= 0.82`
* [ ] `mAP50-95 >= 0.55`
* [ ] `Recall_person >= 0.82`
* [ ] `Recall_motor >= 0.80`
* [ ] `Recall_bike >= 0.78`
* [ ] `Recall_near_cabin >= 0.90`
* [ ] `Recall_side_blind_spot >= 0.85`
* [ ] latency <= 50 ms
* [ ] realtime >= 20 FPS
* [ ] model size phù hợp deploy

---

# 13. Cấu hình khuyến nghị nên chốt hiện tại

## Pilot 10K

* Model chính: `YOLOv9-S`
* Baseline phụ: `YOLOv9-T`, `YOLOv9-C`
* `img=640`
* `batch=32`
* `epochs=100`
* `patience=20`

## Full 100K

* Fine-tune từ `pilot_small_v1/weights/best.pt`
* `img=640`
* `batch=64`
* `epochs=150`
* `patience=30`

---

# 14. Lệnh train khuyến nghị chính thức

## Pilot

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

## Full

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

## Validation

```bash
python val_dual.py \
  --data data/blindspot.yaml \
  --weights runs/train/fulldata_v1/weights/best.pt \
  --img 640 \
  --batch 32
```

---

# 15. Tóm tắt cuối cùng

## Mục tiêu thật của dự án này không phải chỉ là:

* “mAP cao”

## Mà là:

* detect tốt người / xe máy
* ít bỏ sót trong vùng nguy hiểm
* chạy đủ nhanh trên video thực
* dùng tốt với pipeline ROI blind spot

## Ba tầng metric cần nhớ:

* **Tầng 1:** metric YOLO tổng quát
  → model có học tốt không?

* **Tầng 2:** metric theo class
  → class nào đang yếu?

* **Tầng 3:** metric theo ROI zone
  → model có an toàn trong vùng nguy hiểm không?

## Quyết định hiện tại nên chốt:

* dùng **YOLOv9-S** làm base model
* pilot 10K với `batch=32`, `epochs=100`, `img=640`
* đánh giá không chỉ bằng mAP tổng, mà phải thêm:

  * recall theo class
  * recall theo ROI zone

---

