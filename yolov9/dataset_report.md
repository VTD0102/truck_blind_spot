Dataset source

Dataset ban đầu từ:
Roboflow vehicle-counting dataset

Dataset đúng format YOLO
Class chuẩn:
0 person
1 bicycle
2 car
3 motorcycle

Chú ý thực hiện train model với data: blindspot.yaml
                                hyp: hyp.blindspot.yaml (vì làm vào ban ngày nên cần căn chỉnh độ sáng, độ nhận diện của ảnh)
Ví dụ: python train.py \
  --data data/blindspot.yaml \
  --cfg models/detect/yolov9n.yaml \
  --weights yolov9n.pt \
  --hyp data/hyps/hyp.blindspot.yaml \
  --epochs 50 \
  --batch 8 \
  --img 640

Các bước đã làm: 
remap class:    bus → car
                truck → car
auto-label person & bicycle
kiểm tra bbox thủ công

Cấu trúc thư mục datasets
data/datasets/
├── train
│   ├── images
│   └── labels
├── valid
│   ├── images
│   └── labels
└── test
    ├── images
    └── labels

Cấu trúc thư mục folder standardize_dataset
standardize_dataset
├── check images (kiểm tra xem có ảnh trùng lặp hay ảnh lỗi không)
│   ├── check_duplicate_images.py
│   └── check_error_images.py
├── check_dataset.py (kiểm tra bbox của tất cả các ảnh)
├── analyze_dataset.py (đếm số class trong toàn bộ dataset mẫu)
├── remap_labels.py (remap lại các class sử dụng trong dataset, vì dataset cũ chỉ làm
                        các class truck, bus, không làm person, bicycle)
└── auto_add_person_bicycle.py (tự động add các label person, bicycle cho từng ảnh)


Dataset statistics: Kết quả đếm số lượng class trong toàn bộ dataset
person	3914
bicycle	1234 (ít nhất)
car	6168
motorcycle	10968 (nhiều nhất)

Dataset Version History (lịch sử chỉnh sửa, chuẩn hóa dataset)
v1_raw
- Original dataset from Roboflow/GitHub
- Classes: bus, car, motorcycle, truck

v2_remap
- Remapped bus -> car
- Remapped truck -> car
- Standardized class IDs for project

v3_auto_label
- Added person and bicycle using pretrained auto-label
- Updated label files for train/valid/test

v4_final
- Manual inspection completed
- Corrected selected bounding boxes
- Final dataset for YOLOv9 training



