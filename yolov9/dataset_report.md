Dataset source

Dataset ban đầu từ:

Roboflow vehicle-counting dataset
Dataset preprocessing

Các bước đã làm: remap class
bus → car
truck → car
auto-label person & bicycle
kiểm tra bbox thủ công

Dataset statistics: đếm số lượng class trong toàn bộ dataset
person	3914
bicycle	1234
car	6168
motorcycle	10968
Dataset challenges

Ví dụ:
một số ảnh mờ
occlusion giữa person và motorcycle
class imbalance

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