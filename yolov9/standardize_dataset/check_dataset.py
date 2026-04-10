import cv2
import os

# ============================================================
# check_dataset.py
# Hiển thị ảnh kèm bounding box và tên class để kiểm tra
# label có đúng không sau khi filter và remap.
#
# Điều khiển:
#   SPACE / Enter : ảnh tiếp theo
#   'q'           : thoát
#
# Đổi SPLIT thành "val" để kiểm tra validation set.
# ============================================================

SPLIT = "train"     # "train" hoặc "val"

# Tên class theo blindspot.yaml
CLASS_NAMES = {
    0: "person",
    1: "bike",
    2: "motor",
    3: "car",
    4: "truck",
    5: "bus",
}

# Màu bbox theo từng class (BGR)
COLORS = {
    0: (0,   255,   0),     # person - xanh lá
    1: (0,   165, 255),     # bike   - cam
    2: (255, 128,   0),     # motor  - xanh dương nhạt
    3: (0,     0, 255),     # car    - đỏ
    4: (128,   0, 255),     # truck  - tím
    5: (255,   0, 128),     # bus    - hồng
}

img_path   = f"yolov9/data/datasets/{SPLIT}/images"
label_path = f"yolov9/data/datasets/{SPLIT}/labels"

files = sorted([f for f in os.listdir(img_path)
                if f.lower().endswith((".jpg", ".png"))])

print(f"[{SPLIT}] {len(files)} ảnh. SPACE/Enter = tiếp, 'q' = thoát.")

for file in files:
    img_file   = os.path.join(img_path, file)
    label_file = os.path.join(label_path, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(img_file)
    if img is None:
        print(f"Không đọc được ảnh: {img_file}")
        continue

    h, w = img.shape[:2]

    if os.path.exists(label_file):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id      = int(parts[0])
                x, y, bw, bh = map(float, parts[1:])

                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                color = COLORS.get(cls_id, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                label = CLASS_NAMES.get(cls_id, str(cls_id))
                cv2.putText(img, label, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    else:
        print(f"Không có label: {label_file}")

    cv2.imshow("check_dataset", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
