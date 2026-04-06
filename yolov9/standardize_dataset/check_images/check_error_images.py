import cv2
import os

# ============================================================
# check_error_images.py
# Tìm ảnh bị lỗi / không đọc được trong từng split.
# Dùng OpenCV để thử đọc từng ảnh — nếu trả về None là lỗi.
# Chạy từ thư mục gốc dự án.
# ============================================================

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    img_dir = f"yolov9/data/datasets/{split}/images"
    if not os.path.exists(img_dir):
        print(f"[{split}] Không tìm thấy thư mục: {img_dir}")
        continue

    errors = []     # danh sách ảnh lỗi
    total  = 0      # tổng số ảnh kiểm tra

    for img in sorted(os.listdir(img_dir)):
        if not img.lower().endswith((".jpg", ".png")):
            continue
        total += 1
        path = os.path.join(img_dir, img)
        if cv2.imread(path) is None:
            errors.append(img)  # ảnh không đọc được -> có thể bị corrupt

    if errors:
        print(f"[{split}] Tìm thấy {len(errors)}/{total} ảnh lỗi:")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"[{split}] Không có ảnh lỗi. ({total} ảnh OK)")
