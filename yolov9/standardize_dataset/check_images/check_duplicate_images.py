import os
import hashlib

# ============================================================
# check_duplicate_images.py
# Tìm ảnh trùng lặp (cùng nội dung) trong từng split
# bằng cách so sánh MD5 hash.
# Chạy từ thư mục gốc dự án.
# ============================================================

def file_hash(path):
    """Tính MD5 hash của file để phát hiện nội dung trùng."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

SPLITS = ["train", "valid", "test"]

for split in SPLITS:
    img_dir = f"yolov9/data/datasets/{split}/images"
    if not os.path.exists(img_dir):
        print(f"[{split}] Không tìm thấy thư mục: {img_dir}")
        continue

    hashes = {}         # hash -> tên file đầu tiên thấy
    duplicates = []     # danh sách cặp ảnh trùng

    for img in sorted(os.listdir(img_dir)):
        if not img.lower().endswith((".jpg", ".png")):
            continue
        path = os.path.join(img_dir, img)
        h = file_hash(path)
        if h in hashes:
            duplicates.append((img, hashes[h]))     # ảnh trùng với file gốc
        else:
            hashes[h] = img

    if duplicates:
        print(f"[{split}] Tìm thấy {len(duplicates)} cặp ảnh trùng:")
        for dup, orig in duplicates:
            print(f"  {dup}  <->  {orig}")
    else:
        print(f"[{split}] Không có ảnh trùng. ({len(hashes)} ảnh)")
