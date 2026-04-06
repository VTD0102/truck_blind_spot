from collections import Counter
import os

# ============================================================
# analyze_dataset.py
# Đếm số lượng labels theo từng class trong train và val.
# Hiển thị: id, tên class, số lượng labels, tỉ lệ phần trăm.
# (test không cần vì labels đang rỗng)
# ============================================================

# Tên class theo blindspot.yaml
CLASS_NAMES = {
    0: "person",
    1: "bike",
    2: "motor",
    3: "car",
    4: "truck",
    5: "bus",
}

# Các split cần phân tích (bỏ test vì labels rỗng)
SPLITS = {
    "train": "yolov9/data/datasets/train/labels",
    "val":   "yolov9/data/datasets/valid/labels",
}

for split_name, label_dir in SPLITS.items():
    if not os.path.exists(label_dir):
        print(f"[{split_name}] Không tìm thấy thư mục: {label_dir}")
        continue

    counter = Counter()     # đếm số labels theo class id
    file_count = 0          # số file label đọc được

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        file_count += 1
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    counter[int(parts[0])] += 1

    total = sum(counter.values())   # tổng số labels

    print(f"\n[{split_name}] {file_count} ảnh | {total} labels tổng cộng:")
    print(f"  {'ID':<4} {'Class':<14} {'Labels':>8}  {'Tỉ lệ':>7}")
    print(f"  {'-'*38}")
    for cls_id in sorted(counter.keys()):
        name  = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        count = counter[cls_id]
        pct   = 100 * count / total if total > 0 else 0
        print(f"  {cls_id:<4} {name:<14} {count:>8}  {pct:>6.1f}%")
    print(f"  {'-'*38}")
    print(f"  {'Tổng':<18} {total:>8}  {'100.0%':>7}")

print("\nPhân tích hoàn tất.")
