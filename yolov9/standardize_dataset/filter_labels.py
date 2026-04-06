import os

# ============================================================
# filter_labels.py
# Xóa các labels của class không sử dụng trong dự án.
#
# Dataset gốc BDD100K có 10 class:
#   0: bike, 1: bus, 2: car, 3: motor, 4: person,
#   5: rider, 6: traffic light, 7: traffic sign, 8: train, 9: truck
#
# Dự án chỉ dùng 6 class:
#   person, bike, motor, car, truck, bus (sẽ remap ở bước sau)
# => Xóa: rider(5), traffic light(6), traffic sign(7), train(8)
# ============================================================

# IDs cần XÓA (từ BDD100K gốc)
REMOVE_IDS = {5, 6, 7, 8}  # rider, traffic light, traffic sign, train

# Tên class để log cho rõ
REMOVE_NAMES = {
    5: "rider",
    6: "traffic light",
    7: "traffic sign",
    8: "train",
}

# Thư mục labels của từng split (chạy từ thư mục gốc dự án)
LABEL_DIRS = [
    "yolov9/data/datasets/train/labels",
    "yolov9/data/datasets/valid/labels",
    "yolov9/data/datasets/test/labels",
]

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):
        print(f"[SKIP] Không tìm thấy: {label_dir}")
        continue

    removed_total = 0   # tổng số dòng đã xóa trong split này
    files_changed = 0   # số file bị thay đổi

    for file_name in os.listdir(label_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(label_dir, file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        removed = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            cls_id = int(parts[0])
            if cls_id in REMOVE_IDS:
                removed += 1    # bỏ dòng này (xóa label không dùng)
            else:
                new_lines.append(line)  # giữ lại

        if removed > 0:
            removed_total += removed
            files_changed += 1
            with open(file_path, "w") as f:
                f.writelines(new_lines)

    split_name = label_dir.split("/")[-2]   # train / valid / test
    print(f"[{split_name:5s}] {files_changed:5d} files thay đổi | "
          f"{removed_total:6d} labels đã xóa "
          f"(rider, traffic light, traffic sign, train)")

print("\nLọc labels hoàn tất.")
