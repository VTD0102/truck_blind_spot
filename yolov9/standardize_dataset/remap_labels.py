import os

# ============================================================
# remap_labels.py
# Remap class ID từ BDD100K gốc sang blindspot.yaml mới.
# Chạy SAU filter_labels.py (các class không dùng đã bị xóa).
#
# BDD100K gốc (sau khi lọc chỉ còn):
#   0: bike  | 1: bus  | 2: car  | 3: motor  | 4: person  | 9: truck
#
# blindspot.yaml mới:
#   0: person | 1: bike | 2: motor | 3: car | 4: truck | 5: bus
# ============================================================

# Mapping: old_id (BDD100K) -> new_id (blindspot.yaml)
ID_MAP = {
    0: 1,   # bike   -> bike  (id 1)
    1: 5,   # bus    -> bus   (id 5)
    2: 3,   # car    -> car   (id 3)
    3: 2,   # motor  -> motor (id 2)
    4: 0,   # person -> person(id 0)
    9: 4,   # truck  -> truck (id 4)
}

# Tên class để log cho rõ
ID_NAMES_OLD = {0: "bike", 1: "bus", 2: "car", 3: "motor", 4: "person", 9: "truck"}
ID_NAMES_NEW = {0: "person", 1: "bike", 2: "motor", 3: "car", 4: "truck", 5: "bus"}

# Thư mục labels của từng split (chạy từ thư mục gốc dự án)
LABEL_DIRS = [
    "yolov9/data/datasets/train/labels",
    "yolov9/data/datasets/valid/labels",
    "yolov9/data/datasets/test/labels",
]

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):   # kiểm tra thư mục có tồn tại không
        print(f"[SKIP] Không tìm thấy: {label_dir}")
        continue

    files_changed = 0   # số file đã remap
    error_count = 0     # số dòng lỗi format hoặc ID không hợp lệ

    for file_name in os.listdir(label_dir):   # duyệt từng file
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(label_dir, file_name)

        with open(file_path, "r") as f:
            lines = f.readlines()   # đọc toàn bộ file

        new_lines = []
        changed = False

        for line in lines:
            parts = line.strip().split()   # tách các trường

            if len(parts) != 5:   # format YOLO phải có đúng 5 trường
                print(f"[WARN] Lỗi format: {file_path} | '{line.strip()}'")
                error_count += 1
                continue

            old_id = int(parts[0])   # lấy class id gốc

            if old_id not in ID_MAP:   # ID không nằm trong mapping (chưa lọc?)
                print(f"[WARN] ID không có trong mapping: {file_path} | id={old_id}")
                error_count += 1
                continue

            new_id = ID_MAP[old_id]
            parts[0] = str(new_id)          # thay id cũ bằng id mới
            new_lines.append(" ".join(parts) + "\n")
            changed = True

        if changed:
            files_changed += 1
            with open(file_path, "w") as f:
                f.writelines(new_lines)   # ghi đè file với id đã remap

    split_name = label_dir.split("/")[-2]   # train / valid / test
    status = f"{files_changed:5d} files đã remap"
    if error_count:
        status += f" | {error_count} dòng lỗi/bỏ qua"
    print(f"[{split_name:5s}] {status}")

print("\nRemap hoàn tất.")
