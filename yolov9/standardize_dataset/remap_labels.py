import os

# Mapping từ dataset cũ -> project mới
# old_id: new_id
ID_MAP = {
    0: 2,  # bus -> car
    1: 2,  # car -> car
    2: 3,  # motorcycle -> motorcycle
    3: 2   # truck -> car
}

label_dirs = [
    "yolov9/data/datasets/train/labels",
    "yolov9/data/datasets/valid/labels",
    "yolov9/data/datasets/test/labels"
]

for label_dir in label_dirs:
    if not os.path.exists(label_dir):
        print(f"Không tìm thấy thư mục: {label_dir}")
        continue

    for file_name in os.listdir(label_dir):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(label_dir, file_name)

        new_lines = []
        changed = False

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"Lỗi format ở {file_path}: {line.strip()}")
                continue

            old_id = int(parts[0])

            if old_id not in ID_MAP:
                print(f"ID không có trong mapping ở {file_path}: {old_id}")
                continue

            new_id = ID_MAP[old_id]
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts))
            changed = True

        if changed:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

print("Remap hoàn tất.")