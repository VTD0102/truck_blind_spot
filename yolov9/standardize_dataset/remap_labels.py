import os   # thư viện làm việc với file/thư mục

# Mapping từ dataset cũ -> project mới
# old_id: new_id
ID_MAP = {
    0: 2,  # bus -> car
    1: 2,  # car -> car
    2: 3,  # motorcycle -> motorcycle
    3: 2   # truck -> car
}

label_dirs = [
    "yolov9/data/datasets/train/labels",  # thư mục train
    "yolov9/data/datasets/valid/labels",  # thư mục validation
    "yolov9/data/datasets/test/labels"    # thư mục test
]

for label_dir in label_dirs:   # duyệt từng thư mục label
    if not os.path.exists(label_dir):   # kiểm tra thư mục có tồn tại không
        print(f"Không tìm thấy thư mục: {label_dir}")
        continue

    for file_name in os.listdir(label_dir):   # duyệt từng file
        if not file_name.endswith(".txt"):    # bỏ qua file không phải txt
            continue

        file_path = os.path.join(label_dir, file_name)  # tạo đường dẫn file

        new_lines = []   # lưu nội dung mới
        changed = False  # đánh dấu file có bị thay đổi không

        with open(file_path, "r") as f:
            lines = f.readlines()   # đọc toàn bộ file

        for line in lines:   # duyệt từng dòng
            parts = line.strip().split()   # tách dữ liệu

            if len(parts) != 5:   # kiểm tra format YOLO (phải có 5 phần)
                print(f"Lỗi format ở {file_path}: {line.strip()}")
                continue

            old_id = int(parts[0])   # lấy class id cũ

            if old_id not in ID_MAP:   # nếu không có trong mapping
                print(f"ID không có trong mapping ở {file_path}: {old_id}")
                continue

            new_id = ID_MAP[old_id]   # lấy class id mới
            parts[0] = str(new_id)   # thay id cũ bằng id mới
            new_lines.append(" ".join(parts))   # ghép lại dòng
            changed = True   # đánh dấu đã thay đổi

        if changed:   # nếu có thay đổi
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")   # ghi đè file

print("Remap hoàn tất.")   # thông báo hoàn thành