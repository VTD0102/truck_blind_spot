from pathlib import Path
from ultralytics import YOLO

# ===== CẤU HÌNH =====
MODEL_NAME = "yolov8x.pt"   # chính xác cao, hơi chậm
CONF_THRES = 0.25
IMG_SIZE = 640

# Chỉ thêm 2 class 
AUTO_CLASSES = {
    "person": 0,
    "bicycle": 1,
}

# Ngưỡng để coi là trùng bbox cũ
IOU_THRESHOLD = 0.5

SPLITS = ["train", "valid", "test"]

# ===== PATH =====
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

DATASET_ROOT = PROJECT_ROOT / "data" / "datasets"


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x_center, y_center, w, h


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def load_existing_labels(label_path, img_w, img_h):
    existing_lines = []
    existing_boxes = []

    if not label_path.exists():
        return existing_lines, existing_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            existing_lines.append(line.strip())
            existing_boxes.append((cls_id, yolo_to_xyxy(x, y, w, h, img_w, img_h)))

    return existing_lines, existing_boxes


def is_duplicate(new_box, existing_boxes, target_cls):
    for cls_id, old_box in existing_boxes:
        if cls_id != target_cls:
            continue
        if compute_iou(new_box, old_box) >= IOU_THRESHOLD:
            return True
    return False


def process_split(model, split):
    img_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"

    if not img_dir.exists():
        print(f"[WARN] Không tìm thấy thư mục ảnh: {img_dir}")
        return

    if not label_dir.exists():
        print(f"[WARN] Không tìm thấy thư mục label: {label_dir}")
        return

    image_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    print(f"\n=== Split: {split} | {len(image_files)} ảnh ===")

    added_total = 0
    changed_files = 0

    for idx, img_path in enumerate(image_files, 1):
        label_path = label_dir / f"{img_path.stem}.txt"

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            imgsz=IMG_SIZE,
            verbose=False
        )

        if len(results) == 0:
            continue

        r = results[0]
        img_h, img_w = r.orig_shape

        existing_lines, existing_boxes = load_existing_labels(label_path, img_w, img_h)
        new_lines = list(existing_lines)
        added_in_this_file = 0

        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                coco_cls_id = int(box.cls[0].item())
                cls_name = model.names[coco_cls_id]

                if cls_name not in AUTO_CLASSES:
                    continue

                project_cls = AUTO_CLASSES[cls_name]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                new_box = [x1, y1, x2, y2]

                if is_duplicate(new_box, existing_boxes, project_cls):
                    continue

                x, y, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

                if w <= 0 or h <= 0:
                    continue

                new_line = f"{project_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                new_lines.append(new_line)
                existing_boxes.append((project_cls, new_box))
                added_in_this_file += 1
                added_total += 1

        if added_in_this_file > 0:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
            changed_files += 1

        if idx % 100 == 0 or idx == len(image_files):
            print(f"[{split}] {idx}/{len(image_files)} ảnh")

    print(f"[DONE] {split}: thêm {added_total} bbox vào {changed_files} file label")


def main():
    print("Đang load model pretrained...")
    model = YOLO(MODEL_NAME)

    for split in SPLITS:
        process_split(model, split)

    print("\nHoàn tất.")
    print("Đã giữ nguyên label cũ và chỉ thêm person/bicycle nếu model phát hiện được.")


if __name__ == "__main__":
    main()