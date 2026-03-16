# Training YOLOv9 Report

## 1. Mục tiêu huấn luyện
- Thiết lập môi trường huấn luyện YOLOv9 trên hạ tầng GPU hiệu năng cao.
- Huấn luyện mô hình nhận diện **điểm mù xe tải** dựa trên dataset.
- Thực hiện thực nghiệm trên nhiều cấu hình (model size) để tối ưu hóa giữa **độ chính xác (mAP)** và **tốc độ (FPS)**.

---

## 2. Môi trường huấn luyện (Environment)

- **Hardware:** Google Colab Pro – GPU NVIDIA H100 (80GB VRAM)  
- **Framework:** PyTorch 2.6+, CUDA 12.x   
- **Công cụ theo dõi:** TensorBoard / CSV Logger  

---

## 3. Cấu hình huấn luyện (Training Config)


- **Data config:** `data/blindspot.yaml`  
- **Hyperparameters:** `data/hyps/hyp.blindspot.yaml`  
- **Image size:** `640`  
- **Epochs:** `100`  
- **Optimizer:** `SGD / AdamW (Auto-tuning)`  


---

# 4. Kết quả huấn luyện (Training Results)

Dưới đây là bảng so sánh hiệu năng giữa hai phiên bản **YOLOv9-Tiny (t)** và **YOLOv9-Small (s)**.

| Cấu hình | mAP@0.5 | Precision | Recall | Kích thước file | Thời gian huấn luyện |
|----------|--------|-----------|--------|------------------|----------------------|
| YOLOv9-Tiny | 0.69 | 0.69 | 0.68 | ~15 MB | 0.275 hours |
| YOLOv9-Small | 0.7 | 0.7 | 0.68 | 20.3 MB | 0.329 hours |


---

# 5. Kết quả bàn giao (Deliverables)

## Model Weights
- `best_tiny.pt`
- `best_small.pt`  
*(Lưu tại thư mục `training_results/`)*

## Training Logs
- `results_tiny.csv`
- `results_small.csv`

## Đánh giá
Phiên bản **YOLOv9-Small** đạt **độ chính xác cao hơn**, do đó **phù hợp hơn cho việc triển khai thực tế**.

---

# 6. Training Version History

| Version | Mô tả |
|--------|------|
| **v1_test_env** | Thiết lập môi trường và kiểm tra dataset |
| **v2_train_tiny** | Huấn luyện 100 epochs phiên bản Tiny (tốc độ cao) |
| **v3_train_small** | Huấn luyện 100 epochs phiên bản Small (độ chính xác cao) |

---