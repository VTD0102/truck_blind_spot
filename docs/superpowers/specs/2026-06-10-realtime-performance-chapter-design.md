# Design Spec: Chương "Tối ưu hiệu năng realtime" cho Báo cáo TTCS

> **Mục tiêu**: Chèn chương mới "VI. Tối ưu hiệu năng realtime" vào `Báo cáo TTCS _ N6 _ CK.docx`,
> đánh số lại chương "Định hướng phát triển" thành VII, và sửa 2 chỗ liên quan.
> **Phạm vi nội dung**: chỉ phần đã triển khai trong code (app-level Phase 3 + MPS + CoreML).
> OpenVINO/INT8 chỉ nhắc 1 câu ở phần nhận xét (đã thiết kế, chưa triển khai).
> **Số liệu FPS**: dùng placeholder `[ĐIỀN SỐ]` cho CPU/MPS; CoreML = 55 FPS (thực đo trên Mac M3).
> Người dùng sẽ điền số thực đo sau.

## 1. Nội dung chương VI (chèn sau chương V "Demo của dự án")

Văn phong: tiếng Việt học thuật, nhất quán với các chương hiện có (đoạn văn dẫn giải + bảng).
Độ dài mục tiêu: ~2.5–3 trang.

### VI.1. Yêu cầu realtime và phân tích bottleneck (Heading2)
- Ngưỡng realtime của hệ thống: ≥ 25 FPS (target mặc định trong `app.py`).
- Baseline: PyTorch CPU trên MacBook M3 chỉ đạt ~9 FPS → không đáp ứng realtime.
- Bảng phân tích thời gian từng bước pipeline (đo bằng `tools/benchmark.py`):
  | Bước | Thời gian/frame | Tỷ trọng |
  |---|---|---|
  | YOLOv9 inference | ~110 ms | ~95% |
  | Tracking + Prediction + Visualization | < 5 ms | ~5% |
- Kết luận: bottleneck nằm gần như hoàn toàn ở bước inference → hai hướng tối ưu:
  (1) giảm tải mức ứng dụng, (2) tăng tốc inference bằng phần cứng chuyên dụng.

### VI.2. Tối ưu mức ứng dụng (Heading2)
- **Frame skipping (Kalman predict-only)**: khi FPS đo được < 70% target, hệ thống bỏ qua
  bước detection ở frame kế tiếp; track vẫn được duy trì nhờ bước predict của Kalman Filter.
  Cờ `--enable-frame-skip` trong `app.py`.
- **Adaptive quality scaling**: khi FPS < 80% target → tự giảm resolution đầu vào;
  khi FPS > 110% target → khôi phục dần. Cờ `--enable-adaptive-scale`.
- Performance overlay: hiển thị FPS làm mượt, timing breakdown từng stage.
- Nhận xét: hai kỹ thuật này giúp duy trì realtime trên phần cứng yếu nhưng đánh đổi
  độ chính xác (bỏ frame detection / giảm resolution) → cần tăng tốc inference gốc.

### VI.3. Tăng tốc inference trên Apple Silicon (Heading2)
- **Giai đoạn 1 — MPS backend** (`--device mps`): chạy model trên GPU M3 qua Metal
  Performance Shaders; bypass `select_device` của YOLOv9 (chỉ biết CUDA/CPU);
  NMS có op không hỗ trợ trên MPS nên chuyển tensor về CPU trước khi NMS; giữ FP32.
- **Giai đoạn 2 — CoreML backend** (`--backend coreml`): export model theo đường
  TorchScript → CoreML bằng `tools/export_coreml.py`, sinh `weights/best_roiv2.mlpackage`;
  `YOLOv9Detector` bỏ qua `DetectMultiBackend`, gọi thẳng CoreML runtime → model chạy
  trên Apple Neural Engine (ANE), CoreML tự quản lý precision.
- Ràng buộc: `--backend coreml` và `--device mps` loại trừ nhau (parser error trong app.py).

### VI.4. Kết quả benchmark (Heading2)
- Bảng kết quả trên MacBook M3 16GB (điều kiện đo: `[ĐIỀN ĐIỀU KIỆN: video, resolution,
  cờ tối ưu]`):
  | Cấu hình | Backend | FPS thực đo | Đạt realtime (≥25)? |
  |---|---|---|---|
  | CPU (baseline) | pytorch | `[ĐIỀN SỐ]` (~9) | Không |
  | GPU M3 | pytorch + MPS | `[ĐIỀN SỐ]` | `[ĐIỀN]` |
  | Neural Engine | coreml | 55 | Có (≈2.2× ngưỡng) |
- Nhận xét: CoreML đạt 55 FPS, vượt ngưỡng realtime ~2.2 lần, còn dư địa cho
  đa camera hoặc model lớn hơn.

### VI.5. Nhận xét và đánh đổi (Heading2)
- CoreML/ANE chỉ khả dụng trên macOS/Apple Silicon → không portable.
- Hướng tương đương cho CPU Intel: OpenVINO + INT8 quantization (đã thiết kế trong
  docs nội bộ, chưa triển khai) — 1 câu, nối sang chương Định hướng.
- Tối ưu app-level (frame skip, adaptive scale) vẫn giữ vai trò lưới an toàn khi
  phần cứng không đủ mạnh.

## 2. Chỉnh sửa kèm theo (3 chỗ)

1. Heading1 "VI. Định hướng phát triển của dự án" → "VII. Định hướng phát triển của dự án".
   (5 heading con không mang số La Mã → không cần sửa.)
2. Bullet "Bổ sung benchmark tốc độ xử lý (FPS, latency end-to-end, GPU/CPU usage)..."
   trong mục Định hướng §3 → viết lại: benchmark trên Apple Silicon đã trình bày ở
   chương VI; định hướng còn lại là mở rộng sang laptop Intel và embedded board.
3. Cuối mục V.5 (Kết quả demo) thêm câu dẫn: "Hiệu năng xử lý chi tiết và các kỹ thuật
   tối ưu realtime được trình bày ở chương VI."

## 3. Phương án kỹ thuật

- **python-docx 1.2.0** (đã có sẵn): backup `Báo cáo TTCS _ N6 _ CK.docx` →
  `Báo cáo TTCS _ N6 _ CK.backup.docx` trước khi ghi.
- Chèn paragraph với style `Heading1`/`Heading2` sẵn có; bảng dùng style giống
  các bảng hiện hữu trong file (clone style từ bảng có sẵn).
- Vị trí chèn: ngay trước paragraph heading "VI. Định hướng phát triển của dự án"
  (python-docx: tạo element rồi `addprevious()` vào thân document).
- TOC là field → người dùng mở Word, chọn Update Field (F9) để mục lục cập nhật.
- Điều kiện tiên quyết: **đóng Word** trước khi chạy script (file trong OneDrive).

## 4. Kiểm chứng

1. Đếm paragraph trước/sau: số paragraph sau = trước + số đoạn chèn (không mất đoạn gốc).
2. Trích xuất lại text: thứ tự Heading1 đúng I → II → III → IV → V → VI (mới) → VII.
3. Xác nhận 3 chỗ chỉnh sửa có mặt trong text trích xuất.
4. File mở được (zip hợp lệ, `python-docx` đọc lại không lỗi).

## Quyết định đã chốt với người dùng

- Phạm vi: chỉ phần đã làm + đo được (option 1).
- Số liệu: placeholder `[ĐIỀN SỐ]`, người dùng điền sau (CoreML = 55 FPS giữ nguyên).
- Định dạng giao: chèn thẳng vào file docx báo cáo.
- Không đọc/đụng `yolov9/`; không sửa nội dung chương II (Dataset & Training).
