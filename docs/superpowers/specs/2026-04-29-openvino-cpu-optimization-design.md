# Design Spec: CPU Performance Optimization via ONNX + OpenVINO

> **Dự án**: Truck Blind Spot Detection  
> **Ngày**: 2026-04-29  
> **Vấn đề**: app.py chỉ đạt 1–2 FPS trên CPU  
> **Mục tiêu**: Đạt ≥ 25 FPS trên Intel laptop CPU  

---

## 1. Phân tích Bottleneck

Theo benchmark trong `phase2.md`:

| Stage | Latency (CPU) | % tổng |
|-------|--------------|--------|
| YOLOv9 inference | 200–500 ms | ~95% |
| Tracking + prediction + visualization | < 5 ms | ~5% |

**Root cause**: PyTorch CPU inference không tận dụng tối ưu ISA của Intel (AVX-512, VNNI). OpenVINO được Intel tối ưu đặc biệt cho x86, có thể giảm inference time xuống 15–50 ms/frame (5–15×).

---

## 2. Giải Pháp: ONNX + OpenVINO Backend

### 2.1 Tổng quan kiến trúc thay đổi

```
tools/export_openvino.py     ← MỚI: script export PT → ONNX → OpenVINO IR
src/detector.py              ← SỬA: thêm OpenVINO backend
requirements.txt             ← SỬA: thêm openvino, nncf
─────────────────────────────────────────────────────────────
weights/best_roiv2.onnx                   (generated)
weights/best_roiv2_openvino/model.xml     (generated)
weights/best_roiv2_openvino/model.bin     (generated)
weights/best_roiv2_openvino_int8/         (generated, optional)
```

**Không thay đổi**: `src/pipeline.py`, `src/tracking/`, `src/visualize.py`, `src/roi.py`, `app.py` (chỉ thêm 1 CLI arg).

### 2.2 Luồng dữ liệu `predict()` sau thay đổi

```
frame (numpy BGR)
  → letterbox()                     [SHARED — Python/NumPy, giữ nguyên]
  → transpose + ascontiguousarray   [SHARED]
  ├── backend="pytorch"
  │     → torch.from_numpy → .to(device)
  │     → model.forward() → _unwrap_predictions()
  │     → torch.Tensor
  └── backend="openvino"
        → infer_request.infer({0: numpy_fp32})
        → numpy output → torch.from_numpy()
        → torch.Tensor
  → non_max_suppression()           [SHARED — giữ nguyên]
  → scale_boxes()                   [SHARED]
  → List[Detection]                 [SHARED]
```

---

## 3. Chi Tiết Thiết Kế

### 3.1 `src/detector.py`

**Thay đổi `__init__`**:
- Thêm param `backend: str = "pytorch"` — giá trị hợp lệ: `"pytorch"`, `"openvino"`
- Thêm param `openvino_model_dir: str = ""` — nếu rỗng, tự suy ra từ `weights_path` (thay `.pt` → `_openvino/`)
- Khi `backend="openvino"`: bỏ `select_device()`, bỏ `warmup()`, load OpenVINO compiled model

**Thêm method `_init_openvino(model_dir)`**:
```python
import openvino as ov
core = ov.Core()
compiled = core.compile_model(xml_path, "CPU")
self._ov_infer = compiled.create_infer_request()
self._ov_output_key = compiled.output(0)
```

**Thay đổi `predict()`**:
- Tách preprocess thành method riêng `_preprocess(frame) → numpy_fp32`
- Thêm dispatch block sau preprocess
- Postprocess NMS/scale_boxes không đổi

**Interface không thay đổi**: `predict(frame: np.ndarray) -> List[Detection]`

### 3.2 `tools/export_openvino.py`

Script chạy một lần:

```
python3 tools/export_openvino.py --weights weights/best_roiv2.pt
python3 tools/export_openvino.py --weights weights/best_roiv2.pt --int8 --calib-dir assets/videos/
```

**Step 1 — PT → ONNX**:
- Load model qua `DetectMultiBackend`
- `torch.onnx.export()` với dummy input shape `(1, 3, 640, 640)`, `opset_version=12`
- Output: `weights/best_6k.onnx`

**Step 2 — ONNX → OpenVINO IR**:
- `openvino.convert_model("best_6k.onnx")`
- `openvino.save_model(ov_model, "weights/best_6k_openvino/model.xml")`
- Output: `weights/best_6k_openvino/model.xml` + `model.bin`

**Step 3 (optional `--int8`) — INT8 Quantization**:
- Dùng NNCF: `nncf.quantize(ov_model, calibration_dataset)`
- `calibration_dataset`: ~100 frame từ `--calib-dir` (video hoặc thư mục ảnh)
- Output: `weights/best_6k_openvino_int8/model.xml`
- Mức giảm accuracy: ~1–3% mAP, tăng thêm ~1.5–2× tốc độ

### 3.3 `app.py` — thêm CLI argument

```bash
python3 app.py --backend openvino
python3 app.py --backend openvino --weights weights/best_roiv2.pt  # tự tìm best_roiv2_openvino/
```

Một arg `--backend` duy nhất, default `"pytorch"` để backward-compatible.

### 3.4 `src/pipeline.py` — pass-through

Thêm `backend: str = "pytorch"` vào `BlindSpotPipeline.__init__` và forward xuống `YOLOv9Detector`. Không thay đổi logic.

---

## 4. Dependencies Mới

```
# requirements.txt
openvino>=2024.0
nncf>=2.9          # optional, chỉ cần cho --int8
```

---

## 5. Lộ Trình Triển Khai

| Bước | Action | FPS dự kiến |
|------|--------|-------------|
| 0 | Baseline hiện tại (PyTorch CPU) | 1–2 FPS |
| 1 | Export FP32 + chạy `--backend openvino` | 10–20 FPS |
| 2 | Export INT8 + chạy với model int8 | 20–35 FPS |

Nếu sau bước 1 đã đạt ≥ 25 FPS: bỏ qua bước 2.

---

## 6. Các File Bị Ảnh Hưởng

| File | Thay đổi |
|------|---------|
| `src/detector.py` | Thêm backend param + OpenVINO inference path |
| `src/pipeline.py` | Thêm backend param, forward xuống detector |
| `app.py` | Thêm `--backend` CLI arg |
| `tools/export_openvino.py` | File mới |
| `requirements.txt` | Thêm openvino, nncf |

---

## 7. Giới Hạn Đã Biết

| # | Vấn đề | Giải pháp |
|---|--------|-----------|
| 1 | INT8 có thể giảm recall trên class nhỏ (bike, motor) | Chạy `roi_evaluation.py` sau quantize để kiểm tra |
| 2 | OpenVINO không hỗ trợ một số op của YOLOv9 | Fallback về FP32 OpenVINO; hoặc `--simplify` khi export ONNX |
| 3 | FPS thực tế phụ thuộc CPU cụ thể | Đo lại sau khi deploy |
