# Design Spec: Apple Silicon GPU Acceleration (MPS + CoreML)

> **Dự án**: Truck Blind Spot Detection
> **Ngày**: 2026-06-10
> **Vấn đề**: app.py chỉ đạt ~9 FPS trên CPU (M3 Mac không có CUDA)
> **Mục tiêu**: Đạt ≥ 28 FPS trên macOS M3 16GB RAM
> **Phạm vi**: Chỉ macOS Apple Silicon — không yêu cầu cross-platform

---

## 1. Phân Tích Bottleneck

| Stage | Latency (CPU) | % tổng |
|-------|--------------|--------|
| YOLOv9 inference | ~110 ms | ~95% |
| Tracking + prediction + visualization | < 5 ms | ~5% |

**Root cause**: `select_device("")` của YOLOv9 vendored fallback về CPU trên macOS (không có CUDA). M3 GPU (18-core) và Neural Engine (16-core, ~18 TOPS) hoàn toàn không được dùng.

**Tại sao không thể chỉ truyền `--device mps` ngay bây giờ**: `select_device()` trong `yolov9/utils/torch_utils.py` không xử lý chuỗi `"mps"` → crash hoặc fallback về CPU silent.

---

## 2. Giải Pháp: Hai Phase Leo Thang

### Phase 1 — PyTorch MPS (nhanh, ít rủi ro)

Bypass `select_device()` trong `src/detector.py` khi `device == "mps"`, dùng `torch.device("mps")` trực tiếp.

**Dự kiến**: 25–45 FPS. Nếu đạt ≥ 28 FPS → dừng, không cần Phase 2.

### Phase 2 — CoreML (nếu Phase 1 chưa đủ)

Export `.pt` → ONNX → `.mlpackage`, dùng `coremltools` với `compute_units=ALL` để tận dụng Neural Engine + GPU + CPU.

**Dự kiến**: 40–80 FPS.

---

## 3. Kiến Trúc Thay Đổi

```
tools/export_coreml.py          ← MỚI (Phase 2): script export PT → ONNX → mlpackage
src/detector.py                 ← SỬA (Phase 1 + 2): thêm MPS bypass + CoreML backend
src/pipeline.py                 ← SỬA (Phase 2 only): forward `backend` param
app.py                          ← SỬA (Phase 1 + 2): cải thiện help text + thêm --backend
requirements.txt                ← SỬA (Phase 2 only): thêm coremltools, onnx
docs/run-commands.md            ← MỚI (sau cùng): tài liệu tất cả câu lệnh chạy
─────────────────────────────────────────────────────────────────────────────────
weights/best_roiv2.onnx                  (generated, Phase 2)
weights/best_roiv2.mlpackage/            (generated, Phase 2)
```

**Không thay đổi**: `src/tracking/`, `src/roi.py`, `src/visualize.py`, `yolov9/` (RULES §5).

**Interface bất biến**: `YOLOv9Detector.predict(frame) -> List[Detection]` — pipeline xuôi không đổi.

---

## 4. Chi Tiết Phase 1: MPS Patch

### 4.1 `src/detector.py` — `__init__`

Thêm param `backend: str = "pytorch"`. Tách logic device initialization:

```python
if backend == "pytorch" and device == "mps":
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS không khả dụng trên máy này. Chạy lại với --device cpu."
        )
    self.device = torch.device("mps")
elif backend == "coreml":
    self.device = torch.device("cpu")  # CoreML tự quản lý device
else:
    self.device = select_device(device)  # Giữ nguyên cho cpu / cuda

# FP16: chỉ bật cho CUDA — MPS FP32 đã đủ nhanh, FP16 trên MPS còn quirks
use_fp16 = half or (self.device.type == "cuda")
```

### 4.2 `src/detector.py` — `predict()`

`non_max_suppression()` của YOLOv9 có một số op không chạy được trên MPS tensor. Chuyển predictions về CPU trước NMS:

```python
predictions = self._unwrap_predictions(predictions)
if self.device.type == "mps":
    predictions = predictions.cpu()
# non_max_suppression() và scale_boxes() giữ nguyên
```

### 4.3 `app.py`

Cập nhật help text `--device` để người dùng biết `mps` là giá trị hợp lệ. Không thêm logic mới.

### Chạy Phase 1

```bash
python3 app.py --device mps
python3 app.py --device mps --source assets/videos/demo4.mp4
```

---

## 5. Chi Tiết Phase 2: CoreML Backend

### 5.1 `tools/export_coreml.py` (file mới)

Script chạy một lần. Hai bước:

**Bước 1 — PT → ONNX**:
- Load model qua `DetectMultiBackend` (CPU)
- `torch.onnx.export()` với dummy input `(1, 3, 640, 640)`, `opset_version=12`
- Output: `weights/best_roiv2.onnx`

**Bước 2 — ONNX → CoreML**:
- `coremltools.convert(onnx_model, compute_units=ct.ComputeUnit.ALL)`
- `ALL` → CoreML scheduler tự chọn Neural Engine / GPU / CPU theo từng op
- Output: `weights/best_roiv2.mlpackage`

```bash
python3 tools/export_coreml.py --weights weights/best_roiv2.pt
# Optional: chỉ định output dir
python3 tools/export_coreml.py --weights weights/best_roiv2.pt --output-dir weights/
```

### 5.2 `src/detector.py` — CoreML backend

Thêm method `_init_coreml(mlpackage_path)`:

```python
import coremltools as ct
self._coreml_model = ct.models.MLModel(mlpackage_path)
```

Tách preprocess thành shared private method `_preprocess(frame) -> np.ndarray` (letterbox → transpose → /255, float32, shape `(1,3,640,640)`). Cả PyTorch và CoreML path dùng chung method này.

Dispatch trong `predict()`:

```python
if self.backend == "coreml":
    input_arr = self._preprocess(frame)  # numpy (1, 3, 640, 640) float32
    out = self._coreml_model.predict({"image": input_arr})
    # Output key được xác định lúc export và lưu vào self._coreml_output_key
    predictions = torch.from_numpy(out[self._coreml_output_key])  # CPU tensor
else:
    # PyTorch path hiện tại giữ nguyên (dùng _preprocess nội bộ)
```

`_coreml_output_key` được detect tự động trong `_init_coreml()`:
```python
spec = self._coreml_model.get_spec()
self._coreml_output_key = spec.description.output[0].name
```

NMS + scale_boxes không thay đổi — cùng nhận CPU tensor.

**Error handling**: nếu `.mlpackage` chưa tồn tại → raise `FileNotFoundError` với thông báo hướng dẫn chạy `export_coreml.py`.

### 5.3 `src/pipeline.py`

Thêm `backend: str = "pytorch"` vào `BlindSpotPipeline.__init__`, forward xuống `YOLOv9Detector`. Không thay đổi logic pipeline.

### 5.4 `app.py`

Thêm `--backend` argument độc lập (không mutually exclusive với `--device` ở tầng argparse vì `--device` còn dùng cho `cpu`/`cuda`). Validation xung đột được xử lý trong code sau khi parse:

```python
parser.add_argument("--backend", choices=["pytorch", "coreml"], default="pytorch")

# Sau parse_args():
if args.backend == "coreml" and args.device == "mps":
    parser.error("--backend coreml và --device mps không dùng cùng nhau.")
```

Khi `--backend coreml`: `device` bị ignore (CoreML tự quản lý compute unit).

### Chạy Phase 2

```bash
# Bước 1: export (một lần)
python3 tools/export_coreml.py --weights weights/best_roiv2.pt

# Bước 2: chạy
python3 app.py --backend coreml
```

---

## 6. Dependencies Mới (Phase 2)

```
# requirements.txt
coremltools>=7.0    # export + load .mlpackage
onnx>=1.14.0        # intermediate format khi export
```

Phase 1 không thêm dependency — PyTorch MPS đã có sẵn trong `torch>=2.0.0`.

---

## 7. Error Handling

| Tình huống | Hành vi |
|---|---|
| `--device mps` nhưng MPS không available | `RuntimeError` tiếng Việt + gợi ý `--device cpu` |
| `--backend coreml` nhưng `.mlpackage` chưa export | `FileNotFoundError` + hướng dẫn chạy `export_coreml.py` |
| `--device mps` và `--backend coreml` cùng lúc | `argparse` mutually exclusive group báo lỗi ngay |
| MPS op không support (runtime) | PyTorch raise lỗi rõ ràng — không silent fallback |

---

## 8. FPS Dự Kiến

| Mode | Flag | Dự kiến FPS | Target |
|------|------|-------------|--------|
| CPU hiện tại | (default) | ~9 FPS | — |
| Phase 1: MPS | `--device mps` | 25–45 FPS | ≥ 28 FPS ✓ |
| Phase 2: CoreML | `--backend coreml` | 40–80 FPS | ≥ 28 FPS ✓✓ |

---

## 9. File Bị Ảnh Hưởng

| File | Phase 1 | Phase 2 |
|------|---------|---------|
| `src/detector.py` | MPS device bypass + NMS CPU fix | CoreML init + inference dispatch |
| `src/pipeline.py` | Không đổi | Thêm `backend` param |
| `app.py` | Cải thiện `--device` help text | Thêm `--backend` mutually exclusive |
| `tools/export_coreml.py` | — | File mới |
| `requirements.txt` | Không đổi | Thêm `coremltools`, `onnx` |
| `docs/run-commands.md` | — | File mới (deliverable cuối) |

---

## 10. Giới Hạn Đã Biết

| # | Vấn đề | Giải pháp |
|---|--------|-----------|
| 1 | Một số op YOLOv9 chưa implement đầy đủ trên MPS | Predictions về CPU trước NMS (§4.2) |
| 2 | CoreML op compatibility với YOLOv9 head chưa được test | Cần chạy `export_coreml.py` và kiểm tra output accuracy |
| 3 | FP16 trên MPS còn quirks (PyTorch < 2.2) | Dùng FP32 cho MPS — tốc độ MPS FP32 đã đủ |
| 4 | Phase 2 không áp dụng được cho máy non-Apple | Phase 2 chỉ là optional escalation cho M3 Mac |
