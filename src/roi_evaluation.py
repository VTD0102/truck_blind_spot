"""
ROI-Aware Recall Evaluation for Truck Blind Spot Detection.

Evaluates YOLOv9 detection recall broken down by ROI zone and object class,
providing granular safety metrics beyond global mAP/recall figures.

For each ground-truth object the zone is determined using the same
bbox_bottom_center anchor and MultiPolygonROI priority logic used at
inference time.  Predictions are matched to GT objects via greedy IoU
matching (sorted by confidence desc), consistent with standard object
detection evaluation.

Usage:
    python -m src.roi_evaluation \\
        --weights weights/best_6k.pt \\
        --data configs/blindspot.yaml \\
        --roi configs/roi.json \\
        --roi-profile front_camera \\
        --split val \\
        --conf-thres 0.25 \\
        --iou-match 0.5 \\
        --output-dir outputs/roi_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .detector import Detection, YOLOv9Detector
    from .roi import MultiPolygonROI
except ImportError:
    from detector import Detection, YOLOv9Detector  # type: ignore
    from roi import MultiPolygonROI  # type: ignore

logger = logging.getLogger(__name__)

# Sentinel value for objects that fall outside every ROI zone.
OUTSIDE_ROI = "outside_roi"

# ─── Data structures ──────────────────────────────────────────────────────────


@dataclass
class GTObject:
    """Single ground-truth annotation after zone assignment."""

    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]       # xyxy pixel coordinates
    anchor_point: Tuple[int, int]          # bbox_bottom_center
    zone_name: str                          # ROI zone name or OUTSIDE_ROI
    is_matched: bool = False               # set True when a prediction matches


@dataclass
class ZoneMetrics:
    """Cumulative TP / GT counters for one ROI zone."""

    zone_name: str
    gt: int = 0
    tp: int = 0

    @property
    def fn(self) -> int:
        return self.gt - self.tp

    @property
    def recall(self) -> float:
        return self.tp / self.gt if self.gt > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "zone": self.zone_name,
            "gt": self.gt,
            "tp": self.tp,
            "fn": self.fn,
            "recall": round(self.recall, 4),
        }


@dataclass
class ZoneClassMetrics:
    """Cumulative TP / GT counters for one (zone, class) pair."""

    zone_name: str
    class_name: str
    gt: int = 0
    tp: int = 0

    @property
    def fn(self) -> int:
        return self.gt - self.tp

    @property
    def recall(self) -> float:
        return self.tp / self.gt if self.gt > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "zone": self.zone_name,
            "class": self.class_name,
            "gt": self.gt,
            "tp": self.tp,
            "fn": self.fn,
            "recall": round(self.recall, 4),
        }


# ─── Core evaluator ───────────────────────────────────────────────────────────


class ROIEvaluator:
    """
    Runs inference on a dataset and accumulates per-zone, per-class recall.

    Parameters
    ----------
    detector : YOLOv9Detector
        Loaded detector instance ready for inference.
    roi : MultiPolygonROI
        ROI profile to use for zone assignment of GT objects.
    class_names : Dict[int, str]
        Mapping from class index to class name string.
    iou_match_threshold : float
        Minimum IoU for a prediction to be considered a TP match.
    """

    def __init__(
        self,
        detector: YOLOv9Detector,
        roi: MultiPolygonROI,
        class_names: Dict[int, str],
        iou_match_threshold: float = 0.5,
    ) -> None:
        self.detector = detector
        self.roi = roi
        self.class_names = class_names
        self.iou_match_threshold = iou_match_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_dataset(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
    ) -> Tuple[Dict, List[Dict], List[Dict]]:
        """
        Iterate over all images in the dataset and aggregate metrics.

        Returns
        -------
        overall : Dict
            High-level recall numbers across all ROI zones combined.
        zone_list : List[Dict]
            Per-zone metrics rows (zone, gt, tp, fn, recall).
        zone_class_list : List[Dict]
            Per-(zone, class) metrics rows sorted by (zone, class).
        """
        # Initialise zone metrics from the loaded ROI profile so that zones
        # with zero GT objects still appear in the output table.
        zone_metrics: Dict[str, ZoneMetrics] = {
            zone.name: ZoneMetrics(zone_name=zone.name)
            for zone in self.roi.zones
        }
        zone_class_metrics: Dict[Tuple[str, str], ZoneClassMetrics] = {}

        total_gt = 0
        total_gt_in_roi = 0
        total_tp_in_roi = 0
        n_images = len(image_paths)

        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            if (i + 1) % 50 == 0 or (i + 1) == n_images:
                logger.info("  [%d / %d] %s", i + 1, n_images, img_path.name)

            gt_objects = self._process_image(img_path, lbl_path)
            total_gt += len(gt_objects)

            for gt in gt_objects:
                if gt.zone_name == OUTSIDE_ROI:
                    continue

                total_gt_in_roi += 1
                if gt.is_matched:
                    total_tp_in_roi += 1

                # ── Zone bucket ───────────────────────────────────────────
                if gt.zone_name not in zone_metrics:
                    zone_metrics[gt.zone_name] = ZoneMetrics(zone_name=gt.zone_name)
                zm = zone_metrics[gt.zone_name]
                zm.gt += 1
                if gt.is_matched:
                    zm.tp += 1

                # ── Zone × class bucket ───────────────────────────────────
                key = (gt.zone_name, gt.class_name)
                if key not in zone_class_metrics:
                    zone_class_metrics[key] = ZoneClassMetrics(
                        zone_name=gt.zone_name, class_name=gt.class_name
                    )
                zcm = zone_class_metrics[key]
                zcm.gt += 1
                if gt.is_matched:
                    zcm.tp += 1

        overall = {
            "total_gt": total_gt,
            "total_gt_in_roi": total_gt_in_roi,
            "total_tp_in_roi": total_tp_in_roi,
            "total_fn_in_roi": total_gt_in_roi - total_tp_in_roi,
            "recall_in_any_front_roi": round(
                total_tp_in_roi / total_gt_in_roi if total_gt_in_roi > 0 else 0.0, 4
            ),
        }

        zone_list = sorted(
            [zm.to_dict() for zm in zone_metrics.values()],
            key=lambda x: x["zone"],
        )
        zone_class_list = sorted(
            [zcm.to_dict() for zcm in zone_class_metrics.values()],
            key=lambda x: (x["zone"], x["class"]),
        )

        return overall, zone_list, zone_class_list

    # ── Per-image processing ──────────────────────────────────────────────────

    def _process_image(
        self, img_path: Path, lbl_path: Path
    ) -> List[GTObject]:
        """
        Detect objects in one image, load its GT labels, and run IoU matching.
        Returns a list of GTObjects with `is_matched` populated.
        All GT objects for which inference fails become FN.
        """
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Cannot read image: %s — skipping", img_path)
            return []

        img_h, img_w = frame.shape[:2]
        self.roi.update_frame_size(img_w, img_h)

        gt_objects = self._load_gt(lbl_path, img_w, img_h)
        if not gt_objects:
            return []

        try:
            predictions = self.detector.predict(frame)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Inference failed for %s: %s — all GT become FN", img_path.name, exc
            )
            return gt_objects  # is_matched stays False → counted as FN

        self._match(predictions, gt_objects)
        return gt_objects

    def _load_gt(
        self, lbl_path: Path, img_w: int, img_h: int
    ) -> List[GTObject]:
        """
        Parse a YOLO-format label file into GTObject instances.

        Each line: class_id  x_center  y_center  width  height  (normalised).
        Assigns each object to its ROI zone using bbox_bottom_center.
        Returns an empty list if the label file is absent (not an error).
        """
        if not lbl_path.exists():
            logger.debug("Label file not found: %s", lbl_path)
            return []

        gt_objects: List[GTObject] = []

        with lbl_path.open("r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    logger.warning(
                        "Malformed label at %s line %d: '%s'",
                        lbl_path.name, lineno, line,
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    bw  = float(parts[3])
                    bh  = float(parts[4])
                except ValueError:
                    logger.warning(
                        "Non-numeric value at %s line %d: '%s'",
                        lbl_path.name, lineno, line,
                    )
                    continue

                # Normalised → pixel xyxy, clamped to image bounds
                x1 = max(0, int((x_c - bw / 2) * img_w))
                y1 = max(0, int((y_c - bh / 2) * img_h))
                x2 = min(img_w - 1, int((x_c + bw / 2) * img_w))
                y2 = min(img_h - 1, int((y_c + bh / 2) * img_h))

                # Skip degenerate boxes that survived clamping
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = self.class_names.get(class_id, str(class_id))
                anchor = self.roi.get_reference_point((x1, y1, x2, y2))
                zone = self.roi.get_zone(anchor)
                zone_name = zone.name if zone is not None else OUTSIDE_ROI

                gt_objects.append(
                    GTObject(
                        class_id=class_id,
                        class_name=class_name,
                        bbox=(x1, y1, x2, y2),
                        anchor_point=anchor,
                        zone_name=zone_name,
                    )
                )

        return gt_objects

    def _match(
        self,
        predictions: List[Detection],
        gt_objects: List[GTObject],
    ) -> None:
        """
        Greedy IoU matching (standard object-detection convention).

        Predictions are sorted by confidence descending.  Each prediction is
        matched to the highest-IoU unmatched GT of the same class that meets
        `iou_match_threshold`.  Each GT can be matched at most once.

        Mutates gt_objects[i].is_matched in place.
        """
        sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)

        for pred in sorted_preds:
            best_iou = self.iou_match_threshold  # minimum bar
            best_idx = -1

            for idx, gt in enumerate(gt_objects):
                if gt.is_matched:
                    continue
                if gt.class_id != pred.class_id:
                    continue
                iou = _compute_iou(pred.bbox, gt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                gt_objects[best_idx].is_matched = True


# ─── IoU helper (module-level, reusable) ──────────────────────────────────────


def _compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union for two xyxy bounding boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ─── Dataset path resolution ──────────────────────────────────────────────────


def load_dataset_paths(
    data_source: str,
    split: str,
    project_root: Path,
) -> Tuple[List[Path], List[Path]]:
    """
    Resolve sorted lists of (image_path, label_path) pairs for evaluation.

    Supports two modes:
    1. YOLO data.yaml  — parses ``path`` / ``train`` / ``val`` / ``test``
       keys and resolves relative paths from the yaml's ``path`` entry.
    2. Direct directory — treats ``data_source`` (or ``data_source/split``)
       as the images directory.

    Labels are expected in a parallel ``labels/`` sibling of ``images/``
    (standard YOLO layout).  Falls back to the same directory as the image.
    """
    src = Path(data_source)
    if not src.is_absolute():
        src = project_root / src

    if src.suffix.lower() in (".yaml", ".yml"):
        return _paths_from_yaml(src, split)

    # Direct directory (allow appending split sub-directory)
    images_dir = src / split if (src / split).is_dir() else src
    return _paths_from_images_dir(images_dir)


def _paths_from_yaml(
    yaml_path: Path, split: str
) -> Tuple[List[Path], List[Path]]:
    """Parse a YOLO data.yaml and find images for the requested split."""
    with yaml_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Resolve base dataset directory.
    # Relative paths are resolved from PROJECT_ROOT (not the yaml file's directory)
    # so that 'path: yolov9/data/datasets' works as expected from the project root.
    base_str = cfg.get("path", "")
    if not base_str:
        base = yaml_path.parent
    else:
        base = Path(base_str)
        if not base.is_absolute():
            base = PROJECT_ROOT / base

    # Select the correct split key.
    # Support val/valid aliases: some datasets use 'valid/', others use 'val/'.
    _ALIASES: Dict[str, str] = {"val": "valid", "valid": "val"}
    if split in cfg:
        split_key = split
    elif _ALIASES.get(split, "") in cfg:
        split_key = _ALIASES[split]
        logger.info("Split '%s' not found in yaml; using alias '%s'", split, split_key)
    else:
        split_key = "val"
        logger.warning("Split '%s' not found in yaml; defaulting to 'val'", split)
    rel_images = cfg.get(split_key, "valid/images")

    images_dir = base / rel_images
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found for split '{split_key}': {images_dir}\n"
            f"  Check the 'path' and '{split_key}' keys in {yaml_path}"
        )

    return _paths_from_images_dir(images_dir)


def _paths_from_images_dir(
    images_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Return sorted (image_paths, label_paths) for every image in a directory."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    label_paths = [_find_label_path(p) for p in image_paths]
    return image_paths, label_paths


def _find_label_path(img_path: Path) -> Path:
    """
    Derive the YOLO label path for an image.

    Standard YOLO layout:  …/images/<split>/img.jpg → …/labels/<split>/img.txt
    Falls back to the same directory if 'images' is not in the path.
    """
    parts = img_path.parts
    for i, part in enumerate(reversed(parts)):
        if part == "images":
            idx = len(parts) - 1 - i
            lbl_path = (
                Path(*parts[:idx]) / "labels" / Path(*parts[idx + 1:])
            ).with_suffix(".txt")
            return lbl_path

    # Fallback: label alongside image
    return img_path.with_suffix(".txt")


# ─── Console output ────────────────────────────────────────────────────────────

_W = 60  # table width


def _hline(width: int = _W) -> str:
    return "─" * width


def print_overall(overall: Dict) -> None:
    print(f"\n{'═' * _W}")
    print("  OVERALL ROI METRICS")
    print(f"{'═' * _W}")
    rows = [
        ("Total GT objects",         overall["total_gt"]),
        ("GT objects inside any ROI", overall["total_gt_in_roi"]),
        ("TP inside ROI",             overall["total_tp_in_roi"]),
        ("FN inside ROI",             overall["total_fn_in_roi"]),
    ]
    for label, value in rows:
        print(f"  {label:<32}: {value}")
    recall = overall["recall_in_any_front_roi"]
    print(f"  {'Recall (any front ROI)':<32}: {recall:.4f}")
    print(f"{'═' * _W}")


def print_zone_table(zone_list: List[Dict]) -> None:
    col_zone   = 28
    col_num    = 6
    col_recall = 10
    width      = col_zone + col_num * 3 + col_recall + 4

    print(f"\n{'═' * width}")
    print("  RECALL BY ZONE")
    print(f"{'═' * width}")
    header = (
        f"  {'Zone':<{col_zone}}"
        f"{'GT':>{col_num}}"
        f"{'TP':>{col_num}}"
        f"{'FN':>{col_num}}"
        f"{'Recall':>{col_recall}}"
    )
    print(header)
    print(f"  {_hline(width - 2)}")

    for row in zone_list:
        print(
            f"  {row['zone']:<{col_zone}}"
            f"{row['gt']:>{col_num}}"
            f"{row['tp']:>{col_num}}"
            f"{row['fn']:>{col_num}}"
            f"{row['recall']:>{col_recall}.4f}"
        )
    print(f"{'═' * width}")


def print_zone_class_table(
    zone_class_list: List[Dict],
    highlight_classes: Optional[List[str]] = None,
) -> None:
    highlight = set(highlight_classes or ["person", "bike", "motor"])
    col_zone   = 28
    col_cls    = 10
    col_num    = 6
    col_recall = 10
    width      = col_zone + col_cls + col_num * 3 + col_recall + 4

    print(f"\n{'═' * width}")
    print("  RECALL BY ZONE × CLASS")
    print(f"{'═' * width}")
    header = (
        f"  {'Zone':<{col_zone}}"
        f"{'Class':<{col_cls}}"
        f"{'GT':>{col_num}}"
        f"{'TP':>{col_num}}"
        f"{'FN':>{col_num}}"
        f"{'Recall':>{col_recall}}"
    )
    print(header)
    print(f"  {_hline(width - 2)}")

    current_zone = None
    for row in zone_class_list:
        if row["zone"] != current_zone:
            if current_zone is not None:
                print()
            current_zone = row["zone"]

        marker = " *" if row["class"] in highlight else ""
        print(
            f"  {row['zone']:<{col_zone}}"
            f"{row['class']:<{col_cls}}"
            f"{row['gt']:>{col_num}}"
            f"{row['tp']:>{col_num}}"
            f"{row['fn']:>{col_num}}"
            f"{row['recall']:>{col_recall}.4f}"
            f"{marker}"
        )

    print(f"{'═' * width}")
    print("  * = priority safety class  (person / bike / motor)")


# ─── File output ──────────────────────────────────────────────────────────────


def save_outputs(
    output_dir: Path,
    overall: Dict,
    zone_list: List[Dict],
    zone_class_list: List[Dict],
    eval_config: Dict,
) -> None:
    """Write evaluation results to JSON and CSV files under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON summary (overall + per-zone) ─────────────────────────────────
    summary_path = output_dir / "roi_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "eval_config": eval_config,
                "overall_metrics": overall,
                "zone_metrics": zone_list,
            },
            fh,
            indent=2,
        )
    logger.info("Summary JSON   → %s", summary_path)

    # ── Per-zone CSV ───────────────────────────────────────────────────────
    zone_csv = output_dir / "roi_eval_zone_metrics.csv"
    _write_csv(zone_csv, ["zone", "gt", "tp", "fn", "recall"], zone_list)
    logger.info("Zone CSV       → %s", zone_csv)

    # ── Per-zone × class CSV ──────────────────────────────────────────────
    zc_csv = output_dir / "roi_eval_zone_class_metrics.csv"
    _write_csv(zc_csv, ["zone", "class", "gt", "tp", "fn", "recall"], zone_class_list)
    logger.info("Zone×class CSV → %s", zc_csv)


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROI-aware recall evaluation for truck blind spot detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to YOLOv9 weights (.pt)",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to YOLO data.yaml OR direct images directory",
    )
    parser.add_argument(
        "--roi", type=str, default="configs/roi.json",
        help="Path to ROI JSON config",
    )
    parser.add_argument(
        "--roi-profile", type=str, default="front_camera",
        help="ROI profile name (e.g. front_camera, rear_camera)",
    )
    parser.add_argument(
        "--classes-config", type=str, default="configs/classes.yaml",
        help="Path to classes YAML config",
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val", "valid", "test"],
        help="Dataset split to evaluate ('valid' is accepted as alias for 'val')",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25,
        help="Confidence threshold for the detector",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45,
        help="NMS IoU threshold for the detector",
    )
    parser.add_argument(
        "--iou-match", type=float, default=0.5,
        help="IoU threshold for GT↔prediction matching",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="CUDA device string ('0', '0,1') or 'cpu'",
    )
    parser.add_argument(
        "--imgsz", nargs=2, type=int, default=[640, 640],
        metavar=("H", "W"),
        help="Inference image size",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/roi_eval",
        help="Directory for saved evaluation files",
    )
    return parser.parse_args()


def _resolve(path: str, root: Path) -> str:
    """Return absolute path, resolving relative paths against project root."""
    p = Path(path)
    return str(p if p.is_absolute() else root / p)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    args = parse_args()

    weights_path      = _resolve(args.weights, PROJECT_ROOT)
    roi_path          = _resolve(args.roi, PROJECT_ROOT)
    classes_cfg_path  = _resolve(args.classes_config, PROJECT_ROOT)
    output_dir        = Path(_resolve(args.output_dir, PROJECT_ROOT))

    # ── Load detector ─────────────────────────────────────────────────────
    logger.info("Loading model: %s", weights_path)
    detector = YOLOv9Detector(
        weights_path=weights_path,
        classes_config_path=classes_cfg_path,
        device=args.device,
        image_size=tuple(args.imgsz),
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
    )

    # ── Load ROI ──────────────────────────────────────────────────────────
    logger.info("Loading ROI profile '%s': %s", args.roi_profile, roi_path)
    roi = MultiPolygonROI(roi_config_path=roi_path, profile_name=args.roi_profile)

    # ── Resolve dataset ───────────────────────────────────────────────────
    logger.info("Resolving dataset (split=%s): %s", args.split, args.data)
    image_paths, label_paths = load_dataset_paths(
        args.data, args.split, PROJECT_ROOT
    )

    # Quick sanity check on label coverage
    missing_labels = sum(1 for p in label_paths if not p.exists())
    logger.info(
        "Dataset: %d images | %d label files missing",
        len(image_paths), missing_labels,
    )

    # ── Run evaluation ────────────────────────────────────────────────────
    evaluator = ROIEvaluator(
        detector=detector,
        roi=roi,
        class_names=detector.class_names,
        iou_match_threshold=args.iou_match,
    )

    logger.info("Starting evaluation …")
    overall, zone_list, zone_class_list = evaluator.evaluate_dataset(
        image_paths, label_paths
    )

    # ── Console output ────────────────────────────────────────────────────
    print_overall(overall)
    print_zone_table(zone_list)
    print_zone_class_table(zone_class_list, highlight_classes=["person", "bike", "motor"])

    # ── Persist results ───────────────────────────────────────────────────
    eval_config = {
        "weights":     weights_path,
        "data":        args.data,
        "roi":         roi_path,
        "roi_profile": args.roi_profile,
        "split":       args.split,
        "conf_thres":  args.conf_thres,
        "iou_thres":   args.iou_thres,
        "iou_match":   args.iou_match,
        "imgsz":       args.imgsz,
        "num_images":  len(image_paths),
    }
    save_outputs(output_dir, overall, zone_list, zone_class_list, eval_config)
    logger.info("Done. Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()