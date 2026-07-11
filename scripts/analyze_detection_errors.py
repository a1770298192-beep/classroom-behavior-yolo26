"""Perform fixed-threshold, box-level error analysis on a YOLO test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


COLORS = ["#2563EB", "#0891B2", "#16A34A", "#CA8A04", "#EA580C", "#DC2626", "#9333EA"]


def setup_font() -> tuple[str, Path]:
    path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
    return family, path


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=float)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0, None).prod(axis=1)
    return intersection / np.clip(left_area[:, None] + right_area[None, :] - intersection, 1e-9, None)


def read_ground_truth(label: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes, boxes, sizes = [], [], []
    scale = min(640 / width, 640 / height)
    for line in label.read_text(encoding="utf-8").splitlines():
        class_id, x, y, box_width, box_height = map(float, line.split())
        x1 = (x - box_width / 2) * width
        y1 = (y - box_height / 2) * height
        x2 = (x + box_width / 2) * width
        y2 = (y + box_height / 2) * height
        area_640 = (box_width * width * scale) * (box_height * height * scale)
        size = "small" if area_640 < 32**2 else "medium" if area_640 < 96**2 else "large"
        classes.append(int(class_id))
        boxes.append([x1, y1, x2, y2])
        sizes.append(size)
    return np.array(classes, dtype=int), np.array(boxes, dtype=float), sizes


def greedy_match(ious: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    work = ious.copy()
    while work.size and work.max(initial=0) >= threshold:
        gt_index, pred_index = np.unravel_index(np.argmax(work), work.shape)
        score = float(work[gt_index, pred_index])
        matches.append((int(gt_index), int(pred_index), score))
        work[gt_index, :] = -1
        work[:, pred_index] = -1
    return matches


def draw_error_image(
    image_path: Path,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_conf: np.ndarray,
    names_zh: list[str],
    font_path: Path,
    output: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), max(22, image.width // 100))
    for box, class_id in zip(gt_boxes, gt_classes):
        draw.rectangle(tuple(box), outline="#22C55E", width=max(3, image.width // 800))
        draw.text((box[0] + 3, box[1] + 3), f"真:{names_zh[class_id]}", font=font, fill="#22C55E", stroke_width=2, stroke_fill="black")
    for box, class_id, confidence in zip(pred_boxes, pred_classes, pred_conf):
        draw.rectangle(tuple(box), outline="#EF4444", width=max(3, image.width // 800))
        draw.text((box[0] + 3, max(0, box[1] - font.size - 5)), f"预:{names_zh[class_id]} {confidence:.2f}", font=font, fill="#EF4444", stroke_width=2, stroke_fill="white")
    image.thumbnail((1800, 1100), Image.Resampling.LANCZOS)
    image.save(output, quality=94)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    args = parser.parse_args()
    _, font_path = setup_font()
    args.output.mkdir(parents=True, exist_ok=True)

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data["path"])
    class_config = yaml.safe_load(args.classes.read_text(encoding="utf-8"))["classes"]
    names = [str(item["name"]) for item in class_config]
    names_zh = [str(item["zh"]) for item in class_config]
    image_paths = sorted((root / data["test"]).glob("*.jpg"))
    model = YOLO(str(args.model.resolve()))
    predictions = model.predict(image_paths, imgsz=640, conf=args.conf, iou=0.7, device=0, batch=16, verbose=False)

    event_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    confusion = np.zeros((7, 7), dtype=int)
    gt_size_total = Counter()
    gt_size_correct = Counter()
    sample_payload: dict[str, tuple] = {}

    for image_path, result in zip(image_paths, predictions):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = root / "labels" / "test" / f"{image_path.stem}.txt"
        gt_classes, gt_boxes, gt_sizes = read_ground_truth(label_path, width, height)
        if result.boxes is None:
            pred_boxes = np.empty((0, 4))
            pred_classes = np.empty((0,), dtype=int)
            pred_conf = np.empty((0,))
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_classes = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()

        matches = greedy_match(iou_matrix(gt_boxes, pred_boxes), args.match_iou)
        matched_gt = {gt for gt, _, _ in matches}
        matched_pred = {pred for _, pred, _ in matches}
        correct = confused = 0

        for gt_index, pred_index, iou in matches:
            gt_class = int(gt_classes[gt_index])
            pred_class = int(pred_classes[pred_index])
            gt_size_total[gt_sizes[gt_index]] += 1
            if gt_class == pred_class:
                status = "true_positive"
                correct += 1
                gt_size_correct[gt_sizes[gt_index]] += 1
            else:
                status = "class_confusion"
                confused += 1
                confusion[gt_class, pred_class] += 1
            event_rows.append(
                {
                    "image": image_path.name,
                    "status": status,
                    "gt_class": gt_class,
                    "gt_name": names[gt_class],
                    "pred_class": pred_class,
                    "pred_name": names[pred_class],
                    "confidence": float(pred_conf[pred_index]),
                    "iou": iou,
                    "size": gt_sizes[gt_index],
                }
            )

        for gt_index, (gt_class, size) in enumerate(zip(gt_classes, gt_sizes)):
            if gt_index in matched_gt:
                continue
            gt_size_total[size] += 1
            event_rows.append(
                {
                    "image": image_path.name,
                    "status": "false_negative",
                    "gt_class": int(gt_class),
                    "gt_name": names[int(gt_class)],
                    "pred_class": None,
                    "pred_name": None,
                    "confidence": None,
                    "iou": None,
                    "size": size,
                }
            )
        for pred_index, pred_class in enumerate(pred_classes):
            if pred_index in matched_pred:
                continue
            event_rows.append(
                {
                    "image": image_path.name,
                    "status": "false_positive",
                    "gt_class": None,
                    "gt_name": None,
                    "pred_class": int(pred_class),
                    "pred_name": names[int(pred_class)],
                    "confidence": float(pred_conf[pred_index]),
                    "iou": None,
                    "size": None,
                }
            )

        missed = len(gt_classes) - len(matched_gt)
        false_positive = len(pred_classes) - len(matched_pred)
        image_rows.append(
            {
                "image": image_path.name,
                "gt": len(gt_classes),
                "pred": len(pred_classes),
                "correct": correct,
                "confused": confused,
                "missed": missed,
                "false_positive": false_positive,
                "error_score": confused * 2 + missed + false_positive,
            }
        )
        sample_payload[image_path.name] = (image_path, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_conf)

    events = pd.DataFrame(event_rows)
    image_stats = pd.DataFrame(image_rows).sort_values("error_score", ascending=False)
    events.to_csv(args.output / "error_events.csv", index=False, encoding="utf-8-sig")
    image_stats.to_csv(args.output / "image_error_summary.csv", index=False, encoding="utf-8-sig")

    status_counts = events["status"].value_counts().to_dict()
    per_class = []
    for class_id, name in enumerate(names):
        gt_events = events[events["gt_class"] == class_id]
        pred_events = events[events["pred_class"] == class_id]
        per_class.append(
            {
                "class_id": class_id,
                "name": name,
                "zh": names_zh[class_id],
                "gt": int(len(gt_events)),
                "correct": int((gt_events["status"] == "true_positive").sum()),
                "confused": int((gt_events["status"] == "class_confusion").sum()),
                "missed": int((gt_events["status"] == "false_negative").sum()),
                "false_positive": int((pred_events["status"] == "false_positive").sum()),
                "fixed_threshold_recall": float((gt_events["status"] == "true_positive").mean()),
            }
        )

    size_recall = {
        size: {
            "gt": int(gt_size_total[size]),
            "correct": int(gt_size_correct[size]),
            "recall": float(gt_size_correct[size] / gt_size_total[size]) if gt_size_total[size] else 0.0,
        }
        for size in ("small", "medium", "large")
    }
    confusion_rows = []
    for gt_class in range(7):
        for pred_class in range(7):
            if confusion[gt_class, pred_class]:
                confusion_rows.append(
                    {
                        "gt": names[gt_class],
                        "pred": names[pred_class],
                        "count": int(confusion[gt_class, pred_class]),
                    }
                )
    confusion_rows.sort(key=lambda item: item["count"], reverse=True)
    summary = {
        "confidence_threshold": args.conf,
        "matching_iou_threshold": args.match_iou,
        "images": len(image_paths),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "per_class": per_class,
        "size_recall": size_recall,
        "class_confusions": confusion_rows,
    }
    (args.output / "error_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report-ready charts.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    recall_values = [size_recall[size]["recall"] for size in ("small", "medium", "large")]
    bars = ax.bar(["小目标", "中目标", "大目标"], recall_values, color=["#DC2626", "#F59E0B", "#16A34A"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("固定阈值召回率")
    ax.set_title("不同目标尺度的正确检测召回率（conf=0.25, IoU=0.50）")
    for bar, value in zip(bars, recall_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(args.output / "recall_by_size.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    top_errors = image_stats.head(6)["image"].tolist()
    for index, image_name in enumerate(top_errors, start=1):
        draw_error_image(*sample_payload[image_name], names_zh, font_path, args.output / f"error_sample_{index}.jpg")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

