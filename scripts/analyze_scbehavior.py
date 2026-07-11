"""Generate reproducible SCBehavior statistics and report-ready figures."""

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


COLORS = ["#2563EB", "#0891B2", "#16A34A", "#CA8A04", "#EA580C", "#DC2626", "#9333EA"]


def configure_chinese_font() -> Path:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese font found")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False
    return font_path


def load_classes(path: Path) -> list[dict[str, object]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["classes"]


def read_records(root: Path, classes: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    boxes: list[dict[str, object]] = []
    images: list[dict[str, object]] = []
    class_lookup = {int(item["id"]): item for item in classes}

    for split in ("train", "val"):
        for image_path in sorted((root / "images" / split).glob("*.jpg")):
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            per_image = Counter()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id_raw, x, y, width, height = map(float, line.split())
                class_id = int(class_id_raw)
                item = class_lookup[class_id]
                scale_640 = min(640 / image_width, 640 / image_height)
                width_640 = width * image_width * scale_640
                height_640 = height * image_height * scale_640
                area_640 = width_640 * height_640
                if area_640 < 32 * 32:
                    size_group = "small"
                elif area_640 < 96 * 96:
                    size_group = "medium"
                else:
                    size_group = "large"
                boxes.append(
                    {
                        "split": split,
                        "image": image_path.name,
                        "class_id": class_id,
                        "class_name": item["name"],
                        "class_zh": item["zh"],
                        "x": x,
                        "y": y,
                        "width_norm": width,
                        "height_norm": height,
                        "area_norm": width * height,
                        "width_px": width * image_width,
                        "height_px": height * image_height,
                        "width_at_640": width_640,
                        "height_at_640": height_640,
                        "area_at_640": area_640,
                        "size_group_at_640": size_group,
                        "aspect_ratio": width / height,
                    }
                )
                per_image[class_id] += 1
            images.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "width": image_width,
                    "height": image_height,
                    "box_count": sum(per_image.values()),
                    **{f"class_{class_id}_count": per_image[class_id] for class_id in class_lookup},
                }
            )
    return pd.DataFrame(boxes), pd.DataFrame(images)


def save_class_distribution(boxes: pd.DataFrame, classes: list[dict[str, object]], output: Path) -> None:
    counts = boxes.groupby("class_id").size().reindex(range(len(classes)), fill_value=0)
    labels = [str(item["zh"]) for item in classes]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(labels, counts.values, color=COLORS)
    ax.set_title("SCBehavior 各类行为标注框数量", fontsize=16, pad=14)
    ax.set_xlabel("行为类别")
    ax.set_ylabel("标注框数量")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.012, f"{count}\n{count / counts.sum():.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_size_distribution(boxes: pd.DataFrame, output: Path) -> None:
    order = ["small", "medium", "large"]
    labels = ["小目标", "中目标", "大目标"]
    counts = boxes["size_group_at_640"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(labels, counts.values, color=["#DC2626", "#F59E0B", "#16A34A"])
    ax.set_title("缩放至 640×640 后的目标尺度分布", fontsize=16, pad=14)
    ax.set_ylabel("标注框数量")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.012, f"{count}\n{count / counts.sum():.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_box_scatter(boxes: pd.DataFrame, output: Path) -> None:
    sample = boxes.sample(min(len(boxes), 4000), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample["width_at_640"], sample["height_at_640"], s=8, alpha=0.28, c=sample["class_id"], cmap="tab10")
    ax.axvline(32, color="#DC2626", linestyle="--", linewidth=1, label="32 px")
    ax.axhline(32, color="#DC2626", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("目标框宽高分布（映射到 640×640 输入）", fontsize=15, pad=14)
    ax.set_xlabel("目标框宽度 / px（对数轴）")
    ax.set_ylabel("目标框高度 / px（对数轴）")
    ax.grid(alpha=0.18, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_image(root: Path, split: str, image_name: str, classes: list[dict[str, object]], font_path: Path, output: Path) -> None:
    image_path = root / "images" / split / image_name
    label_path = root / "labels" / split / f"{Path(image_name).stem}.txt"
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(24, image.width // 80)
    font = ImageFont.truetype(str(font_path), font_size)
    lookup = {int(item["id"]): item for item in classes}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_raw, x, y, width, height = map(float, line.split())
        class_id = int(class_id_raw)
        left = int((x - width / 2) * image.width)
        top = int((y - height / 2) * image.height)
        right = int((x + width / 2) * image.width)
        bottom = int((y + height / 2) * image.height)
        color = COLORS[class_id]
        draw.rectangle((left, top, right, bottom), outline=color, width=max(3, image.width // 700))
        text = str(lookup[class_id]["zh"])
        text_box = draw.textbbox((left, top), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_top = max(0, top - text_height - 8)
        draw.rectangle((left, label_top, left + text_width + 10, label_top + text_height + 8), fill=color)
        draw.text((left + 5, label_top + 2), text, fill="white", font=font)
    image.thumbnail((1800, 1100), Image.Resampling.LANCZOS)
    image.save(output, quality=94)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    font_path = configure_chinese_font()
    classes = load_classes(args.classes)
    boxes, images = read_records(args.root, classes)
    boxes.to_csv(args.stats_dir / "scbehavior_boxes.csv", index=False, encoding="utf-8-sig")
    images.to_csv(args.stats_dir / "scbehavior_images.csv", index=False, encoding="utf-8-sig")

    save_class_distribution(boxes, classes, args.output_dir / "class_distribution.png")
    save_size_distribution(boxes, args.output_dir / "target_size_distribution.png")
    save_box_scatter(boxes, args.output_dir / "target_width_height.png")

    rare_candidates = images[(images["class_5_count"] > 0) | (images["class_6_count"] > 0)].copy()
    rare_candidates["rare_count"] = rare_candidates["class_5_count"] + rare_candidates["class_6_count"]
    selected = rare_candidates.sort_values(["rare_count", "box_count"], ascending=False).head(2)
    typical = images.iloc[(images["box_count"] - images["box_count"].median()).abs().argsort()[:1]]
    for index, row in enumerate(pd.concat([selected, typical]).drop_duplicates(["split", "image"]).itertuples(), start=1):
        annotate_image(args.root, row.split, row.image, classes, font_path, args.output_dir / f"annotated_sample_{index}.jpg")

    class_summary = []
    for item in classes:
        subset = boxes[boxes["class_id"] == int(item["id"])]
        class_summary.append(
            {
                "class_id": int(item["id"]),
                "name": item["name"],
                "zh": item["zh"],
                "boxes": int(len(subset)),
                "share": float(len(subset) / len(boxes)),
                "images_present": int(subset[["split", "image"]].drop_duplicates().shape[0]),
                "small_share_at_640": float((subset["size_group_at_640"] == "small").mean()),
                "median_width_at_640": float(subset["width_at_640"].median()),
                "median_height_at_640": float(subset["height_at_640"].median()),
            }
        )
    size_counts = boxes["size_group_at_640"].value_counts().to_dict()
    summary = {
        "images": int(len(images)),
        "boxes": int(len(boxes)),
        "boxes_per_image": {
            "min": int(images["box_count"].min()),
            "median": float(images["box_count"].median()),
            "mean": float(images["box_count"].mean()),
            "max": int(images["box_count"].max()),
        },
        "size_counts_at_640": {key: int(value) for key, value in size_counts.items()},
        "class_summary": class_summary,
    }
    (args.stats_dir / "scbehavior_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
