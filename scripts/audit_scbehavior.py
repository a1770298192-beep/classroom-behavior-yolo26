"""Audit SCBehavior images and YOLO labels without modifying raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, object] = {"root": str(args.root)}
    class_counts: Counter[int] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    broken_images: list[str] = []
    invalid_labels: list[dict[str, object]] = []
    empty_labels: list[str] = []
    orphan_images: list[str] = []
    orphan_labels: list[str] = []
    auxiliary_label_files: list[str] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []

    for split in ("train", "val"):
        image_dir = args.root / "images" / split
        label_dir = args.root / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        image_by_stem = {path.stem: path for path in images}
        all_text_files = sorted(label_dir.glob("*.txt"))
        labels = [path for path in all_text_files if path.stem in image_by_stem]
        auxiliary_label_files.extend(
            str(path.relative_to(args.root)) for path in all_text_files if path.stem not in image_by_stem
        )
        label_by_stem = {path.stem: path for path in labels}

        orphan_images.extend(str(image_by_stem[stem].relative_to(args.root)) for stem in image_by_stem.keys() - label_by_stem.keys())
        orphan_labels.extend(str(label_by_stem[stem].relative_to(args.root)) for stem in label_by_stem.keys() - image_by_stem.keys())

        boxes = 0
        for image_path in images:
            relative = str(image_path.relative_to(args.root))
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    widths.append(image.width)
                    heights.append(image.height)
                    image.load()
                hashes[sha256(image_path)].append(relative)
            except Exception as exc:
                broken_images.append(f"{relative}: {exc}")

        for label_path in labels:
            relative = str(label_path.relative_to(args.root))
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels.append(relative)
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                problem = None
                parsed: list[float] = []
                if len(parts) != 5:
                    problem = f"expected 5 columns, got {len(parts)}"
                else:
                    try:
                        parsed = [float(value) for value in parts]
                    except ValueError:
                        problem = "non-numeric value"
                if parsed:
                    class_raw, x, y, width, height = parsed
                    class_id = int(class_raw)
                    if class_raw != class_id or not 0 <= class_id <= 6:
                        problem = f"invalid class id {class_raw}"
                    elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        problem = "normalized coordinate outside valid range"
                    elif (
                        x - width / 2 < -1e-6
                        or x + width / 2 > 1 + 1e-6
                        or y - height / 2 < -1e-6
                        or y + height / 2 > 1 + 1e-6
                    ):
                        problem = "bounding box crosses image boundary"
                    else:
                        class_counts[class_id] += 1
                        boxes += 1
                if problem:
                    invalid_labels.append({"file": relative, "line": line_number, "problem": problem, "text": line})

        split_stats[split] = {"images": len(images), "labels": len(labels), "valid_boxes": boxes}

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    report.update(
        {
            "splits": split_stats,
            "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
            "broken_images": broken_images,
            "invalid_labels": invalid_labels,
            "empty_labels": empty_labels,
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels,
            "auxiliary_label_files": auxiliary_label_files,
            "exact_duplicate_groups": duplicate_groups,
            "image_dimensions": {
                "min_width": min(widths, default=0),
                "max_width": max(widths, default=0),
                "min_height": min(heights, default=0),
                "max_height": max(heights, default=0),
            },
        }
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# SCBehavior 数据质量审计",
        "",
        "## 数据规模",
        "",
        "| 划分 | 图片 | 标签文件 | 合法标注框 |",
        "|---|---:|---:|---:|",
    ]
    for split, values in split_stats.items():
        lines.append(f"| {split} | {values['images']} | {values['labels']} | {values['valid_boxes']} |")
    lines.extend(["", "## 类别 ID 统计", "", "| 类别 ID | 合法标注框 |", "|---:|---:|"])
    for class_id, count in sorted(class_counts.items()):
        lines.append(f"| {class_id} | {count} |")
    lines.extend(
        [
            "",
            "## 质量问题",
            "",
            f"- 损坏图片：{len(broken_images)}",
            f"- 非法标注行：{len(invalid_labels)}",
            f"- 空标签文件：{len(empty_labels)}",
            f"- 缺少标签的图片：{len(orphan_images)}",
            f"- 缺少图片的标签：{len(orphan_labels)}",
            f"- 标签目录中的辅助文本文件：{len(auxiliary_label_files)}",
            f"- 完全重复图片组：{len(duplicate_groups)}",
            "",
            "详细文件清单见同目录 JSON 审计结果。",
        ]
    )
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if broken_images or invalid_labels or orphan_images or orphan_labels else 0


if __name__ == "__main__":
    raise SystemExit(main())
