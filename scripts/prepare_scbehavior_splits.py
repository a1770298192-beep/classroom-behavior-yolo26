"""Create scene-aware train/validation/test splits for SCBehavior."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import font_manager
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torchvision.models import ResNet18_Weights, resnet18


SPLITS = ("train", "val", "test")
TARGET = np.array([0.70, 0.15, 0.15])


def configure_chinese_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["axes.unicode_minus"] = False


def collect_samples(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_split in ("train", "val"):
        for image in sorted((root / "images" / source_split).glob("*.jpg")):
            label = root / "labels" / source_split / f"{image.stem}.txt"
            counts = np.zeros(7, dtype=int)
            for line in label.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
            row: dict[str, object] = {
                "source_split": source_split,
                "image_name": image.name,
                "image_path": str(image),
                "label_path": str(label),
                "box_count": int(counts.sum()),
            }
            row.update({f"class_{index}": int(value) for index, value in enumerate(counts)})
            rows.append(row)
    return pd.DataFrame(rows)


def extract_embeddings(samples: pd.DataFrame, cache: Path) -> np.ndarray:
    if cache.exists():
        payload = np.load(cache, allow_pickle=False)
        if len(payload) == len(samples):
            return payload
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    transform = weights.transforms()
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), 32):
            paths = samples.iloc[start : start + 32]["image_path"]
            batch = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to(device)
            features = torch.nn.functional.normalize(model(batch), dim=1)
            vectors.append(features.cpu().numpy())
    result = np.concatenate(vectors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, result)
    return result


def optimize_group_assignment(group_stats: np.ndarray, trials: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    totals = group_stats.sum(axis=0)
    best_score = float("inf")
    best: np.ndarray | None = None
    for _ in range(trials):
        assignment = rng.choice(3, size=len(group_stats), p=TARGET)
        if len(set(assignment.tolist())) < 3:
            continue
        achieved = np.stack([group_stats[assignment == split].sum(axis=0) for split in range(3)])
        if np.any(achieved[:, 0] == 0):
            continue
        # Require useful rare-class evaluation samples.
        if np.any(achieved[1:, 6] < 5) or np.any(achieved[1:, 7] < 10):
            continue
        ratios = achieved / totals
        image_error = np.square((ratios[:, 0] - TARGET) / TARGET).sum()
        class_error = np.square((ratios[:, 1:] - TARGET[:, None]) / TARGET[:, None]).mean()
        score = 2.5 * image_error + class_error
        if score < best_score:
            best_score = float(score)
            best = assignment.copy()
    if best is None:
        raise RuntimeError("Could not find a valid group assignment")
    return best, best_score


def write_dataset(samples: pd.DataFrame, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    for split in SPLITS:
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
    for row in samples.itertuples():
        shutil.copy2(row.image_path, destination / "images" / row.target_split / row.image_name)
        shutil.copy2(row.label_path, destination / "labels" / row.target_split / f"{Path(row.image_name).stem}.txt")


def save_split_chart(summary: pd.DataFrame, class_names: list[str], output: Path) -> None:
    values = summary[[f"class_{index}" for index in range(7)]].to_numpy().T
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(7)
    width = 0.24
    colors = ["#2563EB", "#F59E0B", "#16A34A"]
    for index, split in enumerate(SPLITS):
        bars = ax.bar(x + (index - 1) * width, values[:, index], width, label=split, color=colors[index])
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x, class_names)
    ax.set_yscale("log")
    ax.set_ylabel("标注框数量（对数轴）")
    ax.set_title("场景分组后各数据子集的类别分布")
    ax.grid(axis="y", alpha=0.2, which="both")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_chinese_font()

    samples = collect_samples(args.raw)
    embeddings = extract_embeddings(samples, args.embeddings)
    cluster_ids = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=args.threshold,
        metric="cosine",
        linkage="single",
    ).fit_predict(embeddings)
    samples["scene_group"] = cluster_ids

    group_rows = []
    for group_id, group in samples.groupby("scene_group"):
        values = [len(group)] + [int(group[f"class_{index}"].sum()) for index in range(7)]
        group_rows.append((int(group_id), values))
    group_rows.sort(key=lambda item: item[0])
    group_ids = np.array([item[0] for item in group_rows])
    group_stats = np.array([item[1] for item in group_rows], dtype=float)
    assignment, score = optimize_group_assignment(group_stats, args.trials, args.seed)
    group_to_split = {int(group): SPLITS[int(split)] for group, split in zip(group_ids, assignment)}
    samples["target_split"] = samples["scene_group"].map(group_to_split)

    write_dataset(samples, args.destination)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    samples.drop(columns=["image_path", "label_path"]).to_csv(args.manifest, index=False, encoding="utf-8-sig")

    summary_rows = []
    for split in SPLITS:
        subset = samples[samples["target_split"] == split]
        summary_rows.append(
            {
                "split": split,
                "images": int(len(subset)),
                "groups": int(subset["scene_group"].nunique()),
                "boxes": int(subset["box_count"].sum()),
                **{f"class_{index}": int(subset[f"class_{index}"].sum()) for index in range(7)},
            }
        )
    summary_frame = pd.DataFrame(summary_rows).set_index("split")

    # Every pair at or below the single-linkage threshold must share a split.
    similarity = embeddings @ embeddings.T
    distances = 1 - similarity
    cross_violations = 0
    minimum_cross_distance = float("inf")
    target_values = samples["target_split"].to_numpy()
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if target_values[left] != target_values[right]:
                distance = float(distances[left, right])
                minimum_cross_distance = min(minimum_cross_distance, distance)
                if distance <= args.threshold + 1e-7:
                    cross_violations += 1

    payload = {
        "seed": args.seed,
        "clustering": {
            "feature_extractor": "torchvision ResNet18 ImageNet1K_V1",
            "metric": "cosine",
            "linkage": "single",
            "distance_threshold": args.threshold,
            "groups": int(samples["scene_group"].nunique()),
            "largest_group": int(samples.groupby("scene_group").size().max()),
        },
        "optimization_trials": args.trials,
        "optimization_score": score,
        "cross_split_pairs_at_or_below_threshold": cross_violations,
        "minimum_cross_split_cosine_distance": minimum_cross_distance,
        "splits": summary_frame.reset_index().to_dict(orient="records"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    class_config = yaml.safe_load(args.classes.read_text(encoding="utf-8"))["classes"]
    names = [str(item["name"]) for item in class_config]
    data_yaml = {
        "path": str(args.destination.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    (args.destination / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    save_split_chart(summary_frame, [str(item["zh"]) for item in class_config], args.summary.parent / "split_class_distribution.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if cross_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
