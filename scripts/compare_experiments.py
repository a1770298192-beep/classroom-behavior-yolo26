"""Aggregate experiment metrics and benchmark batch-1 inference consistently."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import font_manager
from ultralytics import YOLO


EXPERIMENTS = [
    {
        "name": "YOLO26n@640",
        "run": "baseline_yolo26n",
        "test": "baseline_yolo26n_test",
        "model": "experiments/baseline_yolo26n/weights/best.pt",
        "imgsz": 640,
        "gflops": 5.2,
    },
    {
        "name": "YOLO26s@640",
        "run": "experiment_yolo26s_640",
        "test": "experiment_yolo26s_640_test",
        "model": "experiments/experiment_yolo26s_640/weights/best.pt",
        "imgsz": 640,
        "gflops": 20.5,
    },
    {
        "name": "YOLO26n@960",
        "run": "experiment_yolo26n_960",
        "test": "experiment_yolo26n_960_test",
        "model": "experiments/experiment_yolo26n_960/weights/best.pt",
        "imgsz": 960,
        "gflops": 11.7,
    },
]


def benchmark(model: YOLO, image: Path, imgsz: int, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, conf=0.25, device=0, verbose=False)
    torch.cuda.synchronize()
    inference_values = []
    started = time.perf_counter()
    for _ in range(repeats):
        result = model.predict(image, imgsz=imgsz, conf=0.25, device=0, verbose=False)[0]
        inference_values.append(float(result.speed["inference"]))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "model_inference_ms": sum(inference_values) / len(inference_values),
        "end_to_end_ms": elapsed * 1000 / repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chinese_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
        plt.rcParams["axes.unicode_minus"] = False

    rows, class_rows = [], []
    for experiment in EXPERIMENTS:
        run_dir = args.root / "experiments" / experiment["run"]
        test_dir = args.root / "experiments" / experiment["test"]
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        test = json.loads((test_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
        model_path = args.root / experiment["model"]
        model = YOLO(str(model_path))
        speed = benchmark(model, args.image, int(experiment["imgsz"]), args.warmup, args.repeats)
        rows.append(
            {
                "experiment": experiment["name"],
                "imgsz": experiment["imgsz"],
                "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
                "gflops": float(experiment["gflops"]),
                "weight_mb": model_path.stat().st_size / 1024**2,
                "training_seconds": metadata["duration_seconds"],
                "peak_allocated_vram_gib": metadata["peak_cuda_memory_gib"],
                "precision": test["overall"]["metrics/precision(B)"],
                "recall": test["overall"]["metrics/recall(B)"],
                "map50": test["overall"]["metrics/mAP50(B)"],
                "map50_95": test["overall"]["metrics/mAP50-95(B)"],
                **speed,
            }
        )
        for item in test["per_class"]:
            class_rows.append({"experiment": experiment["name"], **item})

    frame = pd.DataFrame(rows)
    classes = pd.DataFrame(class_rows)
    frame.to_csv(args.output / "experiment_comparison.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(args.output / "per_class_comparison.csv", index=False, encoding="utf-8-sig")
    (args.output / "experiment_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2563EB", "#16A34A", "#F59E0B"]
    axes[0].bar(frame["experiment"], frame["map50_95"], color=colors)
    axes[0].set_ylim(0.5, 0.64)
    axes[0].set_ylabel("测试集 mAP50-95")
    axes[0].set_title("检测精度对比")
    for index, value in enumerate(frame["map50_95"]):
        axes[0].text(index, value + 0.003, f"{value:.4f}", ha="center")
    axes[1].bar(frame["experiment"], frame["end_to_end_ms"], color=colors)
    axes[1].set_ylabel("单张端到端时间 / ms")
    axes[1].set_title("Batch=1 重复推理速度")
    for index, value in enumerate(frame["end_to_end_ms"]):
        axes[1].text(index, value + max(frame["end_to_end_ms"]) * 0.02, f"{value:.1f}", ha="center")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(args.output / "accuracy_speed_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
