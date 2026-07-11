"""Evaluate a trained YOLO checkpoint on a configured dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--project", type=Path, default=Path("experiments"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    model = YOLO(str(args.model.resolve()))
    results = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    output = Path(results.save_dir) / "evaluation_metrics.json"
    payload: dict[str, object] = {
        "overall": {str(key): float(value) for key, value in results.results_dict.items()},
        "speed_ms_per_image": {str(key): float(value) for key, value in results.speed.items()},
        "per_class": [],
    }
    class_indices = results.box.ap_class_index.astype(int).tolist()
    for position, class_id in enumerate(class_indices):
        payload["per_class"].append(
            {
                "class_id": class_id,
                "name": model.names[class_id],
                "precision": float(results.box.p[position]),
                "recall": float(results.box.r[position]),
                "ap50": float(results.box.ap50[position]),
                "ap50_95": float(results.box.ap[position]),
            }
        )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
