"""Unified Ultralytics YOLO training entry point with run metadata."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def resolve_path(value: str, project_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def make_json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["data"] = resolve_path(str(config["data"]), project_root)
    config["project"] = resolve_path(str(config["project"]), project_root)
    config["model"] = resolve_path(str(config["model"]), project_root)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing unintended CPU training")
    torch.cuda.reset_peak_memory_stats()

    started = time.time()
    if args.resume:
        checkpoint = args.resume if args.resume.is_absolute() else project_root / args.resume
        model = YOLO(str(checkpoint.resolve()))
        results = model.train(resume=True)
    else:
        model = YOLO(config.pop("model"))
        results = model.train(**config)
    duration = time.time() - started

    save_dir = Path(model.trainer.save_dir)
    metrics = getattr(results, "results_dict", {})
    if not metrics and getattr(model.trainer, "validator", None) is not None:
        metrics = getattr(model.trainer.validator.metrics, "results_dict", {})
    metadata = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "duration_seconds": duration,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "save_dir": str(save_dir),
        "metrics": make_json_safe(metrics),
        "checkpoint_load_test": {},
    }

    for checkpoint_name in ("last.pt", "best.pt"):
        checkpoint = save_dir / "weights" / checkpoint_name
        if checkpoint.exists():
            loaded = YOLO(str(checkpoint))
            metadata["checkpoint_load_test"][checkpoint_name] = {
                "exists": True,
                "task": loaded.task,
                "size_bytes": checkpoint.stat().st_size,
            }
        else:
            metadata["checkpoint_load_test"][checkpoint_name] = {"exists": False}

    (save_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

