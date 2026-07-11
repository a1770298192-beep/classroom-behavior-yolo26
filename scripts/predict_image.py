"""Command-line image inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from campus_behavior.inference import BehaviorDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/final/classroom_behavior_yolo26s.pt")
    parser.add_argument("--classes", type=Path, default=PROJECT_ROOT / "config/classes.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    detector = BehaviorDetector(args.model, args.classes, device=args.device, confidence=args.conf)
    result = detector.infer_image_file(args.input, args.output, args.json)
    print(json.dumps(result.metadata(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

