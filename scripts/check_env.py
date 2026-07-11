"""Verify the local runtime before dataset processing or model training."""

from __future__ import annotations

import platform
import sys

import cv2
import torch
import ultralytics


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable; GPU training is not ready.")
        return 1

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    properties = torch.cuda.get_device_properties(device)
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")

    left = torch.randn((512, 512), device=device)
    right = torch.randn((512, 512), device=device)
    result = left @ right
    torch.cuda.synchronize()
    print(f"CUDA tensor test: OK ({result.shape[0]}x{result.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

