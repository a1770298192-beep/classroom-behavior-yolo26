"""Reusable image and video inference for classroom behavior detection."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


COLORS_RGB = [
    (37, 99, 235),
    (8, 145, 178),
    (22, 163, 74),
    (202, 138, 4),
    (234, 88, 12),
    (220, 38, 38),
    (147, 51, 234),
]


@dataclass(frozen=True)
class Detection:
    class_id: int
    name: str
    name_zh: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageInferenceResult:
    detections: list[Detection]
    counts: dict[str, int]
    annotated_bgr: np.ndarray
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    def metadata(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "detections": [asdict(item) for item in self.detections],
            "speed_ms": {
                "preprocess": self.preprocess_ms,
                "inference": self.inference_ms,
                "postprocess": self.postprocess_ms,
            },
        }


@dataclass(frozen=True)
class VideoInferenceResult:
    output_video: Path
    timeline_csv: Path
    summary_json: Path
    input_frames: int
    analyzed_frames: int
    fps: float
    duration_seconds: float


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


class BehaviorDetector:
    def __init__(
        self,
        model_path: str | Path,
        classes_path: str | Path,
        device: str = "auto",
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.70,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = choose_device(device)
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        config = yaml.safe_load(Path(classes_path).read_text(encoding="utf-8"))["classes"]
        self.classes = {int(item["id"]): item for item in config}
        self.class_names_zh = [str(self.classes[index]["zh"]) for index in sorted(self.classes)]
        self.model = YOLO(str(self.model_path))
        self.font_path = self._find_font()

    @staticmethod
    def _find_font() -> Path:
        candidates = [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
        ]
        font = next((path for path in candidates if path.exists()), None)
        if font is None:
            raise FileNotFoundError("未找到可用中文字体")
        return font

    def infer_bgr(self, frame_bgr: np.ndarray) -> ImageInferenceResult:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("输入图像为空")
        result = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[Detection] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            for box, class_id, confidence in zip(boxes, classes, confidences):
                item = self.classes[class_id]
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        name=str(item["name"]),
                        name_zh=str(item["zh"]),
                        confidence=float(confidence),
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                    )
                )
        counts_counter = Counter(item.name_zh for item in detections)
        counts = {name: int(counts_counter.get(name, 0)) for name in self.class_names_zh}
        annotated = self.draw_detections(frame_bgr, detections)
        return ImageInferenceResult(
            detections=detections,
            counts=counts,
            annotated_bgr=annotated,
            preprocess_ms=float(result.speed.get("preprocess", 0.0)),
            inference_ms=float(result.speed.get("inference", 0.0)),
            postprocess_ms=float(result.speed.get("postprocess", 0.0)),
        )

    def draw_detections(self, frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        font_size = max(18, image.width // 90)
        font = ImageFont.truetype(str(self.font_path), font_size)
        line_width = max(2, image.width // 700)
        for detection in detections:
            color = COLORS_RGB[detection.class_id % len(COLORS_RGB)]
            box = (detection.x1, detection.y1, detection.x2, detection.y2)
            draw.rectangle(box, outline=color, width=line_width)
            label = f"{detection.name_zh} {detection.confidence:.2f}"
            bounds = draw.textbbox((0, 0), label, font=font)
            label_width = bounds[2] - bounds[0] + 10
            label_height = bounds[3] - bounds[1] + 8
            label_y = max(0, detection.y1 - label_height)
            draw.rectangle(
                (detection.x1, label_y, detection.x1 + label_width, label_y + label_height),
                fill=color,
            )
            draw.text((detection.x1 + 5, label_y + 2), label, font=font, fill="white")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def infer_image_file(
        self,
        input_path: str | Path,
        output_image: str | Path,
        output_json: str | Path | None = None,
    ) -> ImageInferenceResult:
        input_path = Path(input_path)
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError(f"无法读取图片: {input_path}")
        result = self.infer_bgr(frame)
        output_image = Path(output_image)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_image), result.annotated_bgr):
            raise OSError(f"无法保存图片: {output_image}")
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {"input": str(input_path), "output": str(output_image), **result.metadata()}
            output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def infer_video_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        frame_stride: int = 1,
        progress: Callable[[int, int], None] | None = None,
    ) -> VideoInferenceResult:
        if frame_stride < 1:
            raise ValueError("frame_stride必须大于等于1")
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_video = output_dir / f"{input_path.stem}_annotated.mp4"
        timeline_csv = output_dir / f"{input_path.stem}_timeline.csv"
        summary_json = output_dir / f"{input_path.stem}_summary.json"
        writer = cv2.VideoWriter(
            str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise OSError("无法创建输出视频")

        timeline: list[dict[str, object]] = []
        frame_index = 0
        analyzed_frames = 0
        started = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                output_frame = frame
                if frame_index % frame_stride == 0:
                    inference = self.infer_bgr(frame)
                    output_frame = inference.annotated_bgr
                    analyzed_frames += 1
                    timeline.append(
                        {
                            "frame": frame_index,
                            "time_seconds": frame_index / fps,
                            "detections": len(inference.detections),
                            **inference.counts,
                            "inference_ms": inference.inference_ms,
                        }
                    )
                writer.write(output_frame)
                frame_index += 1
                if progress is not None:
                    progress(frame_index, total_frames)
        finally:
            capture.release()
            writer.release()
        duration = time.perf_counter() - started

        fieldnames = [
            "frame",
            "time_seconds",
            "detections",
            *self.class_names_zh,
            "inference_ms",
        ]
        with timeline_csv.open("w", newline="", encoding="utf-8-sig") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(timeline)

        aggregates = {}
        for class_name in self.class_names_zh:
            values = [int(row[class_name]) for row in timeline]
            aggregates[class_name] = {
                "mean_per_analyzed_frame": float(np.mean(values)) if values else 0.0,
                "max_per_analyzed_frame": int(max(values, default=0)),
                "sum_detections": int(sum(values)),
            }
        summary = {
            "input": str(input_path),
            "output_video": str(output_video),
            "timeline_csv": str(timeline_csv),
            "fps": fps,
            "input_frames": frame_index,
            "analyzed_frames": analyzed_frames,
            "frame_stride": frame_stride,
            "processing_seconds": duration,
            "processing_fps": frame_index / duration if duration else 0.0,
            "note": "视频计数为每个分析帧中的检测数量，不代表去重后的独立学生人数。",
            "behavior_statistics": aggregates,
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return VideoInferenceResult(
            output_video=output_video,
            timeline_csv=timeline_csv,
            summary_json=summary_json,
            input_frames=frame_index,
            analyzed_frames=analyzed_frames,
            fps=fps,
            duration_seconds=duration,
        )
