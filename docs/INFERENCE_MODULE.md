# 图片与视频推理模块说明

## 核心模块

`src/campus_behavior/inference.py` 提供统一的 `BehaviorDetector` 类，负责：

- 加载系统默认YOLO26s权重；
- 自动选择CUDA或CPU；
- 图片和OpenCV BGR帧推理；
- 跨类别NMS，减少同一学生被多个行为框重复统计；
- 使用Windows中文字体绘制类别和置信度；
- 返回结构化检测框、类别、置信度和行为计数；
- 图片结果与JSON保存；
- 视频逐帧或间隔帧推理；
- 带标注MP4、时间序列CSV和汇总JSON导出。

## 图片推理

```powershell
.\.venv\Scripts\python.exe scripts\predict_image.py `
  input.jpg `
  --output outputs\predictions\result.jpg `
  --json outputs\predictions\result.json `
  --conf 0.25
```

JSON包含7类行为计数、每个检测框坐标、中文与英文类别、置信度以及预处理/推理/后处理时间。

## 视频推理

```powershell
.\.venv\Scripts\python.exe scripts\predict_video.py `
  input.mp4 `
  --output-dir outputs\predictions\video_result `
  --frame-stride 1 `
  --conf 0.25
```

输出包括：

- `<视频名>_annotated.mp4`：标注后视频；
- `<视频名>_timeline.csv`：每个分析帧的时间、检测总数、7类行为计数和推理时间；
- `<视频名>_summary.json`：视频参数、处理速度、各类平均值、最大值和检测次数总和。

当 `frame_stride>1` 时，仅分析指定间隔的帧；未分析帧保留原始画面，不重复上一分析帧，避免输出视频冻结。CSV只记录实际分析的帧。

## 计数语义

图片中的计数表示当前图片内每类检测框数量。视频统计表示每个分析帧中的检测框数量及其时间变化，并不是跨帧跟踪后去重的独立学生人数。例如同一名学生持续10帧被检测为阅读，会产生10次逐帧阅读检测记录。

系统不做人脸识别，也不把检测结果与学生身份绑定。若未来需要统计“独立学生人数”或行为持续时间，应增加匿名多目标跟踪，并单独评估ID切换和遮挡问题。

## 阶段验收

- 真实测试图片成功生成中文标注图和JSON；
- 7类计数字段完整；
- 30帧、10 FPS测试视频成功输出30帧标注视频；
- `frame_stride=2`时生成15条时间序列记录；
- CSV使用UTF-8 BOM，Excel可直接显示中文；
- 汇总JSON中的输入帧数、分析帧数和文件路径验证通过；
- 系统默认模型为 `models/final/classroom_behavior_yolo26s.pt`。

