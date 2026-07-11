# 基于 YOLO26 的课堂学生行为识别系统

贵州大学课程设计项目。系统使用 Python、PyTorch 与 Ultralytics YOLO26，对课堂图片或视频中的学生行为进行目标检测，支持中文可视化、逐帧行为统计、JSON 汇总与 CSV 时间线输出。

> 本系统只提供课堂活动辅助分析，不进行身份识别，也不应依据单帧行为评价学生或教学质量。

## 识别类别

| ID | 标签 | 中文名称 |
|---:|---|---|
| 0 | write | 书写 |
| 1 | read | 阅读 |
| 2 | lookup | 抬头听讲 |
| 3 | turn_head | 转头 |
| 4 | raise_hand | 举手 |
| 5 | stand | 站立 |
| 6 | discuss | 讨论 |

## 实验结果

数据经审计后包含 400 张图片和 8,083 个目标框，并采用场景感知方式划分为 281 张训练图片、59 张验证图片和 60 张测试图片。

最终选用 YOLO26s@640。独立测试集结果如下：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.7373 | 0.7926 | 0.8221 | 0.6065 |

## 项目结构

```text
config/                 类别、训练和数据配置
data/splits/            可复现的数据划分清单
docs/                   数据审计、实验和设计文档
scripts/                数据准备、训练、评估、分析和推理脚本
src/campus_behavior/    核心图片及视频推理模块
report/                 课程论文 LaTeX 源文件
template/               贵州大学课程论文 LaTeX 模板
```

原始数据、训练权重、虚拟环境、训练缓存、预测视频和最终课程PDF不会在本仓库中重新发布。

## 环境安装

推荐使用 Python 3.11：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_env.py
```

## 数据准备

数据来自 [CCNUZFW/SCBehavior](https://github.com/CCNUZFW/SCBehavior)。本仓库不包含原始课堂图片和标注，请阅读 [DATASET_NOTICE.md](DATASET_NOTICE.md)，并通过数据获取脚本自行准备：

```bash
python scripts/fetch_scbehavior.py
python scripts/audit_scbehavior.py
python scripts/prepare_scbehavior_splits.py
```

## 训练与评估

```bash
python scripts/train_yolo.py --help
python scripts/evaluate_yolo.py --help
python scripts/compare_experiments.py --help
```

## 图片与视频推理

模型权重不随仓库发布。自行训练后可运行：

```bash
python scripts/predict_image.py --help
python scripts/predict_video.py --help
```

详细说明参见 [推理模块文档](docs/INFERENCE_MODULE.md)。

## 许可与引用

- 本项目原创代码采用 [GNU AGPL-3.0](LICENSE) 许可。
- Ultralytics 是 AGPL-3.0 依赖；商业使用请自行核对其许可要求。
- SCBehavior 上游未提供明确的数据集 LICENSE，本仓库因此不重新分发数据。
- 贵州大学课程论文模板源自带有 CC BY 4.0 声明的 UCAS LaTeX Template 修改版，相关声明保留在模板文件中。

使用数据集开展研究时，请按上游仓库要求引用其列出的相关论文。
