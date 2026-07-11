# Model and Dependency Notice

## 模型权重

本仓库不发布以下文件：

- Ultralytics预训练权重；
- 本项目训练得到的YOLO26权重；
- ONNX、TensorRT等导出模型。

使用者可依据脚本和配置自行下载合规的预训练权重并重新训练。

## Ultralytics

项目通过Python包调用Ultralytics。Ultralytics开源仓库采用GNU AGPL-3.0，并提供单独的企业许可方案。使用、修改或部署时应遵守其最新许可条款：<https://github.com/ultralytics/ultralytics>。

## 使用边界

系统输出是逐帧目标检测结果，不是跨帧去重人数，也不是学生身份、学习态度或教学质量的自动评价。真实校园部署前应完成数据授权、隐私评估、访问控制和保存期限设计。
