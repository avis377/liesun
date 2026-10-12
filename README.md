# liesun
# YOLO 装甲板模型视频推理

## 1. 项目完成的功能

本项目基于 Ultralytics YOLO26n-Pose 装甲板模型 `best.pt`，实现对自拍摄视频的逐帧推理，具体功能包括：

- 加载训练好的 `best.pt` 模型权重。
- 读取指定视频文件，逐帧执行推理。
- 当检测到装甲板目标时，在画面上绘制：
  - 检测框
  - 类别名称
  - 置信度
  - 该目标的 4 个关键点（装甲板四角点，不同颜色区分）
- 将处理后的视频保存为与原视频内容对应、可正常播放的 MP4 文件。
- 支持实时窗口预览，按 `q` 键可提前退出。

---

## 2. Python 版本、主要依赖及 CPU/GPU 说明

### Python 版本
- 推荐 Python 3.10（其他 3.8–3.11 版本通常也可运行）

### 主要依赖
| 依赖 | 用途 |
|------|------|
| ultralytics | 加载 YOLO 模型并执行推理 |
| opencv-python | 视频读取、绘制、显示、保存 |
| numpy | 数组与数值计算 |
| torch / torchvision | PyTorch 运行时（ultralytics 依赖） |
| pathlib | 路径检查（Python 标准库） |

### CPU / GPU
- 脚本会自动检测 CUDA。若机器有可用的 NVIDIA GPU，Ultralytics 会优先使用 GPU 推理；否则使用 CPU。
- 如需显式指定设备，可在 `model.predict()` 中加入参数：
  - GPU：`device=0`
  - CPU：`device='cpu'`
- 本项目在 `cpu` 上测试运行。

---

## 3. 模型权重的准备方式与放置位置

### 权重准备
- 模型文件名为 `best.pt`，由 YOLO26n 装甲板训练流程导出。
- 请从训练输出目录（如 `runs/pose/train/weights/best.pt`）获取该文件。

### 放置位置
将 `best.pt` 与推理脚本 `infer.py` 放在**同一目录**下，例如：
yolo_project/
├── infer.py
├── best.pt
└── test.mp4
如果权重放在其他路径，请修改 `infer.py` 中的：`MODEL_PATH = "best.pt"`

## 4. 依赖安装步骤、实际运行命令和结果保存位置。

### 创建并激活 Conda 环境
- conda create -n yolo_env python=3.10 -y
- conda activate yolo_env

### 安装 PyTorch
- gpu: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuxxx
- cpu：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### 安装其他依赖
- pip install ultralytics opencv-python numpy

### 准备视频
将待推理视频重命名为 test.mp4，或修改`infer.py `中的：`VIDEO_PATH = "test.mp4"`

### 运行推理
`python infer.py`

推理结果默认保存为当前目录下的：
`output_infer.mp4`

该文件与原视频帧率、分辨率一致，可正常播放。若需修改输出路径，调整：
`OUTPUT_PATH = "output_infer.mp4"`

## 5. 视频拍摄地点、时长及场景说明
拍摄地点：D414实验室

视频时长：约11s

## 6.关键点的代表性结果截图

## 7. 实际识别表现
在光照充足、目标清晰时，检测框与关键点定位准确，置信度普遍高于 0.8
但快速运动时帧间稳定性略有下降

- 漏检	可能原因为置信度阈值过高、目标过小、光照不足，可尝试降低 CONF_THRESHOLD（如 0.15）；增加训练数据；调整 IMG_SIZE
- 误检	可能原因为背景干扰、相似目标，可尝试提高 CONF_THRESHOLD（如 0.5）；增加负样本训练
