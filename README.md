# OARENet

Windows 下可直接训练 DeepGlobe road extraction 数据集的 OARENet 版本。

## 已补全内容

- `train.py`
  - 默认训练 `80` 轮
  - 每轮都做验证
  - 每 `5` 轮保存一次 `epoch_xxx.pth`
  - 始终保存 `checkpoints/latest.pth`
  - 按验证集最高 `mIoU` 保存 `checkpoints/best.pth`
- TensorBoard 日志
  - 目录：`logs/OARENET_时间戳`
  - 记录 `loss`、`background_iou`、`road_iou`、`precision`、`recall`
  - 同时记录训练集和验证集
- DeepGlobe 数据读取
  - 兼容当前数据集目录
  - 自动处理 `val.txt` 中仍写成 `train/...` 的路径

## 数据集路径

默认路径：

```powershell
D:\project\pythonProject\Road_Identification\SAM2-UNet\deepglobe
```

目录结构应包含：

```text
deepglobe/
  train/
    data/
    seg/
  val/
    data/
    seg/
  train.txt
  val.txt
```

## 安装依赖

当前仓库不创建新的 Python 环境，只在你现有 Python 中安装项目依赖：

```powershell
./install_dependencies.ps1
```

或：

```powershell
python -m pip install -r requirements.txt
```

## 开始训练

直接使用默认参数：

```powershell
./run_train.ps1
```

等价命令：

```powershell
python train.py --data-root "D:\project\pythonProject\Road_Identification\SAM2-UNet\deepglobe" --epochs 80 --batch-size 4 --num-workers 4 --image-size 512
```

## TensorBoard

```powershell
tensorboard --logdir logs
```

## 说明

- 默认使用 CUDA，如果当前 Python 里的 PyTorch 没有 CUDA 支持，训练会直接报错。
- 默认启用 AMP 混合精度，适合 16GB 显存。
- 默认输入尺寸为 `512`，这是兼顾显存和训练稳定性的保守设置。
