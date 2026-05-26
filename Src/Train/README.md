# YOLOv10 KITTI 训练工具说明

本项目包含两个训练脚本，用于在 KITTI 数据集上训练 YOLOv10 目标检测模型（行人 / 自行车骑手）。

---

## 1. `train_yolov10.py` — 一键训练与验证脚本

**用途**：快速上手，一键完成 YOLOv10n 模型在 KITTI 数据集上的微调训练，并自动生成验证集可视化结果。

### 使用方式

```bash
cd <project_root>/Src/Train
python train_yolov10.py
```

无需额外参数，直接运行即可。

### 工作流程

1. 加载 `pretrain_weight/yolov10n.pt` 预训练权重（首次运行会自动下载）。
2. 使用 `kitti.yaml` 数据集配置进行 20 轮微调，图像尺寸 640×640，batch size 32。
3. 训练结果（loss 曲线、PR 曲线、mAP 等）自动保存至 `runs/yolov10_kitti_baseline/`。
4. 从验证集随机抽取 6 张图片，使用 `best.pt` 做目标检测可视化，结果保存至 `runs/yolov10_predict_6imgs/`。

### 关键配置项（需修改代码）

| 变量 | 说明 | 默认值 |
|---|---|---|
| `epochs` | 训练轮数 | 20 |
| `imgsz` | 输入图像尺寸 | 640 |
| `batch` | 批大小 | 32 |
| `device` | 设备（空字符串 = 自动） | `""` |
| `workers` | DataLoader 工作进程数 | 0（Windows 兼容） |

---

## 2. `ablation_train_or_test.py` — 消融实验训练 / 推理脚本

**用途**：通过命令行参数灵活控制训练超参、数据增强策略、模型结构消融（PSA / SCDown / C2fCIB / Neck 替换），支持训练或纯推理两种模式，适合做对比实验。

### 使用方式

```bash
cd <project_root>
python Src/Train/ablation_train_or_test.py [OPTIONS]
```

> **注意**：脚本内部会自动 `chdir` 到项目根目录，因此建议在项目根目录下运行，避免路径解析问题。

### 基础参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | 模型规模 | `yolov10n`（可选 `yolov10s`、`yolov10m`） |
| `--epochs` | 训练轮数 | 20 |
| `--imgsz` | 输入图像尺寸 | 640 |
| `--batch_size` | 批大小 | 32 |
| `--device` | 设备 ID 或 `cpu` | `0`（自动检测） |
| `--workers` | DataLoader workers | 0 |
| `--data_yaml` | 数据集 yaml 路径 | `Src/Train/kitti.yaml`（自动查找） |
| `--run_name` | 实验名称（输出目录名） | 自动生成（`ablation_{model}_{时间戳}`） |

### 数据增强参数

| 参数 | 说明 | 取值范围 |
|---|---|---|
| `--mosaic` | Mosaic 增强概率 | 0 ~ 1 |
| `--mixup` | MixUp 增强概率 | 0 ~ 1 |
| `--hsv_h` | HSV-H 色调变换系数 | 浮点数 |
| `--hsv_s` | HSV-S 饱和度变换系数 | 浮点数 |
| `--hsv_v` | HSV-V 明度变换系数 | 浮点数 |

### 结构消融参数

| 参数 | 说明 |
|---|---|
| `--model_yaml <path>` | 模型结构 YAML 路径（进行结构消融时必须指定） |
| `--psa on\|off` | 启用 / 关闭 PSA 注意力模块（off 时替换为 `nn.Identity`） |
| `--scdown on\|off` | 启用 / 关闭 SCDown 下采样模块（off 时替换为普通 `Conv`） |
| `--scdown_conv` | 等价于 `--scdown off`，将所有 SCDown 替换为 Conv |
| `--scdown_maxpool` | 将所有 SCDown 替换为 `MaxPool2d` 下采样 |
| `--c2fcib_lk on\|off` | 启用 / 关闭 C2fCIB 模块的大核卷积（large kernel） |
| `--replace_neck` | 是否替换 Neck 结构 |
| `--neck_yaml <path>` | 替换 Neck 时使用的 YAML 文件路径 |

> 参数冲突约束：`--scdown_conv`、`--scdown_maxpool`、`--scdown on/off` 三者互斥，不可同时使用。

### 推理参数

| 参数 | 说明 |
|---|---|
| `--skip_train` | 跳过训练，仅进行推理 |
| `--weights <path>` | 推理使用的权重文件（默认回退到预训练权重） |
| `--infer_source <path>` | 推理图片路径，支持单文件、目录、通配符、逗号分隔多个路径 |
| `--infer_conf` | 置信度阈值（默认 0.25） |
| `--infer_iou` | NMS IoU 阈值（默认 0.7） |
| `--use_nms on\|off` | 是否启用 NMS（默认 `off`） |
| `--infer_name` | 推理结果子目录名 |

### 示例

**1) 基础训练（等同于默认配置）**

```bash
python Src/Train/ablation_train_or_test.py --model yolov10n --epochs 20
```

**2) 关闭 PSA 模块进行结构消融训练**

```bash
python Src/Train/ablation_train_or_test.py \
    --model_yaml path/to/yolov10n.yaml \
    --psa off \
    --run_name no_psa_exp
```

**3) 关闭 SCDown（替换为普通 Conv）**

```bash
python Src/Train/ablation_train_or_test.py \
    --model_yaml path/to/yolov10n.yaml \
    --scdown off \
    --run_name no_scdown_exp
```

**4) 调整数据增强策略**

```bash
python Src/Train/ablation_train_or_test.py \
    --mosaic 0.5 \
    --mixup 0.2 \
    --hsv_h 0.015 \
    --run_name aug_exp
```

**5) 跳过训练，仅做推理**

```bash
python Src/Train/ablation_train_or_test.py \
    --skip_train \
    --weights runs/ablation/<exp_name>/weights/best.pt \
    --infer_source kitti_yolo/images/val
```

### 输出结构

```
runs/
├── yolov10_kitti_baseline/          # train_yolov10.py 输出
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── results.png
│   └── ...
├── yolov10_predict_6imgs/           # train_yolov10.py 可视化输出
└── ablation/
    ├── ablation_configs/            # 结构消融生成的模型 YAML
    └── <run_name>/                  # 每次实验的训练结果与可视化
        ├── weights/
        ├── infer_speed.txt          # 推理速度统计
        └── ...
```
