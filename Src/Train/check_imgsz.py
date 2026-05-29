"""检查训练时图片的实际尺寸——等比例缩放还是正方形拉伸"""
import os
import sys
from pathlib import Path

# 切到项目根目录（跟 yolo26_train.py 保持一致）
base_dir = r"C:\Users\23978\Desktop\大三课程（设计）\智能汽车技术\Smart_Car_Proj"
os.chdir(base_dir)
yaml_rel_path = os.path.join("Src", "Train", "kitti.yaml")

from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.cfg import get_cfg

# 1. 加载模型获取 stride
model = YOLO(os.path.join(base_dir, "yolo26n.pt"))
gs = max(int(model.model.stride.max()), 32)
print(f"模型 stride: {gs}")

# 2. 构建训练数据集（跟 model.train() 内部一样）
import yaml
with open(yaml_rel_path, encoding="utf-8") as f:
    data_dict = yaml.safe_load(f)
data_dict["yaml_file"] = yaml_rel_path
# 让 path 变为绝对路径
if data_dict.get("path"):
    data_dict["path"] = str(Path(base_dir) / data_dict["path"])

args_dict = dict(
    mode="train",
    imgsz=640,
    batch=4,
    single_cls=False,
    fraction=1.0,
    workers=0,
    rect=False,
    close_mosaic=0,
    task="detect",
    cache=None,
    data=yaml_rel_path,
    device="cpu",
)
args = get_cfg(overrides=args_dict)
args.stride = gs

train_ds = build_yolo_dataset(
    cfg=args,
    img_path=os.path.join(base_dir, "datasets/kitti/images/train"),
    batch=4,
    data=data_dict,
    mode="train",
    rect=False,
    stride=gs,
)

print(f"训练集大小: {len(train_ds)} 张")
print()

# 3. 查看 load_image 中间结果（过 Mosaic 之前）
print("=== 跳过 transforms，直接看 load_image 的输出 ===")
raw_img, raw_ori, raw_resized = train_ds.load_image(0)
print(f"  load_image 之前: ori_shape = {raw_ori}")
print(f"  load_image 之后: resized_shape = {raw_resized}")
print(f"  load_image 返回的 numpy shape: {raw_img.shape} (H×W×C)")
oh, ow = raw_ori
rh, rw = raw_resized
print(f"  原始宽高比: {ow/oh:.4f}, 缩放后宽高比: {rw/rh:.4f}")
print(f"  宽高比保持不变: {'✓ 是' if abs(ow/oh - rw/rh) < 0.01 else '✗ 否（被拉伸）'}")
print(f"  最长边={max(rh, rw)} == imgsz={640}: {'✓' if max(rh, rw) == 640 else '✗'}")

print()
print("=== 过 transforms 之后（最终 tensor）===")
# 3. 取前 5 张检查形状
for i in range(5):
    sample = train_ds[i]
    img = sample["img"]  # tensor, shape [C, H, W]
    h, w = img.shape[1:]
    ratio = h / w
    print(f"图片 {i}: tensor shape={list(img.shape)} (C×H×W), "
          f"宽高比={ratio:.4f}")
    # 检查像素：取中心行，看左右边缘是否有114 padding
    center_row = img[0, h//2, :]  # 第一通道中心行
    left_pixel = center_row[0].item()
    right_pixel = center_row[-1].item()
    print(f"         中心行左边缘像素={left_pixel:.0f}, 右边缘像素={right_pixel:.0f} "
          f"{'(有padding)' if left_pixel==114 or right_pixel==114 else ''}")

