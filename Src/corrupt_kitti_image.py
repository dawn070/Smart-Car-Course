"""Batch-generate weather-style corruptions for KITTI image_2 folder.

This script reads all images under a KITTI `image_2` directory, applies a chosen
corruption using `albumentations`, and writes results to a new directory while
keeping image size and filename unchanged.

Example:
  python Src/corrupt_kitti_image.py --input datasets/kitti/images/test \
    --corruption rain --severity 3 --output datasets/kitti/corrupt_images/rain

Supported corruptions:
  - rain: 雨水效果
  - fog: 雾气效果
  - snow: 下雪效果
  - frost: 结冰效果
  - blur: 运动模糊
  - brightness: 亮度变化
  - contrast: 对比度变化
  - jpeg: JPEG压缩
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import albumentations. Please install it: "
        "pip install albumentations"
    ) from e


SUPPORTED_CORRUPTIONS = {
    "rain",
    "fog",
    "snow",
    "frost",
    "blur",
    "brightness",
    "contrast",
    "jpeg",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        candidates: Iterable[Path] = input_dir.rglob("*")
    else:
        candidates = input_dir.glob("*")

    files = [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def _get_transform(corruption_name: str, severity: int, **kwargs) -> A.Compose:
    """Create albumentations transform pipeline based on corruption type and severity."""
    # Normalize severity to [0, 1] range
    alpha = min(severity / 5.0, 1.0)
    
    transforms = []
    
    if corruption_name == "rain":
        # Get rain parameters from kwargs or use defaults
        drop_width = kwargs.get('rain_drop_width', 1)
        drop_length = kwargs.get('rain_drop_length', 20)
        drop_width_min = kwargs.get('rain_drop_width_min', 1)
        drop_width_max = kwargs.get('rain_drop_width_max', 2)
        blur_value = kwargs.get('rain_blur_value', 1)
        
        transforms.append(
            A.RandomRain(
                p=1.0,
                drop_width=drop_width,
                drop_length=drop_length,
                drop_width_range=(drop_width_min, drop_width_max),
                blur_value=blur_value
            )
        )
    elif corruption_name == "fog":
        transforms.append(
            A.RandomFog(p=1.0, fog_coef=alpha, alpha_coef=alpha)
        )
    elif corruption_name == "snow":
        transforms.append(
            A.RandomSnow(p=1.0, snow_point_range=(0.1, 0.3 + 0.2 * alpha), brightness_coeff=0.5 + 0.5 * alpha)
        )
    elif corruption_name == "frost":
        # Frost effect via brightness and contrast reduction
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=(-0.3 * alpha, 0), contrast_limit=(-0.3 * alpha, 0), p=1.0)
        )
    elif corruption_name == "blur":
        transforms.append(
            A.MotionBlur(blur_limit=int(3 + 5 * alpha), p=1.0)
        )
    elif corruption_name == "brightness":
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=(-0.3 * alpha, 0), contrast_limit=0, p=1.0)
        )
    elif corruption_name == "contrast":
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2 * alpha, 0.2 * alpha), p=1.0)
        )
    elif corruption_name == "jpeg":
        transforms.append(
            A.ImageCompression(quality_lower=int(100 - 50 * alpha), quality_upper=int(100 - 20 * alpha), p=1.0)
        )
    
    return A.Compose(transforms, keypoint_params=None)


def _apply_one(
    input_path: str,
    output_path: str,
    corruption_name: str,
    severity: int,
    overwrite: bool,
    **kwargs
) -> Tuple[str, bool, Optional[str]]:
    """Worker function: returns (filename, ok, error_message)."""
    try:
        in_path = Path(input_path)
        out_path = Path(output_path)

        if (not overwrite) and out_path.exists():
            return (in_path.name, True, None)

        # Read image using OpenCV (BGR format)
        image = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {in_path}")
        
        # Convert BGR to RGB for albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        transform = _get_transform(corruption_name, severity, **kwargs)
        corrupted = transform(image=image)["image"]
        
        # Convert back to BGR for saving
        corrupted = cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), corrupted)
        return (in_path.name, True, None)
    except Exception as ex:
        return (Path(input_path).name, False, f"{type(ex).__name__}: {ex}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="使用 albumentations 对 KITTI 图像批量生成指定天气/退化效果（支持进度条与多进程）。"
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="输入目录：包含图像的文件夹路径。",
    )
    p.add_argument(
        "--corruption",
        required=True,
        choices=sorted(SUPPORTED_CORRUPTIONS),
        help="退化/天气类型：rain、fog、snow、frost、blur、brightness、contrast、jpeg。",
    )
    p.add_argument(
        "--severity",
        type=int,
        default=3,
        help="退化强度等级：1~5（默认 3，越大越严重）。",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录：不填则自动命名为 <input>_<corruption>_s<severity>。",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描输入目录下的图片（包含子目录）。",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="并行进程数：默认 CPU 核心数 - 1；设为 1 则单进程。",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件（默认：存在则跳过）。",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到第一张失败立刻停止（默认：继续处理并最终汇总失败）。",
    )
    
    # Rain-specific parameters
    p.add_argument(
        "--rain-drop-width",
        type=int,
        default=1,
        help="雨滴宽度（像素，默认 1）。",
    )
    p.add_argument(
        "--rain-drop-length",
        type=int,
        default=20,
        help="雨滴长度（像素，默认 20）。",
    )
    p.add_argument(
        "--rain-drop-width-min",
        type=int,
        default=1,
        help="雨滴宽度最小值（默认 1）。",
    )
    p.add_argument(
        "--rain-drop-width-max",
        type=int,
        default=2,
        help="雨滴宽度最大值（默认 2）。",
    )
    p.add_argument(
        "--rain-blur-value",
        type=int,
        default=1,
        help="雨水模糊强度（默认 1，范围 1-3）。",
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_dir: Path = args.input
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    severity: int = int(args.severity)
    if not (1 <= severity <= 5):
        raise SystemExit("--severity must be within [1, 5].")

    corruption_name: str = args.corruption

    output_dir: Path
    if args.output is None:
        output_dir = input_dir.parent / f"{input_dir.name}_{corruption_name}_s{severity}"
    else:
        output_dir = args.output

    images = _list_images(input_dir, recursive=bool(args.recursive))
    if not images:
        raise SystemExit(f"No images found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare extra kwargs for corruption-specific parameters
    extra_kwargs = {}
    if corruption_name == "rain":
        extra_kwargs = {
            'rain_drop_width': args.rain_drop_width,
            'rain_drop_length': args.rain_drop_length,
            'rain_drop_width_min': args.rain_drop_width_min,
            'rain_drop_width_max': args.rain_drop_width_max,
            'rain_blur_value': args.rain_blur_value,
        }

    tasks: List[Tuple[str, str, str, int, bool]] = []
    for in_path in images:
        out_path = output_dir / in_path.name
        tasks.append((str(in_path), str(out_path), corruption_name, severity, bool(args.overwrite)))

    failures: List[Tuple[str, str]] = []

    # Multi-process for large batches; use workers=1 to run single-process.
    workers = int(args.workers)
    if workers <= 1:
        for in_path, out_path, corr, sev, overwrite in tqdm(tasks, desc="Corrupting", unit="img"):
            name, ok, err = _apply_one(in_path, out_path, corr, sev, overwrite, **extra_kwargs)
            if not ok:
                failures.append((name, err or "Unknown error"))
                if args.fail_fast:
                    break
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_apply_one, in_path, out_path, corr, sev, overwrite, **extra_kwargs) 
                      for in_path, out_path, corr, sev, overwrite in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Corrupting", unit="img"):
                name, ok, err = fut.result()
                if not ok:
                    failures.append((name, err or "Unknown error"))
                    if args.fail_fast:
                        for f in futures:
                            f.cancel()
                        break

    if failures:
        print(f"\nDone with failures: {len(failures)}/{len(tasks)}")
        for name, err in failures[:20]:
            print(f"  - {name}: {err}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        return 2

    print(f"\nDone. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
