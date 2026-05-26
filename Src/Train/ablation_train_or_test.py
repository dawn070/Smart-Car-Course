import argparse
import glob
import os
import random
import time
from typing import Any, Dict, Optional

import yaml
from ultralytics import YOLO

random.seed(0)


def _base_dir() -> str:
    # 兼容中文路径：统一在项目根目录执行
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_weights_or_model(model_name: str, base_dir: str) -> str:
    # 优先使用本地预训练权重；如果不存在则交给 Ultralytics 自动下载
    local_weight = os.path.join(base_dir, "pretrain_weight", f"{model_name}.pt")
    return local_weight if os.path.exists(local_weight) else f"{model_name}.pt"


def _set_if_present(cfg: Dict[str, Any], keys, value) -> bool:
    for k in keys:
        if k in cfg:
            cfg[k] = value
            return True
    return False


def _replace_module_in_layers(
    cfg: Dict[str, Any],
    target_module: str,
    new_module: str,
    new_args,
) -> int:
    """在 backbone/head 的层定义中替换模块（保持层数量不变，避免索引错位）。"""
    replaced = 0
    for section in ("backbone", "head"):
        layers = cfg.get(section)
        if not isinstance(layers, list):
            continue
        for i, layer in enumerate(layers):
            # layer 格式: [from, repeats, module, args]
            if isinstance(layer, list) and len(layer) >= 4 and layer[2] == target_module:
                layer[2] = new_module
                layer[3] = new_args
                layers[i] = layer
                replaced += 1
    return replaced


def _apply_arch_ablation(
    model_yaml_path: str,
    output_dir: str,
    run_name: str,
    psa: Optional[bool],
    scdown: Optional[bool],
    scdown_maxpool: Optional[bool],
    c2fcib_lk: Optional[bool],
    replace_neck: bool,
    neck_yaml_path: Optional[str],
) -> str:
    # 读取原始模型 YAML，并根据开关修改
    with open(model_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # PSA/SCDown 的结构消融：Ultralytics 的 v10 yaml 通常是“显式层定义”，没有 enable 字段。
    # 因此这里对 backbone/head 的层列表做替换。
    # 注意：不要直接删除层，否则后续 head 中引用的索引（如 [-1, 10]）会错位。
    if psa is not None:
        if psa is False:
            n = _replace_module_in_layers(cfg, "PSA", "nn.Identity", [])
            if n == 0:
                print("[警告] 未在 backbone/head 的层列表中找到 PSA，无法关闭 PSA。")
            else:
                print(f"[结构消融] PSA off：已将 {n} 个 PSA 层替换为 nn.Identity（保持索引不变）。")
        else:
            # psa=True 时不需要做任何事（yaml 原本就包含 PSA 层）
            pass

    if scdown is not None:
        if scdown is False:
            # 将 SCDown 替换为普通 Conv 下采样（保持 stride=2）。
            # SCDown args 通常是 [c2, k, s]。
            replaced = 0
            for section in ("backbone", "head"):
                layers = cfg.get(section)
                if not isinstance(layers, list):
                    continue
                for i, layer in enumerate(layers):
                    if isinstance(layer, list) and len(layer) >= 4 and layer[2] == "SCDown":
                        old_args = layer[3] if isinstance(layer[3], list) else []
                        c2 = old_args[0] if len(old_args) >= 1 else None
                        k = old_args[1] if len(old_args) >= 2 else 3
                        s = old_args[2] if len(old_args) >= 3 else 2
                        if c2 is None:
                            continue
                        layer[2] = "Conv"
                        layer[3] = [c2, k, s]
                        layers[i] = layer
                        replaced += 1

            if replaced == 0:
                print("[警告] 未在 backbone/head 的层列表中找到 SCDown，无法关闭 SCDown。")
            else:
                print(f"[结构消融] SCDown off：已将 {replaced} 个 SCDown 层替换为 Conv（保持索引不变）。")
        else:
            pass

    if scdown_maxpool:
        # 将 SCDown 替换为 MaxPool 下采样（保持 stride=2）。
        # 使用 nn.MaxPool2d，args: [k, s] 或 [k, s, p]
        replaced = 0
        for section in ("backbone", "head"):
            layers = cfg.get(section)
            if not isinstance(layers, list):
                continue
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and len(layer) >= 4 and layer[2] == "SCDown":
                    old_args = layer[3] if isinstance(layer[3], list) else []
                    # SCDown args 通常为 [c2, k, s]，这里保持 k/s，并补齐 padding
                    # 经验上 k=3,s=2,p=1 可避免多尺度特征 concat 时出现 14/15 这种对不齐
                    k = old_args[1] if len(old_args) >= 2 else 3
                    s = old_args[2] if len(old_args) >= 3 else 2
                    p = (k // 2) if isinstance(k, int) and (k % 2 == 1) else 0
                    layer[2] = "nn.MaxPool2d"
                    layer[3] = [k, s, p]
                    layers[i] = layer
                    replaced += 1

        if replaced == 0:
            print("[警告] 未在 backbone/head 的层列表中找到 SCDown，无法替换为 MaxPool。")
        else:
            print(f"[结构消融] SCDown -> MaxPool：已替换 {replaced} 个 SCDown 层（保持索引不变）。")

    if c2fcib_lk is not None:
        # C2fCIB args 通常为 [c2, shortcut, lk]
        updated = 0
        for section in ("backbone", "head"):
            layers = cfg.get(section)
            if not isinstance(layers, list):
                continue
            for i, layer in enumerate(layers):
                if isinstance(layer, list) and len(layer) >= 4 and layer[2] == "C2fCIB":
                    old_args = layer[3] if isinstance(layer[3], list) else []
                    # 保持原参数数量与顺序
                    if len(old_args) >= 3:
                        old_args[2] = c2fcib_lk
                    elif len(old_args) == 2:
                        old_args.append(c2fcib_lk)
                    elif len(old_args) == 1:
                        old_args.extend([True, c2fcib_lk])
                    else:
                        old_args = [1024, True, c2fcib_lk]
                    layer[3] = old_args
                    layers[i] = layer
                    updated += 1
        if updated == 0:
            print("[警告] 未在 backbone/head 的层列表中找到 C2fCIB，无法设置 lk 参数。")
        else:
            print(f"[结构配置] C2fCIB lk={'on' if c2fcib_lk else 'off'}：已更新 {updated} 个 C2fCIB 层参数。")

    # 可选替换 Neck 结构：将 neck_yaml 中的 neck/head 覆盖到当前模型
    if replace_neck:
        if not neck_yaml_path:
            print("[警告] 已启用 Neck 替换，但未提供 --neck_yaml，已跳过。")
        else:
            with open(neck_yaml_path, "r", encoding="utf-8") as f:
                neck_cfg = yaml.safe_load(f)

            # 优先从 neck_yaml 提取 neck/head，否则用整个 dict
            neck_block = neck_cfg.get("neck") if isinstance(neck_cfg, dict) else None
            if neck_block is None and isinstance(neck_cfg, dict):
                neck_block = neck_cfg.get("head")
            if neck_block is None:
                neck_block = neck_cfg

            if isinstance(cfg, dict):
                if "neck" in cfg:
                    cfg["neck"] = neck_block
                elif "head" in cfg:
                    cfg["head"] = neck_block
                else:
                    print("[警告] model.yaml 中未找到 neck/head 字段，已跳过 Neck 替换。")

    # 将修改后的 yaml 输出到 runs/ablation/ablation_configs 便于记录
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_name}_model.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_path


def _resolve_val_images_dir(data_yaml_path: str, base_dir: str) -> Optional[str]:
    try:
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        if not isinstance(data_cfg, dict):
            return None
        root = data_cfg.get("path")
        val_rel = data_cfg.get("val")
        if not root or not val_rel:
            return None
        # 兼容两种写法：
        # 1) path 相对 data.yaml 所在目录（Ultralytics 常见）
        # 2) path 相对项目根目录（本项目的 kitti.yaml 目前就是这种写法）
        if os.path.isabs(root):
            root_abs = root
        else:
            yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
            cand1 = os.path.abspath(os.path.join(yaml_dir, root))
            cand2 = os.path.abspath(os.path.join(base_dir, root))
            root_abs = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else cand1)
        return os.path.join(root_abs, val_rel)
    except Exception:
        return None


def _collect_images_from_source(source_str: str, base_dir: str) -> list:
    """解析推理图片来源：文件/目录/通配符/逗号分隔。返回绝对路径列表。"""
    if not source_str:
        return []

    items = [s.strip() for s in source_str.split(",") if s.strip()]
    results = []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    for item in items:
        # 相对路径统一按项目根目录解析
        path = item
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.join(base_dir, path))

        # 通配符
        if any(ch in path for ch in ["*", "?", "[", "]"]):
            for ext_path in glob.glob(path):
                if os.path.isfile(ext_path):
                    results.append(os.path.abspath(ext_path))
            continue

        # 目录
        if os.path.isdir(path):
            for ext in exts:
                results.extend(
                    [
                        os.path.abspath(p)
                        for p in glob.glob(os.path.join(path, "**", ext), recursive=True)
                    ]
                )
            continue

        # 单文件
        if os.path.isfile(path):
            results.append(os.path.abspath(path))

    # 去重并保持稳定顺序
    seen = set()
    uniq = []
    for p in results:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 + KITTI 真正消融实验训练脚本")

    # 基础超参
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--model", type=str, default="yolov10n", choices=["yolov10n", "yolov10s", "yolov10m"], help="模型规模")
    parser.add_argument("--run_name", type=str, default=None, help="实验名称（输出目录名）")
    parser.add_argument("--data_yaml", type=str, default=None, help="数据集 yaml 路径")
    parser.add_argument("--model_yaml", type=str, default=None, help="模型结构 yaml 路径（用于结构消融）")

    # 数据增强超参
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic 概率（0~1）")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp 概率（0~1）")
    parser.add_argument("--hsv_h", type=float, default=None, help="HSV-H 变换系数")
    parser.add_argument("--hsv_s", type=float, default=None, help="HSV-S 变换系数")
    parser.add_argument("--hsv_v", type=float, default=None, help="HSV-V 变换系数")

    # 结构消融相关
    parser.add_argument("--psa", type=str, choices=["on", "off"], default=None, help="是否启用 PSA 模块")
    parser.add_argument("--scdown", type=str, choices=["on", "off"], default=None, help="是否启用 SCDown")
    parser.add_argument(
        "--scdown_conv",
        action="store_true",
        help="将所有 SCDown 层替换为 Conv 下采样层（等价于 --scdown off，更直观）",
    )
    parser.add_argument(
        "--scdown_maxpool",
        action="store_true",
        help="将所有 SCDown 层替换为 MaxPool 下采样层",
    )
    parser.add_argument(
        "--c2fcib_lk",
        type=str,
        choices=["on", "off"],
        default=None,
        help="设置 C2fCIB 的 lk 参数（大核卷积开关）",
    )
    parser.add_argument("--replace_neck", action="store_true", help="是否替换 Neck 结构")
    parser.add_argument("--neck_yaml", type=str, default=None, help="Neck 结构 yaml 路径")

    # 其他
    parser.add_argument("--device", type=str, default="0", help="设备，如 0 或 cpu，默认自动检测")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers（Windows 推荐 0）")
    parser.add_argument("--use_nms", type=str, choices=["on", "off"], default="off", help="是否启用 NMS（用于推理可视化）")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练，仅进行推理")
    parser.add_argument("--weights", type=str, default=None, help="推理权重路径（best.pt/last.pt）")
    parser.add_argument("--infer_source", type=str, default=None, help="推理图片/目录/通配符，或逗号分隔多个路径")
    parser.add_argument("--infer_conf", type=float, default=0.25, help="推理置信度阈值")
    parser.add_argument("--infer_iou", type=float, default=0.7, help="NMS 的 IoU 阈值（越小抑制越强）")
    parser.add_argument("--infer_name", type=str, default=None, help="推理结果子目录名")

    args = parser.parse_args()

    # 参数冲突校验：--scdown_conv / --scdown_maxpool / --scdown on/off 不应同时使用
    if args.scdown_conv and args.scdown is not None:
        raise SystemExit("参数冲突：请不要同时使用 --scdown_conv 与 --scdown on/off。")
    if args.scdown_maxpool and args.scdown is not None:
        raise SystemExit("参数冲突：请不要同时使用 --scdown_maxpool 与 --scdown on/off。")
    if args.scdown_conv and args.scdown_maxpool:
        raise SystemExit("参数冲突：请不要同时使用 --scdown_conv 与 --scdown_maxpool。")

    base_dir = _base_dir()
    data_yaml = args.data_yaml or os.path.join(base_dir, "Src", "Train", "kitti.yaml")
    project_dir = os.path.join(base_dir, "runs", "ablation")

    run_name = args.run_name or f"ablation_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"

    # 切换到项目根目录，避免 YOLO 底层对中文路径解析异常
    os.chdir(base_dir)

    if args.skip_train:
        print("=== 跳过训练：直接推理 ===")

        infer_weights = args.weights
        if infer_weights:
            infer_weights = os.path.abspath(infer_weights)
        else:
            infer_weights = _resolve_weights_or_model(args.model, base_dir)
            print(f"[提示] 未提供 --weights，已回退到预训练权重: {infer_weights}")

        if not os.path.exists(infer_weights):
            raise SystemExit(f"未找到权重文件: {infer_weights}")

        if not args.infer_source:
            raise SystemExit("请提供 --infer_source 指定推理图片/目录/通配符。")

        infer_imgs = _collect_images_from_source(args.infer_source, base_dir)
        if not infer_imgs:
            raise SystemExit("未找到可推理的图片，请检查 --infer_source。")

        infer_name = args.infer_name or f"{run_name}_infer"
        use_nms = args.use_nms == "on"
        print("非极大值抑制（NMS）已启用。" if use_nms else "非极大值抑制（NMS）已禁用。")

        model = YOLO(infer_weights)

        t0 = time.perf_counter()
        model.predict(
            source=infer_imgs,
            save=True,
            conf=args.infer_conf,
            iou=args.infer_iou,
            nms=use_nms,
            project=project_dir,
            name=infer_name,
        )
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) / max(1, len(infer_imgs)) * 1000.0
        speed_path = os.path.join(project_dir, infer_name, "infer_speed.txt")
        with open(speed_path, "w", encoding="utf-8") as f:
            f.write(f"sample_size: {len(infer_imgs)}\n")
            f.write(f"total_seconds: {t1 - t0:.6f}\n")
            f.write(f"avg_ms_per_image: {avg_ms:.3f}\n")
            f.write(f"nms: {use_nms}\n")
            f.write(f"iou: {args.infer_iou}\n")

        print(f"\n[任务完成] 推理结果已保存在: {os.path.join(project_dir, infer_name)}")
        print(f"推理速度已保存: {speed_path}")
        return

    print("=== 1. 解析模型与结构消融配置 ===")

    # 结构消融优先使用 model_yaml；否则使用预训练权重
    model_source = None
    ablation_yaml_dir = os.path.join(project_dir, "ablation_configs")
    psa_flag = None if args.psa is None else (args.psa == "on")
    # scdown_flag: True=保持/启用 SCDown；False=用 Conv 替换 SCDown；None=不做处理
    if args.scdown_conv:
        scdown_flag = False
    else:
        scdown_flag = None if args.scdown is None else (args.scdown == "on")
    c2fcib_lk_flag = None if args.c2fcib_lk is None else (args.c2fcib_lk == "on")

    if args.model_yaml:
        model_yaml_path = os.path.abspath(args.model_yaml)
        model_source = _apply_arch_ablation(
            model_yaml_path,
            ablation_yaml_dir,
            run_name,
            psa=psa_flag,
            scdown=scdown_flag,
            scdown_maxpool=args.scdown_maxpool,
            c2fcib_lk=c2fcib_lk_flag,
            replace_neck=args.replace_neck,
            neck_yaml_path=args.neck_yaml,
        )
        print(f"结构消融 YAML: {model_source}")
    else:
        model_source = _resolve_weights_or_model(args.model, base_dir)
        print(f"预训练权重路径: {model_source} | exists={os.path.exists(model_source)}")
        if psa_flag is not None or scdown_flag is not None or args.scdown_maxpool or c2fcib_lk_flag is not None or args.replace_neck:
            print("[提示] 未提供 --model_yaml，结构消融开关将不生效。")

    print("=== 2. 加载模型并开始训练 ===")

    model = YOLO(model_source)

    train_kwargs = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
    )

    # 仅在用户显式指定时传入增强参数，避免覆盖默认值
    if args.mosaic is not None:
        train_kwargs["mosaic"] = args.mosaic
    if args.mixup is not None:
        train_kwargs["mixup"] = args.mixup
    if args.hsv_h is not None:
        train_kwargs["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        train_kwargs["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        train_kwargs["hsv_v"] = args.hsv_v

    model.train(**train_kwargs)

    print("\n=== 3. 模型训练完成 ===")
    print(f"结果已保存在: {os.path.join(project_dir, run_name)}")

    print("\n=== 4. 使用 best.pt 做验证集可视化 ===")
    best_weights = os.path.join(project_dir, run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(project_dir, run_name, "weights", "last.pt")

    if not os.path.exists(best_weights):
        print("[警告] 未找到 best.pt/last.pt，跳过可视化。")
        return

    best_model = YOLO(best_weights)

    if args.infer_source:
        sample_imgs = _collect_images_from_source(args.infer_source, base_dir)
        if not sample_imgs:
            print("[警告] 未找到推理图片，跳过可视化。")
            return
    else:
        val_images_dir = _resolve_val_images_dir(data_yaml, base_dir)
        if not val_images_dir or not os.path.exists(val_images_dir):
            print("[警告] 未找到验证集图像目录，跳过可视化。")
            return

        all_val_imgs = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            all_val_imgs.extend(glob.glob(os.path.join(val_images_dir, ext)))

        if not all_val_imgs:
            print("[警告] 验证集图像为空，跳过可视化。")
            return

        sample_size = min(6, len(all_val_imgs))
        sample_imgs = random.sample(all_val_imgs, sample_size)

    sample_size = len(sample_imgs)
    use_nms = args.use_nms == "on"

    # 统计推理速度（平均每张图）
    t0 = time.perf_counter()
    best_model.predict(
        source=sample_imgs,
        save=True,
        conf=args.infer_conf,
        iou=args.infer_iou,
        nms=use_nms,
        project=project_dir,
        name=args.infer_name or f"{run_name}_predict_{sample_size}imgs",
    )
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / max(1, sample_size) * 1000.0
    infer_name = args.infer_name or f"{run_name}_predict_{sample_size}imgs"
    speed_path = os.path.join(project_dir, run_name, "infer_speed.txt")
    with open(speed_path, "w", encoding="utf-8") as f:
        f.write(f"sample_size: {sample_size}\n")
        f.write(f"total_seconds: {t1 - t0:.6f}\n")
        f.write(f"avg_ms_per_image: {avg_ms:.3f}\n")
        f.write(f"nms: {use_nms}\n")
        f.write(f"iou: {args.infer_iou}\n")

    print(
        f"\n[任务完成] 预测可视化图片已保存在: {os.path.join(project_dir, infer_name)}"
    )
    print(f"推理速度已保存: {speed_path}")


if __name__ == "__main__":
    main()
