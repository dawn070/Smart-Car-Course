from ultralytics import YOLO
import os
import glob
import random
from typing import Optional
import argparse


random.seed(42)


def _find_latest_best_weight(project_dir: str, run_name: str) -> Optional[str]:
	"""尽量找到最新一次训练生成的 best.pt。"""
	# 优先使用固定目录
	candidate = os.path.join(project_dir, run_name, "weights", "best.pt")
	if os.path.exists(candidate):
		return candidate

	# 如果用户之前没开 exist_ok 或者 Ultralytics 自动递增目录名，则兜底搜索
	pattern = os.path.join(project_dir, f"{run_name}*", "weights", "best.pt")
	matches = glob.glob(pattern)
	if not matches:
		return None

	matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return matches[0]


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="YOLO26n fine-tune on KITTI")
	parser.add_argument(
		"--base-dir",
		type=str,
		default=r"C:\Users\23978\Desktop\大三课程（设计）\智能汽车技术\Smart_Car_Proj",
		help="项目根目录",
	)
	parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
	parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
	parser.add_argument("--batch", type=int, default=16, help="批大小")
	parser.add_argument("--device", type=str, default="", help="训练设备")
	parser.add_argument("--workers", type=int, default=0, help="DataLoader 进程数")
	parser.add_argument("--conf", type=float, default=0.25, help="推理置信度阈值")
	parser.add_argument(
		"--run-name",
		type=str,
		default="yolo26_kitti_baseline",
		help="训练输出目录名",
	)
	parser.add_argument(
		"--project-dir",
		type=str,
		default="",
		help="runs 根目录（默认 base_dir/runs）",
	)
	parser.add_argument(
		"--predict-name",
		type=str,
		default="yolo26_predict_6imgs",
		help="预测输出目录名",
	)
	parser.add_argument(
		"--sample-k",
		type=int,
		default=6,
		help="抽样验证集图片数",
	)
	return parser


def main():
	args = _build_arg_parser().parse_args()
	base_dir = args.base_dir

	# 建立保存运行结果的根目录
	project_dir = args.project_dir or os.path.join(base_dir, "runs")

	# 切换到项目根目录并使用相对路径，避免 YOLO 底层对中文路径解析出错
	os.chdir(base_dir)
	yaml_rel_path = os.path.join("Src", "Train", "kitti.yaml")

	print("=== 1. 加载 YOLO26n 预训练模型 ===")
	weights_path = os.path.join(base_dir, "yolo26n.pt")
	print(f"预训练权重路径: {weights_path} | exists={os.path.exists(weights_path)}")
	model = YOLO(weights_path)

	print("\n=== 2. 开始在 KITTI 数据集上进行微调 (Fine-tune) ===")

	run_name = args.run_name
	model.train(
		data=yaml_rel_path,
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		project=project_dir,
		name=run_name,
		exist_ok=True,  # 允许覆盖原来的文件夹，防止不断生成 *1, *2 等目录
		device=args.device,      # 自动检测环境，有GPU用GPU，没有用CPU
		workers=args.workers      # 禁用多进程，避免 Windows 下静默崩溃
	)

	print("\n=== 3. 模型训练完成 ===")
	print("模型已自动在验证集上计算 mAP@0.5 等指标。")
	print(
		f"Loss/指标曲线 (results.png) 等已保存在: {os.path.join(project_dir, run_name)} 目录下。"
	)

	print("\n=== 4. 抽取验证集图片进行检测可视化 ===")
	best_weights = _find_latest_best_weight(project_dir, run_name)
	if not best_weights:
		raise FileNotFoundError(
			f"未找到 best.pt，请检查训练是否成功完成。期望在: {os.path.join(project_dir, run_name, 'weights')}"
		)

	best_model = YOLO(best_weights)

	val_images_dir = os.path.join(base_dir, "datasets", "kitti", "images", "val")
	all_val_imgs = glob.glob(os.path.join(val_images_dir, "*.png")) + glob.glob(
		os.path.join(val_images_dir, "*.jpg")
	)
	if not all_val_imgs:
		raise FileNotFoundError(f"验证集图片为空或路径不对: {val_images_dir}")

	k = min(args.sample_k, len(all_val_imgs))
	sample_imgs = random.sample(all_val_imgs, k)

	best_model.predict(
		source=sample_imgs,
		save=True,          # 在图片上画框并保存
		conf=args.conf,          # 置信度阈值
		project=project_dir,
		name=args.predict_name,
	)

	print(
		f"\n[任务完成] {k}张预测可视化图片已保存在: {os.path.join(project_dir, args.predict_name)} 目录下！"
	)


if __name__ == "__main__":
	main()