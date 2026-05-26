from ultralytics import YOLO
import os
import glob
import random
from typing import Optional
import argparse
from datetime import datetime
from pathlib import Path

from train_log_utils import (
	append_training_log,
	capture_val_output,
	collect_dataset_info,
	extract_metrics_table,
)

from plot_training_curves import plot_2x2_from_results_csv


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
	parser.add_argument("--lr0", type=float, default=1e-4, help="初始学习率")
	parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率比例")
	parser.add_argument("--cos-lr", action="store_true", help="启用 cosine 学习率调度")
	parser.add_argument(
		"--warmup-epochs",
		type=float,
		default=3.0,
		help="warmup 轮数（允许小数）",
	)
	parser.add_argument(
		"--warmup-momentum",
		type=float,
		default=0.8,
		help="warmup 初始动量",
	)
	parser.add_argument(
		"--warmup-bias-lr",
		type=float,
		default=0.1,
		help="warmup bias 学习率",
	)
	parser.add_argument(
		"--optimizer",
		type=str,
		default="auto",
		help="优化器：auto/SGD/Adam/AdamW/RMSProp...（不区分大小写）",
	)
	parser.add_argument("--device", type=str, default="", help="训练设备")
	parser.add_argument("--workers", type=int, default=0, help="DataLoader 进程数")
	parser.add_argument("--conf", type=float, default=0.3, help="推理置信度阈值")
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
		"--log-path",
		type=str,
		default="",
		help="日志文件路径（默认 base_dir/log.txt）",
	)
	parser.add_argument(
		"--predict-name",
		type=str,
		default="yolo26_predict_imgs",
		help="预测输出目录名",
	)
	parser.add_argument(
		"--sample-k",
		type=int,
		default=6,
		help="抽样验证集图片数",
	)
	parser.add_argument(
		"--eval-test",
		action="store_true",
		default=True,
		help="训练结束后使用测试集评估模型性能",
	)
	parser.add_argument(
		"--no-eval-test",
		action="store_false",
		dest="eval_test",
		help="关闭测试集评估",
	)
	parser.add_argument(
		"--plot-curves",
		action="store_true",
		default=True,
		help="训练结束后自动绘制 2x2 曲线图",
	)
	parser.add_argument(
		"--no-plot-curves",
		action="store_false",
		dest="plot_curves",
		help="关闭训练曲线绘图",
	)
	parser.add_argument(
		"--plot-dpi",
		type=int,
		default=150,
		help="曲线图输出 DPI",
	)
	return parser


def main():
	args = _build_arg_parser().parse_args()
	base_dir = args.base_dir
	start_time = datetime.now()

	# 建立保存运行结果的根目录
	project_dir = args.project_dir or os.path.join(base_dir, "runs")
	log_path = Path(args.log_path) if args.log_path else (Path(base_dir) / "log.txt")

	# 切换到项目根目录并使用相对路径，避免 YOLO 底层对中文路径解析出错
	os.chdir(base_dir)
	yaml_rel_path = os.path.join("Src", "Train", "kitti.yaml")

	dataset_info = collect_dataset_info(base_dir, yaml_rel_path)

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
		optimizer=args.optimizer.lower() if isinstance(args.optimizer, str) else args.optimizer,
		lr0=args.lr0,
		lrf=args.lrf,
		cos_lr=args.cos_lr,
		warmup_epochs=args.warmup_epochs,
		warmup_momentum=args.warmup_momentum,
		warmup_bias_lr=args.warmup_bias_lr,
		project=project_dir,
		name=run_name,
		exist_ok=True,  # 允许覆盖原来的文件夹，防止不断生成 *1, *2 等目录
		device=args.device,      # 自动检测环境，有GPU用GPU，没有用CPU
		workers=args.workers      # 禁用多进程，避免 Windows 下静默崩溃
	)

	print("\n=== 3. 模型训练完成 ===")
	print("模型训练过程中会使用验证集进行评估，并自动保存最优权重 best.pt。")
	print(
		f"Loss/指标曲线 (results.png) 等已保存在: {os.path.join(project_dir, run_name)} 目录下。"
	)

	if args.plot_curves:
		run_dir = Path(project_dir) / run_name
		results_csv = run_dir / "results.csv"
		out_png = run_dir / "curves_2x2.png"
		try:
			if results_csv.exists():
				plot_2x2_from_results_csv(
					results_csv=results_csv,
					out_path=out_png,
					dpi=int(args.plot_dpi),
					show=False,
					title=f"Training Curves: {run_dir.as_posix()}",
				)
			else:
				print(f"[提示] 未找到 results.csv，跳过绘图: {results_csv}")
		except Exception as e:
			print(f"[提示] 绘图失败（不影响训练结果）：{e}")

	print("\n=== 4. 抽取验证集图片进行检测可视化 ===")
	best_weights = _find_latest_best_weight(project_dir, run_name)
	if not best_weights:
		raise FileNotFoundError(
			f"未找到 best.pt，请检查训练是否成功完成。期望在: {os.path.join(project_dir, run_name, 'weights')}"
		)

	best_model = YOLO(best_weights)
	val_stdout = ""
	metrics_table = ""
	if args.eval_test:
		print("\n=== 4.1 使用测试集评估模型性能 ===")
		val_stdout = capture_val_output(
			best_model.val,
			data=yaml_rel_path,
			split="test",
		)
		metrics_table = extract_metrics_table(val_stdout)

	val_images_dir_for_predict = str(dataset_info.val_images_dir)
	all_val_imgs = glob.glob(os.path.join(val_images_dir_for_predict, "*.png")) + glob.glob(
		os.path.join(val_images_dir_for_predict, "*.jpg")
	)
	if not all_val_imgs:
		raise FileNotFoundError(f"验证集图片为空或路径不对: {val_images_dir_for_predict}")

	k = min(args.sample_k, len(all_val_imgs))
	sample_imgs = random.sample(all_val_imgs, k)

	best_model.predict(
		source=sample_imgs,
		save=True,          # 在图片上画框并保存
		conf=args.conf,          # 置信度阈值
		project=project_dir,
		name=os.path.join(run_name, args.predict_name),
	)

	print(
		f"\n[任务完成] {k}张预测可视化图片已保存在: {os.path.join(project_dir, run_name, args.predict_name)} 目录下！"
	)

	# 追加日志（每次运行追加一段，排版尽量工整）
	end_time = datetime.now()
	append_training_log(
		log_path=log_path,
		start_time=start_time,
		end_time=end_time,
		dataset=dataset_info,
		args=vars(args),
		project_dir=project_dir,
		run_name=run_name,
		best_weights=best_weights,
		eval_test_enabled=bool(args.eval_test),
		test_metrics_table=metrics_table,
		test_raw_output=val_stdout,
	)


if __name__ == "__main__":
	main()