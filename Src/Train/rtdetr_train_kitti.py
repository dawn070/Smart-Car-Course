from __future__ import annotations

import argparse
import glob
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from train_log_utils import (
	append_training_log,
	collect_dataset_info,
	write_test_results_csv,
)

from plot_training_curves import plot_2x2_from_results_csv


random.seed(42)


def _find_latest_best_weight(project_dir: str, run_name: str) -> Optional[str]:
	"""尽量找到最新一次训练生成的 best.pt。"""
	# 优先使用固定目录
	candidate = os.path.join(project_dir, run_name, "weights", "best.pt")
	if os.path.exists(candidate):
		return candidate

	# 如果 Ultralytics 自动递增目录名，则兜底搜索
	pattern = os.path.join(project_dir, f"{run_name}*", "weights", "best.pt")
	matches = glob.glob(pattern)
	if not matches:
		return None

	matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return matches[0]


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="RT-DETR fine-tune on KITTI")
	parser.add_argument(
		"--base-dir",
		type=str,
		default=r"C:\\Users\\23978\\Desktop\\大三课程（设计）\\智能汽车技术\\Smart_Car_Proj",
		help="项目根目录",
	)
	parser.add_argument(
		"--weights",
		type=str,
		default="rtdetr-l.pt",
		help=(
			"预训练权重路径（支持相对 base-dir 的相对路径）。"
			"例如: rtdetr-l.pt 或 runs/.../weights/best.pt"
		),
	)
	parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
	parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
	parser.add_argument("--batch", type=int, default=8, help="批大小（RT-DETR-L 建议适当小一点）")
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
		"--optimizer",
		type=str,
		default="auto",
		help="优化器：auto/SGD/Adam/AdamW/RMSProp...（不区分大小写）",
	)
	parser.add_argument("--device", type=str, default="cuda", help="训练设备")
	parser.add_argument("--workers", type=int, default=0, help="DataLoader 进程数")
	parser.add_argument("--conf", type=float, default=0.3, help="推理置信度阈值")
	parser.add_argument(
		"--freeze",
		type=int,
		default=0,
		help=(
			"冻结网络前 N 层（0 表示不冻结）。"
			"用于微调时稳定训练、减少显存占用。"
		),
	)
	parser.add_argument(
		"--run-name",
		type=str,
		default="rtdetr_kitti_finetune",
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
		default="rtdetr_predict_imgs",
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


def _load_rtdetr(weights_path: str):
	"""Load RT-DETR model from Ultralytics.

	Ultralytics 新版本提供 RTDETR 类；为了兼容不同版本，这里做一次兜底。
	"""
	try:
		from ultralytics import RTDETR  # type: ignore

		return RTDETR(weights_path)
	except Exception:
		from ultralytics import YOLO  # type: ignore

		return YOLO(weights_path)


def main():
	args = _build_arg_parser().parse_args()
	base_dir = args.base_dir
	start_time = datetime.now()

	project_dir = args.project_dir or os.path.join(base_dir, "runs")
	log_path = Path(args.log_path) if args.log_path else (Path(base_dir) / "log.txt")

	# 切换到项目根目录并使用相对路径，避免底层对中文路径解析出错
	os.chdir(base_dir)
	yaml_rel_path = os.path.join("Src", "Train", "kitti.yaml")
	dataset_info = collect_dataset_info(base_dir, yaml_rel_path)

	print("=== 1. 加载 RT-DETR 预训练模型 ===")
	weights_arg = str(args.weights).strip()
	weights_path = weights_arg
	if not os.path.isabs(weights_path):
		weights_path = os.path.join(base_dir, weights_path)
	print(f"预训练权重路径: {weights_path} | exists={os.path.exists(weights_path)}")
	if not os.path.exists(weights_path):
		raise FileNotFoundError(
			"未找到预训练权重文件。请将 rtdetr-l.pt 放到项目根目录，"
			"或用 --weights 指定正确路径。\n"
			f"当前: {weights_path}"
		)

	model = _load_rtdetr(weights_path)

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
		freeze=int(args.freeze) if args.freeze is not None else 0,
		project=project_dir,
		name=run_name,
		exist_ok=True,
		device=args.device,
		workers=args.workers,
	)

	print("\n=== 3. 模型训练完成 ===")
	print(
		f"训练输出目录: {os.path.join(project_dir, run_name)} | "
		"包含 weights/best.pt、results.csv 等。"
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

	best_model = _load_rtdetr(best_weights)
	test_results_csv = None
	if args.eval_test:
		print("\n=== 4.1 使用测试集评估模型性能 ===")
		metrics = best_model.val(
			data=yaml_rel_path,
			split="test",
			project=project_dir,
			name=os.path.join(run_name, "test_eval"),
			exist_ok=True,
		)
		test_results_csv = Path(project_dir) / run_name / "test_eval" / "results.csv"
		write_test_results_csv(
			results_csv=test_results_csv,
			metrics=metrics,
			total_images=dataset_info.counts_test.images,
			total_instances=dataset_info.counts_test.instances,
		)

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
		save=True,
		conf=args.conf,
		project=project_dir,
		name=os.path.join(run_name, args.predict_name),
	)

	print(
		f"\n[任务完成] {k}张预测可视化图片已保存在: "
		f"{os.path.join(project_dir, run_name, args.predict_name)}"
	)

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
		test_results_csv=test_results_csv,
	)


if __name__ == "__main__":
	main()
