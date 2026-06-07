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

	# 如果用户之前没开 exist_ok 或者 Ultralytics 自动递增目录名，则兜底搜索
	pattern = os.path.join(project_dir, f"{run_name}*", "weights", "best.pt")
	matches = glob.glob(pattern)
	if not matches:
		return None

	matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return matches[0]


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="YOLO26 fine-tune on KITTI")
	parser.add_argument(
		"--base-dir",
		type=str,
		default=r"C:\Users\23978\Desktop\大三课程（设计）\智能汽车技术\Smart_Car_Proj",
		help="项目根目录",
	)
	parser.add_argument(
		"--weights",
		type=str,
		default="yolo26s.pt",
		help=(
			"预训练权重路径（支持相对 base-dir 的相对路径）。"
			"例如: yolo26s.pt 或 runs/.../weights/best.pt"
		),
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
	parser.add_argument("--device", type=str, default="cuda", help="训练设备")
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
		default=10,
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
	parser.add_argument(
		"--eval-only",
		action="store_true",
		default=False,
		help="跳过训练，直接使用指定权重进行测试集评估与预测可视化",
	)
	parser.add_argument(
		"--eval-weights",
		type=str,
		default="",
		help="eval-only 模式下使用的权重文件路径（支持相对 base-dir 的相对路径）",
	)
	parser.add_argument(
		"--eval-test-dir",
		type=str,
		default="",
		help="eval-only 模式下测试集目录（覆盖 yaml 中的 test 路径，支持绝对或相对 base-dir 的路径）",
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

	print("=== 1. 加载 YOLO26s 预训练模型 ===")
	weights_arg = str(args.weights).strip()
	weights_path = weights_arg
	if not os.path.isabs(weights_path):
		weights_path = os.path.join(base_dir, weights_path)
	print(f"预训练权重路径: {weights_path} | exists={os.path.exists(weights_path)}")
	if not os.path.exists(weights_path):
		raise FileNotFoundError(
			"未找到预训练权重文件。请将 yolo26s.pt 放到项目根目录，"
			"或用 --weights 指定正确路径。\n"
			f"当前: {weights_path}"
		)
	model = YOLO(weights_path)

	run_name = args.run_name

	if args.eval_only:
		# ========== eval-only 模式：跳过训练，直接加载指定权重进行评估 ==========
		print("\n=== [eval-only 模式] 跳过训练，直接加载指定权重 ===")
		eval_weights_arg = args.eval_weights.strip() if args.eval_weights else args.weights
		best_weights = eval_weights_arg
		if not os.path.isabs(best_weights):
			best_weights = os.path.join(base_dir, best_weights)
		print(f"评估权重路径: {best_weights} | exists={os.path.exists(best_weights)}")
		if not os.path.exists(best_weights):
			raise FileNotFoundError(
				"未找到评估权重文件。请用 --eval-weights 指定正确的权重路径。\n"
				f"当前: {best_weights}"
			)

		# 如果指定了 --eval-test-dir，动态生成临时 yaml 覆盖测试集路径
		if args.eval_test_dir.strip():
			eval_test_dir = args.eval_test_dir.strip()
			if not os.path.isabs(eval_test_dir):
				eval_test_dir = os.path.join(base_dir, eval_test_dir)
			print(f"使用自定义测试集目录: {eval_test_dir}")
			if not os.path.isdir(eval_test_dir):
				raise FileNotFoundError(f"测试集目录不存在: {eval_test_dir}")

			tmp_yaml = os.path.join(base_dir, "Src", "Train", "_eval_temp.yaml")
			with open(tmp_yaml, "w", encoding="utf-8") as f:
				f.write(f"path: {eval_test_dir}\n")
				f.write("train: .\n")
				f.write("val: .\n")
				f.write("test: .\n")
				f.write("nc: 2\n")
				f.write("names:\n")
				f.write("  0: Pedestrian\n")
				f.write("  1: Cyclist\n")
			eval_data_yaml = tmp_yaml
		else:
			eval_data_yaml = yaml_rel_path
	else:
		# ========== 正常训练模式 ==========
		print("\n=== 2. 开始在 KITTI 数据集上进行微调 (Fine-tune) ===")

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
		eval_data_yaml = yaml_rel_path

	# ========== 公共部分：测试集评估 + 可视化 ==========
	best_model = YOLO(best_weights)
	test_results_csv = None
	if args.eval_test:
		print("\n=== 4.1 使用测试集评估模型性能 ===")
		metrics = best_model.val(
			data=eval_data_yaml,
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

	test_images_dir_for_predict = str(dataset_info.test_images_dir)
	all_test_imgs = glob.glob(os.path.join(test_images_dir_for_predict, "*.png")) + glob.glob(
		os.path.join(test_images_dir_for_predict, "*.jpg")
	)
	# print("测试集目录为：", all_test_imgs)
	if not all_test_imgs:
		raise FileNotFoundError(f"测试集图片为空或路径不对: {test_images_dir_for_predict}")

	k = min(args.sample_k, len(all_test_imgs))
	sample_imgs = random.sample(all_test_imgs, k)

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
		test_results_csv=test_results_csv,
	)


if __name__ == "__main__":
	main()
