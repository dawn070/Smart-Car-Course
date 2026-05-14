from ultralytics import YOLO
import os
import glob
import random

random.seed(42)

def main():
    base_dir = r"C:\Users\23978\Desktop\大三课程（设计）\智能汽车技术\Smart_Car_Proj"
    yaml_path = os.path.join(base_dir, "Src", "Train", "kitti.yaml")
    
    # 建立保存运行结果的根目录
    project_dir = os.path.join(base_dir, "runs")
    
    print("=== 1. 加载 YOLOv10n 预训练模型 ===")
    # 第一次运行会自动从 Ultralytics 服务器下载 yolov10n.pt 权重文件
    weights_path = os.path.join(base_dir, "pretrain_weight", "yolov10n.pt")
    print(f"预训练权重路径: {weights_path} | exists={os.path.exists(weights_path)}")
    model = YOLO(weights_path) 
    
    print("\n=== 2. 开始在 KITTI 数据集上进行微调 (Fine-tune) ===")
    
    # 切换到当前目录并使用相对路径，避免 YOLO 底层对中文文件夹名解析出错
    os.chdir(base_dir)
    
    model.train(
        data=yaml_path,
        epochs=20,
        imgsz=640,
        batch=32,
        project=project_dir,
        name="yolov10_kitti_baseline",
        exist_ok=True, # 允许覆盖原来的文件夹，防止不断生成 kitti1, kitti2 等目录
        device="", # 自动检测环境，有GPU用GPU，没有用CPU
        workers=0  # 添加 workers=0，禁用多进程，避免 Windows 下静默崩溃
    )
    
    print("\n=== 3. 模型训练完成 ===")
    print("模型已自动在验证集上计算 mAP@0.5。")
    print(f"Loss曲线 (results.png) 和 PR曲线 均已自动保存在: {os.path.join(project_dir, 'yolov10_kitti_baseline')} 目录下。")
    
    print("\n=== 4. 开始提取 6 张图片进行检测可视化 ===")
    
    # 动态获取最新的模型权重目录（或者直接指定使用最新跑完的 yolov10_kitti1）
    best_weights = os.path.join(project_dir, "yolov10_kitti1", "weights", "best.pt")
    
    # 防止找不到报错，做一个检查
    if not os.path.exists(best_weights):
        best_weights = os.path.join(project_dir, "yolov10_kitti_baseline", "weights", "best.pt")

    best_model = YOLO(best_weights)
    
    # 从验证集中随机挑 6 张
    val_images_dir = os.path.join(base_dir, "kitti_yolo", "images", "val")
    all_val_imgs = glob.glob(os.path.join(val_images_dir, "*.png"))
    sample_imgs = random.sample(all_val_imgs, 6)
    
    # 推理预测并保存可视化图像
    best_model.predict(
        source=sample_imgs,
        save=True,          # 要求在图片上画框并保存
        conf=0.25,          # 置信度阈值：只画出大于大25%把握的框
        project=project_dir,
        name="yolov10_predict_6imgs" 
    )
    
    print(f"\n[任务完成] 6张预测可视化图片已保存在: {os.path.join(project_dir, 'yolov10_predict_6imgs')} 目录下！")

if __name__ == '__main__':
    main()
