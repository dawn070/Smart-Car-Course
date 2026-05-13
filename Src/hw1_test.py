import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ========= 1. 解析calib文件 =========
def load_calib(calib_path):
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # 仅将数值字段转换为 float 数组；如 calib_time 这类字符串字段原样保存
                tokens = value.split()
                try:
                    calib[key] = np.array([float(x) for x in tokens], dtype=np.float64)
                except ValueError:
                    calib[key] = value
    return calib


def get_focal_baseline(calib, left_cam_id="02", right_cam_id="03"):
    # 左右投影矩阵（KITTI 风格：P_rect_02 / P_rect_03 对应左右彩色相机）
    P_left = calib[f"P_rect_{left_cam_id}"].reshape(3, 4)
    P_right = calib[f"P_rect_{right_cam_id}"].reshape(3, 4)

    f_left = float(P_left[0, 0])
    f_right = float(P_right[0, 0])
    f = (f_left + f_right) / 2.0

    # 一般情况下 Tx = -f * B（右相机），但有些数据左相机 Tx 也非 0，
    # 用两者 Tx 差来推 baseline 更稳妥。
    B = abs(float(P_right[0, 3]) - float(P_left[0, 3])) / f

    return f, B


# ========= 2. 主流程 =========
def main():

    # 当前目录
    base_dir = "./"

    left_path = os.path.join(base_dir, "homework1_dataset/000030_10_L.png")
    right_path = os.path.join(base_dir, "homework1_dataset/000030_10_R.png")
    calib_path = os.path.join(base_dir, "homework1_dataset/000030.txt")

    # 从标定文件名提取基础名称（如 000000）用于输出文件命名
    calib_basename = os.path.splitext(os.path.basename(calib_path))[0]

    # 你测量/确认的左右彩色相机编号与基线
    left_cam_id = "02"
    right_cam_id = "03"
    measured_baseline_m = 0.54

    # 读取图像（灰度）
    imgL = cv2.imread(left_path, 0)
    imgR = cv2.imread(right_path, 0)

    if imgL is None or imgR is None:
        print("图像读取失败，请检查路径！")
        return

    # ========= 3. 计算视差 =========
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, # 视差最小值，通常为0
        numDisparities=16 * 6, # 视差范围，必须是16的倍数，这里设置为96（即最大视差为95）
        blockSize=5, # 匹配块大小，通常为3~11的奇数，较大值在纹理较少的区域表现更好
        P1=8 * 3 * 5**2, # 惩罚项1，控制视差变化较小的像素对匹配的惩罚，通常为8*通道数*blockSize^2
        P2=32 * 3 * 5**2, # 惩罚项2，控制视差变化较大的像素对匹配的惩罚，通常为P1的4倍
        disp12MaxDiff=1, # 左右视差图的一致性检查最大允许差异，通常为1
        uniquenessRatio=10, # 唯一性比率，控制最佳匹配与次佳匹配之间的最小差异百分比，较大值在纹理较少的区域表现更好
        speckleWindowSize=100, # 斑点过滤窗口大小，控制视差图中斑点噪声的过滤，较大值在纹理较少的区域表现更好
        speckleRange=32 # 斑点过滤视差范围，控制视差图中斑点噪声的过滤，较大值在纹理较少的区域表现更好
    )

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # SGBM 左侧存在无效匹配区：需要至少 x >= max_disparity 才能在右图找到对应点
    max_disparity = int(stereo.getMinDisparity() + stereo.getNumDisparities())
    crop_x0 = max(0, min(max_disparity, disparity.shape[1]))

    # ========= 4. 读取标定 =========
    calib = load_calib(calib_path)
    f, B_from_calib = get_focal_baseline(calib, left_cam_id=left_cam_id, right_cam_id=right_cam_id)
    B = measured_baseline_m

    print(f"焦距 f = {f}")
    print(f"基线 B(由标定推算) = {B_from_calib}")
    print(f"基线 B(实测采用) = {B}")

    # ========= 5. 计算深度 =========
    disp = disparity.copy()
    disp[disp <= 0] = 0.1  # 防止除0

    depth = f * B / disp

    # 裁剪左侧无效区（用于显示与保存）
    disparity_vis = disparity[:, crop_x0:]
    depth_vis = depth[:, crop_x0:]

    # ========= 6. 可视化 =========
    plt.figure("Disparity")
    plt.imshow(disparity_vis, cmap='gray')
    plt.colorbar()
    plt.title("Disparity Map")

    plt.figure("Depth")
    plt.imshow(depth_vis, cmap='jet')
    plt.colorbar()
    plt.title("Depth Map")

    plt.show()

    # ========= 7. 保存 =========
    disp_max = float(np.max(disparity_vis))
    depth_max = float(np.max(depth_vis[np.isfinite(depth_vis)])) if np.any(np.isfinite(depth_vis)) else 0.0

    # 构造输出文件名：{txt文件名}_{depth/disparity}.png
    depth_filename = f"{calib_basename}_depth.png"
    disparity_filename = f"{calib_basename}_disparity.png"

    if disp_max > 0:
        cv2.imwrite(disparity_filename, (disparity_vis / disp_max * 255).astype(np.uint8))
    else:
        cv2.imwrite(disparity_filename, np.zeros_like(disparity_vis, dtype=np.uint8))

    if depth_max > 0:
        cv2.imwrite(depth_filename, (depth_vis / depth_max * 255).astype(np.uint8))
    else:
        cv2.imwrite(depth_filename, np.zeros_like(depth_vis, dtype=np.uint8))


# ========= 入口 =========
if __name__ == "__main__":
    main()