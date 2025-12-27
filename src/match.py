#!/usr/bin/env python3
"""
给定两个 features npz 文件 做匹配，返回 good matches (ratio test)
并可视化保存匹配图像。
用法：
  python src/match.py features/0001.npz features/0002.npz out.jpg
"""
import sys
import cv2
import numpy as np
from pathlib import Path

#暴力匹配（ORB用Hamming距离）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def match_descriptors(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return []
    if len(des1) == 0 or len(des2) == 0:
        return []
    
     # 强制转换为np.float32（ORB的uint8→float32，数值不变，不影响匹配结果）
    des1_norm = des1.astype(np.float32) if des1.dtype != np.float32 else des1
    des2_norm = des2.astype(np.float32) if des2.dtype != np.float32 else des2

    try:
        # k=2：取每个描述子的前2个最佳匹配
        matches = bf.knnMatch(des1_norm, des2_norm, k=2)
    except cv2.error as e:
        print(f"⚠️ 匹配警告：knnMatch执行失败 - {str(e)[:100]}")
        return []
        
    good = []
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])          # 阈值越小越严格
    return good

def visualize(img1_path, img2_path, kp1, kp2, matches, out_path):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        print("cannot read img for visualization", img1_path, img2_path)
        return
    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(out_path), vis)

def load_kps_and_des(npz_path):
    a = np.load(npz_path, allow_pickle=True)
    pts = a.get("pts")
    des = a.get("des")

    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)  # 兜底为ORB标准空数组
    else:
        des = des.astype(np.uint8)  # 非空时再转换类型


    kps = []
    if pts is not None:
        for (x,y) in pts:
            kps.append(cv2.KeyPoint(float(x), float(y), 1))
    return kps, des

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/match.py features/0001.npz features/0002.npz out.jpg")
        sys.exit(1)
    npz1, npz2, out = sys.argv[1:4]
    kp1, des1 = load_kps_and_des(npz1)
    kp2, des2 = load_kps_and_des(npz2)
    good = match_descriptors(des1, des2)
    print("Good matches:", len(good))
    # derive image paths
    img1 = Path("dataset/images_pre") / (Path(npz1).stem + ".jpg")
    img2 = Path("dataset/images_pre") / (Path(npz2).stem + ".jpg")
    visualize(img1, img2, kp1, kp2, good, out)