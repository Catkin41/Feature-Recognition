#!/usr/bin/env python3
"""
遍历 dataset/images_pre，使用 ORB 提取 keypoints 与 descriptors，
并把结果保存到 features/<imagename>.npz（包含 keypoints pts 列表 与 descriptors numpy 数组）
用法：python src/extract_features.py
"""
import cv2
import numpy as np
from pathlib import Path
import sys

IMG_DIR = Path("dataset/images_pre")
FEAT_DIR = Path("features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

orb = cv2.ORB_create(nfeatures=800)
#特征点数量

def extract_and_save(p: Path, out_dir: Path):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("WARN: cannot read", p)
        return
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)
    des = des.astype(np.uint8)
    pts = np.array([kp_i.pt for kp_i in kp], dtype=np.float32) if kp else np.empty((0,2), dtype=np.float32)
    # 保存 pts (N,2) 和 des (N, descriptor_size)
    out_path = out_dir / (p.stem + ".npz")
    np.savez_compressed(str(out_path), pts=pts, des=des)
    return len(kp) if kp else 0

# 新增：单张图片提取函数
def extract_single_image(img_path: str, out_feat_path: str):
    img_p = Path(img_path)
    out_p = Path(out_feat_path)
    # 调用原有提取逻辑，输出到指定路径（而非固定FEAT_DIR）
    img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("WARN: cannot read", img_p)
        return
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        des = np.empty((0, 32), dtype=np.uint8)
    des = des.astype(np.uint8)
    pts = np.array([kp_i.pt for kp_i in kp], dtype=np.float32) if kp else np.empty((0,2), dtype=np.float32)
    np.savez_compressed(str(out_p), pts=pts, des=des)
    print(f"Extracted features for {img_p} → {out_p} (keypoints: {len(kp) if kp else 0})")


def main():
    if len(sys.argv) == 3:
        # 命令行格式：python extra_features.py 图片路径 输出特征路径
        img_path = sys.argv[1]
        out_feat_path = sys.argv[2]
        extract_single_image(img_path, out_feat_path)
    else:
        files = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in (".jpg",".png",".jpeg")])
        if not files:
            print("No preprocessed images found in", IMG_DIR)
            return
        stats = []
        for p in files:
            n = extract_and_save(p, FEAT_DIR)
            stats.append(n)
        print(f"Extracted features for {len(files)} images. avg keypoints:", (sum(stats)/len(stats)) if stats else 0)

if __name__ == "__main__":
    main()