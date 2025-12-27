#!/usr/bin/env python3
"""
暴力检索：给定查询图的 features（或查询图片路径），遍历 features/ 目录对每张图片计算匹配并用 RANSAC 计数内点，
按内点数排序返回 Top-K。
用法：
  python src/search_bruteforce.py --query_path dataset/queries/q1.jpg --topk 5
或先用 features 文件：
  python src/search_bruteforce.py --query_feat features/q1.npz --topk 5
"""
import argparse
import sys
from pathlib import Path
import time
import numpy as np
import cv2
from src.match import load_kps_and_des, match_descriptors
from src.ransac_validate import ransac_inliers

FEAT_DIR = Path("features")

def features_for_image(img_path):
    # if given image path, assume features/<stem>.npz exists
    p = Path(img_path)
    feat = FEAT_DIR / (p.stem + ".npz")
    if not feat.exists():
        raise FileNotFoundError(f"{feat} not found; run extract_features.py first")
    return feat

def score_query(query_feat_path, candidate_feat_path):
    kp1, des1 = load_kps_and_des(query_feat_path)
    kp2, des2 = load_kps_and_des(candidate_feat_path)
    good = match_descriptors(des1, des2)
    inliers, mask = ransac_inliers(query_feat_path, candidate_feat_path, good)
    return inliers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, help="path to query image (dataset/queries/xxx.jpg)")
    parser.add_argument("--query_feat", type=str, help="path to query feature (.npz)")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.query_feat:
        qfeat = Path(args.query_feat)
    elif args.query_path:
        qfeat = features_for_image(args.query_path)
    else:
        print("Provide --query_path or --query_feat")
        return

    all_feats = sorted(FEAT_DIR.glob("*.npz"))
    results = []
    t0 = time.time()
    for f in all_feats:
        if f == qfeat: continue
        score = score_query(qfeat, f)
        results.append((f.name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    elapsed = time.time() - t0
    print(f"Query {qfeat.name} done. elapsed {elapsed:.3f}s. Top-{args.topk}:")
    for name, score in results[:args.topk]:
        print(f"{name}\tinliers={score}")

if __name__ == "__main__":
    main()