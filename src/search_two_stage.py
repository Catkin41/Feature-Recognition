#!/usr/bin/env python3
"""
两阶段检索：
  - 快速阶段：使用不带 RANSAC 的粗匹配 count 来筛选 Top-N
  - 精排阶段：对 Top-N 使用 RANSAC 计内点数并输出 Top-K
用法：
  python src/search_two_stage.py --query_path dataset/queries/q1.jpg --topk 5 --nprobe 30
"""
import argparse
from pathlib import Path
import time
from src.match import load_kps_and_des, match_descriptors
from src.ransac_validate import ransac_inliers

FEAT_DIR = Path("features")

def quick_score(des1, des2):
    # count of good matches (ratio test) as quick proxy
    good = match_descriptors(des1, des2)
    return len(good), good

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, help="path to query image (dataset/queries/xxx.jpg)")
    parser.add_argument("--query_feat", type=str, help="path to query feature (.npz)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--nprobe", type=int, default=30, help="Top-N to refine")
    args = parser.parse_args()

    if args.query_feat:
        qfeat = Path(args.query_feat)
    elif args.query_path:
        qfeat = Path("features") / (Path(args.query_path).stem + ".npz")
    else:
        print("Provide --query_path or --query_feat")
        return

    all_feats = sorted(FEAT_DIR.glob("*.npz"))
    # quick stage
    scores = []
    for f in all_feats:
        if f == qfeat: continue
        kp_q, des_q = load_kps_and_des(qfeat)
        kp_f, des_f = load_kps_and_des(f)
        cnt, good = quick_score(des_q, des_f)
        scores.append((f, cnt, good))
    scores.sort(key=lambda x: x[1], reverse=True)
    topn = scores[:args.nprobe]
    # refine stage
    refined = []
    for f, cnt, good in topn:
        inliers, mask = ransac_inliers(str(qfeat), str(f), good)
        refined.append((f.name, inliers))
    refined.sort(key=lambda x: x[1], reverse=True)
    print("Refined Top-K:")
    for name, inl in refined[:args.topk]:
        print(name, inl)

if __name__ == "__main__":
    main()