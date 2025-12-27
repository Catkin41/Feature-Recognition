#!/usr/bin/env python3
"""
基于提供的 ground-truth CSV 评估 Precision@K。
CSV 格式（header）：query_filename,gt_filename
示例： dataset/queries/q1.jpg,images/0001.jpg
用法：
  python src/evaluate.py --gt groundtruth.csv --method brute --topk 5
注意：本脚本假设已经有 search_bruteforce.py 的输出接口，可按需修改对接。
"""
import argparse
import csv
from pathlib import Path
import subprocess

def load_gt(path):
    gt = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            q, g = row[0].strip(), row[1].strip()
            gt[q] = g
    return gt

def run_search_and_get_topk(query, topk):
    # 调用 search_bruteforce.py 并解析输出（简单方式）
    cmd = ["python", "src/search_bruteforce.py", "--query_path", query, "--topk", str(topk)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.splitlines()
    top = []
    for line in out:
        if '\t' in line:
            name = line.split('\t')[0].strip()
            top.append(name)
    return top

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="ground truth csv")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    gt = load_gt(args.gt)
    total = 0
    correct_at_k = 0
    for q, g in gt.items():
        total += 1
        topk = run_search_and_get_topk(q, args.topk)
        # ground truth filename is assumed relative to dataset/images or features; compare stems
        gt_stem = Path(g).stem
        found = any(Path(name).stem == gt_stem for name in topk)
        if found:
            correct_at_k += 1
    print(f"Precision@{args.topk}: {correct_at_k}/{total} = {correct_at_k/total:.3f}")

if __name__ == "__main__":
    main()