#!/usr/bin/env python3
"""
简易命令行入口：按顺序运行预处理->提取->检索（两阶段）
用法示例：
  python src/cli.py preprocess extract extract_feats brute --query dataset/queries/q1.jpg
本文件是教学示例，按需调整。
"""
import sys
import subprocess

def run(cmd):
    print("RUN:", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print("Command failed:", cmd)
        sys.exit(1)

if __name__ == "__main__":
    # 示例：  python src/cli.py run_all dataset/queries/q1.jpg
    if len(sys.argv) < 2:
        print("Usage: python src/cli.py run_all <query_path>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "run_all":
        query = sys.argv[2]
        run("python src/preprocess.py")
        run("python src/extract_features.py")
        run(f"python src/search_two_stage.py --query_path {query} --topk 5 --nprobe 30")
    else:
        print("Unknown mode")