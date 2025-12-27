#!/usr/bin/env python3
"""
预处理：从 dataset/images -> dataset/images_pre
功能：统一最长边为 max_side（默认800），并灰度化保存
用法：python src/preprocess.py
"""
import cv2
from pathlib import Path

SRC_DIR = Path("dataset/images")
OUT_DIR = Path("dataset/images_pre")
MAX_SIDE = 800

OUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_image(p: Path, out_dir: Path, max_side: int = MAX_SIDE):
    img = cv2.imread(str(p))
    if img is None:
        print("WARN: cannot read", p)
        return
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale != 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out_path = out_dir / p.name
    cv2.imwrite(str(out_path), gray)

def main():
    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        print("No images found in", SRC_DIR)
        return
    for p in files:
        preprocess_image(p, OUT_DIR)
    print("Preprocessed", len(files), "images ->", OUT_DIR)

if __name__ == "__main__":
    main()