#!/usr/bin/env python3
"""
使用 RANSAC (findHomography) 对 matches 做几何验证，返回内点数和 mask。
示例：python src/ransac_validate.py features/0001.npz features/0002.npz
"""
import numpy as np
import cv2
from pathlib import Path
import sys
from typing import Tuple, List, Optional, Union

# ========== 原有核心配置：完全保留 ==========
RANSAC_REPROJ_THRESHOLD = 5.0
MATCHER_NORM_TYPE = cv2.NORM_HAMMING
RATIO_TEST_THRESHOLD = 0.75

def load_kps_des(npz_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载npz文件中的特征点(pts)和描述子(des)
    优化：补充des格式规范，解决None和维度异常问题
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        print(f"❌ 错误：文件 {npz_path} 不存在")
        return None, None
    
    try:
        a = np.load(npz_path, allow_pickle=True)
        pts = a.get("pts")
        des = a.get("des")
        
        # ========== 优化点1：补充des None判断和格式兜底 ==========
        if pts is None:
            pts = np.empty((0, 2), dtype=np.float32)
        if des is None:
            des = np.empty((0, 32), dtype=np.uint8)  # ORB标准空描述子
            print(f"⚠️ 警告：文件 {npz_path} 的des为None，已兜底为空数组")
        
        # ========== 优化点2：补充des合法性校验（维度+长度） ==========
        if len(pts) == 0 or len(des) == 0:
            print(f"⚠️ 警告：文件 {npz_path} 的特征点/描述子为空")
            return pts, des  # 不再返回None，返回空数组，兼容后续逻辑
        # 校验ORB描述子维度（32维），避免维度不匹配导致knnMatch失败
        if des.ndim == 2 and des.shape[1] != 32:
            print(f"⚠️ 警告：文件 {npz_path} 的des非ORB标准32维，维度为{des.shape[1]}")
        
        return pts, des
    except Exception as e:
        print(f"❌ 错误：读取 {npz_path} 失败 - {str(e)}")
        return None, None

def get_good_matches(des1: np.ndarray, des2: np.ndarray) -> List[cv2.DMatch]:
    """
    生成高质量匹配对（BF匹配器 + 比值测试）
    优化：增加异常捕获，解决大量knnMatch警告
    """
    # ========== 优化点3：先校验des1/des2有效性，避免无效调用knnMatch ==========
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("⚠️ 警告：无效的描述子，无法进行匹配")
        return []
    # 类型规范化，兼容ORB uint8类型
    des1_norm = des1.astype(np.float32) if des1.dtype != np.float32 else des1
    des2_norm = des2.astype(np.float32) if des2.dtype != np.float32 else des2

    bf = cv2.BFMatcher(MATCHER_NORM_TYPE, crossCheck=False)
    raw_matches = []
    try:
        # ========== 原有核心逻辑：knnMatch + 比值测试 ==========
        raw_matches = bf.knnMatch(des1_norm, des2_norm, k=2)
    except cv2.error as e:
        print(f"⚠️ 匹配警告：knnMatch执行失败 - {str(e)[:100]}")
        return []

    good_matches = []
    # 优化：增加m的长度判断，避免索引越界
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)
    
    print(f"🔍 原始匹配数：{len(raw_matches)} | 筛选后good matches数：{len(good_matches)}")
    return good_matches

def ransac_inliers(npz1: Union[str, Path], npz2: Union[str, Path], good_matches: List[cv2.DMatch]) -> Tuple[int, Optional[np.ndarray]]:
    """
    原有逻辑完全保留，仅优化注释，不改变参数和返回值
    """
    pts1, _ = load_kps_des(npz1)
    pts2, _ = load_kps_des(npz2)
    
    if pts1 is None or pts2 is None or len(good_matches) < 4:
        print("⚠️ 跳过RANSAC：特征点不足或匹配数<4")
        return 0, None
    
    src = np.float32([pts1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([pts2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"✅ RANSAC验证完成 | 内点数：{inliers} (内点数越高，图片越相似)")
    return inliers, mask

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ 用法错误！正确示例：")
        print("python src/ransac_validate.py features/0001.npz features/0002.npz")
        sys.exit(1)
    
    npz_path1 = sys.argv[1]
    npz_path2 = sys.argv[2]
    
    pts1, des1 = load_kps_des(npz_path1)
    pts2, des2 = load_kps_des(npz_path2)
    # 优化：调整判断逻辑，兼容空描述子
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("❌ 无效的描述子，无法继续执行")
        sys.exit(1)
    
    good_matches = get_good_matches(des1, des2)
    if len(good_matches) < 4:
        print("❌ 无足够的good matches进行RANSAC验证")
        sys.exit(1)
    
    inliers, mask = ransac_inliers(npz_path1, npz_path2, good_matches)
    
    if mask is not None:
        mask_save_path = Path("ransac_mask.npy")
        np.save(mask_save_path, mask)
        print(f"💾 Mask已保存至：{mask_save_path}")
    
    print(f"\n📊 最终结果 | 两张图片的RANSAC内点数：{inliers}")