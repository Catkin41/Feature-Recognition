#!/usr/bin/env python3
"""
合法爬取Pexels CC0图片（非商用）
用途：仅用于以图搜图演示，遵守Pexels API条款
"""
import os
import requests
import dotenv
from pathlib import Path

# 加载环境变量（API Key）
dotenv.load_dotenv()
API_KEY = os.getenv("PEXELS_API_KEY")
if not API_KEY:
    raise ValueError("请在.env文件中配置PEXELS_API_KEY")

# 配置项（新手可修改以下参数）
SEARCH_KEYWORDS = [  # 爬取的关键词（对应演示所需的图片类型）
    "mug", "desk", "book", "lamp", "keyboard",  # 日常物品（相似组）
    "mountain", "sky", "leaf", "striped wallpaper", "checkered fabric"  # 干扰组
]
PER_KEYWORD = 20  # 每个关键词爬取20张，总计200张
SAVE_DIR = Path("dataset/images")  # 保存到项目图库目录
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Pexels API基础配置
BASE_URL = "https://api.pexels.com/v1/search"
HEADERS = {"Authorization": API_KEY}

def download_pexels_image(photo_url, save_path):
    """下载单张图片，处理格式与大小"""
    try:
        # 发送请求，设置超时与流模式（避免内存溢出）
        response = requests.get(photo_url, stream=True, timeout=10)
        response.raise_for_status()  # 捕获HTTP错误
        
        # 保存图片（自动处理JPG格式）
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 保存成功：{save_path}")
    except Exception as e:
        print(f"❌ 下载失败 {photo_url}：{str(e)}")

def crawl_by_keyword(keyword, num):
    """按关键词爬取指定数量的图片"""
    page = 1
    downloaded = 0
    while downloaded < num:
        # 构造API请求参数（每页最多80张，避免频繁请求）
        params = {
            "query": keyword,
            "per_page": min(num - downloaded, 80),
            "page": page,
            "size": "medium"  # 中等分辨率（800×800+，适合特征提取）
        }
        
        # 发送API请求（遵守频率限制：≤200次/小时）
        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 遍历图片并下载
        for photo in data["photos"]:
            if downloaded >= num:
                break
            # 取中等分辨率的图片URL（平衡质量与体积）
            img_url = photo["src"]["medium"]
            # 生成唯一文件名（避免重复）
            img_name = f"{keyword}_{downloaded+1}.jpg"
            save_path = SAVE_DIR / img_name
            # 下载图片
            download_pexels_image(img_url, save_path)
            downloaded += 1
        
        page += 1
        # 无更多结果则停止
        if not data["photos"]:
            break

if __name__ == "__main__":
    # 遍历关键词爬取
    for keyword in SEARCH_KEYWORDS:
        print(f"\n===== 开始爬取关键词：{keyword} =====")
        crawl_by_keyword(keyword, PER_KEYWORD)
    print("\n🎉 所有图片爬取完成，保存至：", SAVE_DIR)