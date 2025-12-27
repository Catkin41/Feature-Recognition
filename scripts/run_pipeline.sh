#!/usr/bin/env bash
# 在 Git Bash 中执行 : bash scripts/run_pipeline.sh dataset/queries/q1.jpg
QUERY="$1"
if [ -z "$QUERY" ]; then
  echo "Usage: bash scripts/run_pipeline.sh dataset/queries/q1.jpg"
  exit 1
fi
source venv/Scripts/activate
python src/preprocess.py
python src/extract_features.py
python src/search_two_stage.py --query_path "$QUERY" --topk 5 --nprobe 30