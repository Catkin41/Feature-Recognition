#!/usr/bin/env bash
# 在 Git Bash 中执行：bash scripts/setup_env.sh
python -m venv venv
source venv/Scripts/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || pip install opencv-python numpy scikit-learn matplotlib pillow
echo "Activated venv. To activate later: source venv/Scripts/activate"
