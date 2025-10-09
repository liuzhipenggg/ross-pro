#!/bin/bash

# 检查是否有8张GPU可用
if [ $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l) -lt 8 ]; then
    echo "警告: 检测到的GPU数量少于8张，可能无法正常运行所有进程"
fi

# 在8张GPU上并行运行程序，添加时间戳
for gpu in {0..7}; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 在GPU $gpu 上启动程序..."
    CUDA_VISIBLE_DEVICES=$gpu python run.py &
done
