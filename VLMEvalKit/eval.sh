#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

set -x

torchrun --nproc-per-node=1 --master-port=8591 run.py --reuse \
    --data POPE HallusionBench MMBench_DEV_EN MMBench_DEV_CN SEEDBench_IMG OCRBench ChartQA_TEST RealWorldQA CV-Bench-2D CV-Bench-3D VStarBench \
    --model $@