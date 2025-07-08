#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

set -x

python3 run.py \
    --data POPE HallusionBench MMBench_DEV_EN MMBench_DEV_CN SEEDBench_IMG OCRBench MMVP ChartQA_TEST RealWorldQA \
    --model llava-siglip-qwen2-7b-pt558k-sft737k ross-siglip-qwen2-7b-flux-kl8-dit3x-pt558k-sft737k