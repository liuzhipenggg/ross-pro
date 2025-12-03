#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

set -x

torchrun --nproc-per-node=7 --master-port=8591 run.py --reuse \
    --data POPE HallusionBench MMBench_DEV_EN MMBench_DEV_CN SEEDBench_IMG OCRBench ChartQA_TEST RealWorldQA CV-Bench-2D CV-Bench-3D VStarBench \
    --model $@

# torchrun --nproc-per-node=8 --master-port=8591 run.py --reuse \
#     --data MMT-Bench_VAL \
#     --model $@

# torchrun --nproc-per-node=1 --master-port=8591 run.py --reuse \
#     --data MMBench_DEV_EN_V11 AesBench_VAL Q-Bench1_VAL A-Bench_VAL CCBench AI2D_TEST MMStar RealWorldQA MLLMGuard_DS BLINK TaskMeAnything_v1_imageqa_random A-OKVQA WorldMedQA-V VisOnlyQA-VLMEvalKit MMSci_DEV_MCQ MMCR SpatialEval MMSIBench_circular MicroVQA StaticEmbodiedBench \
#     --model $@