EXP_NAME=$1

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

CUDA_VISIBLE_DEVICES=1 python3 mmvp_eval.py \
    --model_path ../checkpoints/$EXP_NAME/checkpoint-5755 \
    --conv_mode qwen_2 \
    --answers_file ./answers/$EXP_NAME.jsonl

CUDA_VISIBLE_DEVICES=1 python3 mmvp_test.py \
    --answers_file ./answers/$EXP_NAME.jsonl \
    --csv_file ./all_results.csv