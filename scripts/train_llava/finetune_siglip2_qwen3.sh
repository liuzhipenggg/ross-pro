#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

EXP_NAME="llava-siglip2-qwen3-4b-pt558k-sft737k"
export WANDB_PROJECT=ross-pro

set -x

torchrun --nproc-per-node=8 --nnodes $1 --node_rank $2 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /root/paddlejob/Qwen3-4B-Instruct-2507 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-siglip2-qwen3-4b-pt558k/mm_projector.bin \
    --output_dir ./checkpoints/$EXP_NAME \
    --vision_tower /root/paddlejob/siglip2-so400m-patch14-384 \
    --version qwen_2_5 \
    \
    --data_path /mnt/haochen/datasets/encoded_cambrian_737k \
    --image_folder '' \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5755 \
    --save_total_limit 1 \
    --save_only_model \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $EXP_NAME

bash run.sh