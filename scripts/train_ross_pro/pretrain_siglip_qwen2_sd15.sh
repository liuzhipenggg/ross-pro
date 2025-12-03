#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

EXP_NAME="ross-pro-siglip-qwen2-7b-sd15-kl8-mlp2x-pt558k-xomni"
export WANDB_PROJECT=ross-pro

set -x

torchrun --nproc-per-node=8 --nnodes $1 --node_rank $2 \
    --master_addr="localhost" --master_port="29805" \
    \
    train.py \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --warmup_ratio 0.03 \
    --mm_inv_projector_lr 1e-4 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /root/paddlejob/Qwen2-7B-Instruct \
    --output_dir ./checkpoints/$EXP_NAME \
    --vision_tower /root/paddlejob/siglip-so400m-patch14-384 \
    --version plain \
    --mm_pixel_decoder /root/paddlejob/stable-diffusion-v1-5/vae \
    \
    --data_path /mnt/haochen/datasets/encoded_llava_pretrain \
    --image_folder '' \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type sd15xomni_mlp2x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $EXP_NAME

rm -fr ./checkpoints/$EXP_NAME/checkpoint*
cp -r ./checkpoints/$EXP_NAME /mnt/haochen/ross-pro-ckpt/$EXP_NAME