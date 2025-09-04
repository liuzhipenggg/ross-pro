#!/bin/bash

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

EXP_NAME="ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd-ct1319k-ftsd"
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
    --mm_inv_projector_lr 1e-5 \
    \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /root/paddlejob/Qwen2-7B-Instruct \
    --pretrain_mm_mlp_adapter ./checkpoints/ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd/mm_projector.bin \
    --pretrain_mm_inv_mlp_adapter ./checkpoints/ross-pro-siglip-qwen2-7b-sd3-kl8-mlp2x-pt558k-xomni-ftsd/mm_inv_projector.bin \
    --output_dir ./checkpoints/$EXP_NAME \
    --vision_tower /root/paddlejob/siglip-so400m-patch14-384 \
    --version qwen_2 \
    --mm_pixel_decoder /root/paddlejob/stable-diffusion-3-medium-diffusers/vae \
    \
    --data_path /mnt/haochen/datasets/encoded_cambrian_alignment_coco_118k,/mnt/haochen/datasets/encoded_cambrian_alignment_allava_631k \
    --image_folder '' \
    \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_inv_projector_type sd3xomni_mlp2x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4355 \
    --save_total_limit 1 \
    --save_only_model \
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

mkdir /mnt/haochen/ross-pro-ckpt/$EXP_NAME
rm -fr ./checkpoints/$EXP_NAME/checkpoint*
rsync -ah --progress ./checkpoints/$EXP_NAME/* /mnt/haochen/ross-pro-ckpt/$EXP_NAME