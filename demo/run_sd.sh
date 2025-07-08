export MODEL_NAME="/root/paddlejob/stable-diffusion-v1-4"
export INSTANCE_DIR="../data/dog"
export OUTPUT_DIR="trained-sd"
export WANDB_PROJECT="dreambooth"

export http_proxy=agent.baidu.com:8188
export https_proxy=agent.baidu.com:8188
export no_proxy=baidu.com,baidubce.com,localhost,127.0.0.1,bj.bcebos.com

accelerate launch --main_process_port=8841 train_dreambooth_sd.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400