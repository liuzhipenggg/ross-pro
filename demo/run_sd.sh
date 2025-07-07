export MODEL_NAME="/mnt/haochen/hf_home/stable-diffusion-v1-4"
export INSTANCE_DIR="data/dog"
export OUTPUT_DIR="trained-sd"
export WANDB_PROJECT="dreambooth"

export https_proxy=http://10.162.37.16:8128
export http_proxy=http://10.162.37.16:8128

accelerate launch train_dreambooth.py \
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
  --max_train_steps=400 \
  --push_to_hub