#!/bin/bash

rm -rf ./dream_outputs

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dogs"
export OUTPUT_DIR="./dream_outputs"

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
