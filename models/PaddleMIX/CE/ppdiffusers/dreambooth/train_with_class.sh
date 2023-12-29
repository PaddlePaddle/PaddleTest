#!/bin/bash

rm -rf ./dream_outputs_with_class

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="./dogs"
export CLASS_DIR="./dream_class_image"
export OUTPUT_DIR="./dream_outputs_with_class"

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=10 \
  --max_train_steps=100

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
