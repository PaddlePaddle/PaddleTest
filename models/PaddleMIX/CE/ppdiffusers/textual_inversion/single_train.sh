#!/bin/bash

rm -rf ./textual_inversion_cat/

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="cat_toy_images"

python -u train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --output_dir="textual_inversion_cat"

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
