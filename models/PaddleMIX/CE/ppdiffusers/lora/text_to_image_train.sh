#!/bin/bash

export http_proxy=${proxy}
export https_proxy=${proxy}

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="sd-pokemon-model-lora"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=150 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=50 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --validation_epochs 10

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
