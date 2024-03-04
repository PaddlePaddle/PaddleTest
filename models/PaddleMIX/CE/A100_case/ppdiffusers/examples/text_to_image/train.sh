#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096

rm -rf sdxl-pokemon-model

python -u train_text_to_image_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --pretrained_vae_model_name_or_path=$VAE_NAME \
    --dataset_name=$DATASET_NAME \
    --enable_xformers_memory_efficient_attention \
    --resolution=512 --center_crop --random_flip \
    --proportion_empty_prompts=0.2 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=100 \
    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --mixed_precision="fp16" \
    --report_to="wandb" \
    --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
    --checkpointing_steps=100 \
    --output_dir="sdxl-pokemon-model"

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
    exit 1
fi
