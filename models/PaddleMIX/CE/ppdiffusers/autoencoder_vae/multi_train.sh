#!/bin/bash

rm -rf autoencoder_outputs/

python -u -m paddle.distributed.launch --gpus "0,1" train_vae.py \
  --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
  --ignore_keys decoder. \
  --vae_config_file config/vae.json \
  --freeze_encoder \
  --input_size 256 256 \
  --max_train_steps 100 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --num_workers 2 \
  --logging_steps 50 \
  --save_steps 100 \
  --image_logging_steps 50 \
  --disc_start 50 \
  --kl_weight 0.000001 \
  --disc_weight 0.5 \
  --resolution 512

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
