#!/bin/bash

# wget https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mscoco.en.1k

python generate_pipelines.py \
  --model_file ./laion400m_pretrain_output_trainer/model_state.pdparams \
  --output_path ./ldm_pipelines \
  --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
  --text_encoder_config_file ./config/ldmbert.json \
  --unet_config_file ./config/unet.json \
  --tokenizer_name_or_path bert-base-uncased \
  --model_max_length 77

python generate_images.py \
  --model_name_or_path ./ldm_pipelines \
  --file coco1k \
  --batch_size 1 \
  --save_path ./outputs \
  --guidance_scales 3 \
  --seed 42 \
  --scheduler_type ddim \
  --height 256 \
  --width 256 \
  --num_inference_steps 1 \
  --device gpu

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi
