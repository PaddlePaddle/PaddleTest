#!/bin/bash

# wget https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mscoco.en.1k


python generate_images.py \
    --model_name_or_path ./ldm_pipelines \
    --file coco1k \
    --batch_size 16 \
    --save_path ./outputs \
    --guidance_scales 3 \
    --seed 42 \
    --scheduler_type ddim \
    --height 256 \
    --width 256 \
    --num_inference_steps 50 \
    --device gpu

# 检查命令是否成功执行
if [ $? -ne 0 ]; then
  exit 1
fi