#!/bin/bash

rm -rf ./sd15_control

export FLAGS_conv_workspace_size_limit=4096
python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 100 \
    --logging_steps 50 \
    --image_logging_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir
