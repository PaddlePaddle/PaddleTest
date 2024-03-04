#!/bin/bash

rm -rf ./sd15_openpose

export FLAGS_conv_workspace_size_limit=4096
python -u train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 100 \
    --logging_steps 1 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 100 \
    --seed 4096 \
    --dataloader_num_workers 0 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data_demo/train.openpose.filelist \
    --recompute False --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir \
    --timestep_sample_schedule cubic
