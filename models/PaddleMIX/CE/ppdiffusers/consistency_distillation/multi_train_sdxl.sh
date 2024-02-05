#!/bin/bash

MAX_ITER=50
MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-xl-base-1.0"
IS_SDXL=True
RESOLUTION=512

python -u -m paddle.distributed.launch --gpus "0,1" train_lcm.py \
    --do_train \
    --output_dir "lcm_sdxl_lora_n1c2_outputs" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 2000000 \
    --logging_steps 1 \
    --resolution ${RESOLUTION} \
    --save_steps 25 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --overwrite_output_dir
