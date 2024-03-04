#!/bin/bash

export BATCH_SIZE=2
export MAX_ITER=50

rm -rf ./outputs_ip_adapter_n1c2

python -u -m paddle.distributed.launch --gpus "0,1" train_ip_adapter.py \
    --do_train \
    --output_dir "outputs_ip_adapter_n1c2" \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 1 \
    --resolution 512 \
    --save_steps 25 \
    --save_total_limit 1000 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --image_encoder_name_or_path h94/IP-Adapter/models/image_encoder \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --fp16 True \
    --fp16_opt_level O2
