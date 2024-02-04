#!/bin/bash

python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True

rm -rf ./laion400m_pretrain_output_trainer

python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True

rm -rf ./laion400m_pretrain_output_trainer
rm -rf ./data/
