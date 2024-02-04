#!/bin/bash

python train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512

rm -rf ${work_path}/autoencoder_outputs/*

python -u -m paddle.distributed.launch --gpus "0,1" train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512

rm -rf ./autoencoder_outputs/*
rm -rf ./data/
