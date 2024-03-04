#!/bin/bash

config_file=config/DiT_XL_patch2.json
OUTPUT_DIR=./output/DiT_XL_patch2_trainer

# config_file=config/SiT_XL_patch2.json
# OUTPUT_DIR=./output/SiT_XL_patch2_trainer

feature_path=./fastdit_imagenet256
batch_size=2 # per gpu
num_workers=2
max_steps=50
logging_steps=10
seed=0

USE_AMP=True
FP16_OPT_LEVEL="O1"
enable_tensorboard=True
recompute=True
enable_xformers=True

python -u -m paddle.distributed.launch --gpus "0,1" train_image_generation_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 50 \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps 50 \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 1 \
    --use_ema True \
    --max_grad_norm -1 \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP}

rm -rf ./fastdit_imagenet256
rf -rf ${OUTPUT_DIR}
