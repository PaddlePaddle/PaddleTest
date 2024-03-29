#!/bin/bash

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1           # nnodes, machine num
TRAINING_GPUS_PER_NODE=2 # nproc_per_node
DP_DEGREE=1              # dp_parallel_degree
MP_DEGREE=1              # tensor_parallel_degree
SHARDING_DEGREE=1        # sharding_parallel_degree

uvit_config_file=config/uvit_t2i_small.json
output_dir=output_dir/uvit_t2i_small

feature_path=./datasets/coco256_features
per_device_train_batch_size=2
dataloader_num_workers=2
max_steps=50
save_steps=50
warmup_steps=50
logging_steps=20
image_logging_steps=50
seed=1234

USE_AMP=True
fp16_opt_level="O1"
enable_tensorboard=True
recompute=True
enable_xformers=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_txt2img_mscoco_uvit_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0002 \
    --weight_decay 0.03 \
    --adam_beta1 0.9 \
    --adam_beta2 0.9 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps ${warmup_steps} \
    --image_logging_steps ${image_logging_steps} \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --seed ${seed} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --max_grad_norm -1 \
    --uvit_config_file ${uvit_config_file} \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --use_ema True \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --recompute ${recompute} \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level} \
    --enable_xformers_memory_efficient_attention ${enable_xformers}

rm -rf ./datasets
