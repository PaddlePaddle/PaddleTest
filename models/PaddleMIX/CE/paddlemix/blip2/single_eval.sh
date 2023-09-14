#!/bin/bash

# task_vqa
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_eval_vqav2_zeroshot.py \
    --num_train_epochs 1 \
    --warmup_steps 100

# task_caption
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_eval_caption.py \
    --num_train_epochs 1 \
    --warmup_steps 100
