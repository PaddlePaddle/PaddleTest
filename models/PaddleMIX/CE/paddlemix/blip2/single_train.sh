#!/bin/bash

# stage1
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage1.py \
    --per_device_train_batch_size 64 \
    --warmup_steps 100 \
    --num_train_epochs 1

# stage2
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage2.py \
    --per_device_train_batch_size 128 \
    --warmup_steps 100 \
    --num_train_epochs 1