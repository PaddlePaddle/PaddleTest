#!/bin/bash

# task_vqa
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_eval_vqav2_zeroshot.py \
    --num_train_epochs 1 \
    --warmup_steps 100

# task_caption
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_eval_caption.py \
    --num_train_epochs 1 \
    --warmup_steps 100