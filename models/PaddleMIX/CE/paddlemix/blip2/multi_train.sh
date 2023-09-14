#!/bin/bash

# stage1
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage1.py \
    --per_device_train_batch_size 128 \
    --num_train_epochs 1


# stage2
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage2.py \
    --per_device_train_batch_size 128 \
    --num_train_epochs 1