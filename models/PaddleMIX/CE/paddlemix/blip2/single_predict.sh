#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_predict.py \
    --num_train_epochs 1 \
    --warmup_steps 100

