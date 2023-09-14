#!/bin/bash

fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_predict.py \
    --num_train_epochs 1 \
    --warmup_steps 100