#!/usr/bin/env bash

run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
max_epoch=3
batch_size=128
model_name=dynamic_MobileNetV1_bs128
num_gpu_devices=${1:-"1"}

train_cmd="-c ./ppcls/configs/ImageNet/MobileNetV1/MobileNetV1.yaml
               -o Global.epochs=${max_epoch}
               -o Global.eval_during_train=False
               -o Global.save_interval=2
               -o DataLoader.Train.sampler.batch_size=${batch_size}
               -o DataLoader.Train.loader.num_workers=8"
if [ ${num_gpu_devices} = "1" ]; then
        run_mode=sp
        train_cmd="python -u tools/train.py "${train_cmd}
else
        run_mode=mp
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog_${model_name} tools/train.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0"
fi
log_file=${run_log_path}/dynamic_MobileNetV1_bs128_${num_gpu_devices}_${run_mode}
# 运行模型
    timeout 15m ${train_cmd} > ${log_file} 2>&1

if [ ${run_mode} != "sp"  -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/`ls -l mylog_${model_name}/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
fi
