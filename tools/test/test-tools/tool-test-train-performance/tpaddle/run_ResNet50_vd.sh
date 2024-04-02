#!/usr/bin/env bash
#cd PaddleClas/
# 运行:sh run_ResNet50_vd.sh {num_gpu_devices}

run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
max_epoch=3
batch_size=64
model_name=dynamic_ResNet50_vd
num_gpu_devices=${1:-"1"}

train_cmd="-c ./ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml
               -o Global.epochs=${max_epoch}
               -o Global.eval_during_train=False
               -o Global.save_interval=2
               -o DataLoader.Train.sampler.batch_size=${batch_size}
               -o DataLoader.Train.loader.num_workers=8"

if [ ${num_gpu_devices} = "1" ]; then
        run_mode=sp
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES tools/train.py "${train_cmd}
else
        run_mode=mp
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --gpus=$CUDA_VISIBLE_DEVICES --log_dir ./mylog_${model_name} tools/train.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0"

fi
log_file=${run_log_path}/dynamic_ResNet50_vd_bs64_${num_gpu_devices}_${run_mode}

# 运行模型
    timeout 15m ${train_cmd} > ${log_file} 2>&1
if [ ${run_mode} != "sp"  -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/`ls -l mylog_${model_name}/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
fi
