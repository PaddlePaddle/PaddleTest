#!/usr/bin/env bash

run_log_path=${TRAIN_LOG_DIR:-$(pwd)}
num_gpu_devices=${1:-1}
max_iter=50
model_name="yolov3"
base_batch_size=8

function _set_env(){
    export FLAGS_eager_delete_tensor_gb=0.0
    export FLAGS_fraction_of_gpu_memory_to_use=0.98
    export FLAGS_memory_fraction_of_eager_deletion=1.0
}

_set_env
grep -q "#To address max_iter" ppdet/engine/trainer.py
if [ $? -ne 0 ]; then
   sed -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id = '${max_iter}' #To address max_iter' ppdet/engine/trainer.py
   sed -i '/for step_id, data in enumerate(self.loader):/a\                if step_id >= max_step_id: return' ppdet/engine/trainer.py
fi
    model_name=${model_name}_bs${base_batch_size}
    if [ $num_gpu_devices -eq 1 ]; then norm_type="bn"; else norm_type="sync_bn"; fi
    train_cmd="-c configs/yolov3/yolov3_darknet53_270e_coco.yml
               --opt epoch=1 TrainReader.batch_size=${base_batch_size} worker_num=8 norm_type=${norm_type}"
if [ ${num_gpu_devices} = "1" ]; then
        run_mode=sp
        train_cmd="python -u tools/train.py "${train_cmd}
else
        run_mode=mp
        rm -rf ./mylog_${model_name}
        train_cmd="python -m paddle.distributed.launch --log_dir=./mylog_${model_name} --gpus=$CUDA_VISIBLE_DEVICES tools/train.py "${train_cmd}
        log_parse_file="mylog_${model_name}/workerlog.0"
fi

log_file=${run_log_path}/dynamic_yolov3_bs8_${num_gpu_devices}_${run_mode}

# 运行模型
timeout 15m ${train_cmd} > ${log_file} 2>&1

if [ ${run_mode} != "sp"  -a -d mylog_${model_name} ]; then
        rm ${log_file}
        cp mylog_${model_name}/`ls -l mylog_${model_name}/ | awk '/^[^d]/ {print $5,$9}' | sort -nr | head -1 | awk '{print $2}'` ${log_file}
fi
