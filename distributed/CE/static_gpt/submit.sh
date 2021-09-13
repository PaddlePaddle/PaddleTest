#!/bin/bash
set -xe

server="paddlecloud.baidu-int.com"
port=80

card_type=${card_type:-"gpu_v100"}
num_trainers=${num_trainers:-1}
num_cards=${num_cards:-8}
total_cards=$((num_cards*num_trainers))

#hybrid_config="DP"$4_"MP"$5_"PP"$6

job_name=GPT_$1_$2_$3

#image_addr=${image_addr:-"registry.baidu.com/paddlepaddle-public/paddle_ubuntu1604:mlperf_cuda10.1_cudnn7.6.5_nccl2.4.7_dali0.24.0_py37"}
image_addr=${image_addr:-"registry.baidu.com/wangxi16/paddlecloud:paddlecloud-ubuntu18.04-gcc8.2-cuda11.2-cudnn8"}
k8s_wall_time="00:00:00"

#distribute=" --k8s-not-local --distribute-job-type NCCL2 "
script_file="run_cloud.sh"


files="$script_file before_hook.sh end_hook.sh run_cloud.sh";


paddlecloud job \
    --server ${server} \
    --port ${port} \
    train --job-version paddle-fluid-custom \
    --group-name nlp-32g-cf01-yq01-k8s-gpu-v100-8 \
    --cluster-name v100-32-0-cluster \
    --k8s-gpu-cards ${num_cards} \
    --k8s-priority high \
    --k8s-wall-time ${k8s_wall_time} \
    --k8s-memory 350Gi \
    --job-name ${job_name} \
    --permission group \
    --start-cmd "bash $script_file $1 $2 $3 $4 $5 $6" \
    --job-conf config.ini \
    --files ${files} \
    --k8s-trainers ${num_trainers} ${distribute} \
    --is-auto-over-sell 0 \
    --image-addr "${image_addr}" \
    --k8s-cpu-cores 35
