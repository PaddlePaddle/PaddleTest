#!/usr/bin/env bash

# set slim_dir and logs path
# workspace == PaddleSlim/
export slim_dir=/workspace
if [ -d "/workspace/logs" ];then
    rm -rf /workspace/logs;
fi
mkdir /workspace/logs
export log_path=/workspace/logs


#python version、paddle_compile_path、slim_install_method
bash slim_prepare_env.sh $1 $2 $3

#下载小数据集及预训练模型
bash slim_prepare_data.sh

# run_CI/run_CE/ALL 、cudaid1、cudaid2
bash slim_run_case_linux.sh $4 $5 $6


cd ${log_path}
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---fail case: ${FF}
    exit 1
else
    echo ---all case pass---
    exit 0
fi
