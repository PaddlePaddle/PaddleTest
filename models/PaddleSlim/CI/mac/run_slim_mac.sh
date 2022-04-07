#!/usr/bin/env bash
#python version、paddle_compile_path、slim_install_method
slim_prepare_env_mac.sh $1 $2 $3

#下载小数据集及预训练模型
slim_prepare_data_mac.sh

# cudaid1、cudaid2
slim_run_case_mac.sh


cd ${slim_dir}/logs
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---failed---
    exit 1
else
    echo ---passed---
    exit 0
fi
