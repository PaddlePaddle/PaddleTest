#!/usr/bin/env bash
#python version、paddle_compile_path、slim_install_method
slim_prepare_env.sh $1 $2 $3

#下载小数据集及预训练模型
slim_prepare_data.sh

# run_CI/run_CE/ALL 、cudaid1、cudaid2
slim_run_case.sh $4 $5 $6


cd ${slim_dir}/logs
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    exit 1
else
    exit 0
fi
