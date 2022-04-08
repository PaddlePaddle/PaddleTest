#!/usr/bin/env bash
#python version、paddle_compile_path、slim_install_method
# for logs env
export slim_dir=$PWD/PaddleSlim;

if [ -d "logs" ];then
    rm -rf logs;
fi
mkdir logs
export log_path=$PWD/logs;

bash slim_prepare_env_mac.sh $1 $2 $3

#下载小数据集及预训练模型
bash slim_prepare_data_mac.sh

# cudaid1、cudaid2
bash slim_run_case_mac.sh


cd ${log_path}
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---failed case：${FF}---
    ls *FAIL*|wc -l
    exit 1
else
    echo ---all case passed---
    exit 0
fi
