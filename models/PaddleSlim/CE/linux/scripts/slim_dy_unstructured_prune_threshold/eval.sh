#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型eval阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/dygraph/unstructured_pruning
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    echo -e "\033[31m FAIL_$2 \033[0m"
    echo $2 fail log as follows
    cat ${log_path}/$2.log
    cp ${log_path}/$2.log ${log_path}/FAIL_$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path
echo -e "\033[32m `pwd` eval \033[0m";

if [ "$1" = "linux_st_gpu1" ];then #单卡
    python evaluate.py \
    --pruned_model dy_threshold_models_gpu1/model-pruned.pdparams \
    --data imagenet  > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_gpu2" ];then #单卡
     python evaluate.py \
    --pruned_model dy_threshold_models_gpu2/model-pruned.pdparams \
    --data imagenet > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_cpu" ];then #单卡
     python evaluate.py \
    --pruned_model dy_threshold_models_cpu/model-pruned.pdparams \
    --data imagenet \
    --use_gpu False > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
