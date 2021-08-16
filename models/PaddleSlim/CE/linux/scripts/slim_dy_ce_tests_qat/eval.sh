#!/usr/bin/env bash
#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型eval阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/ce_tests/dygraph/quant
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
    cat ${log_path}/$2.log | grep test_acc1 | tr ',' ' ' | awk -F ' ' \
        '{print"\nacc_top1:" $4"\tacc_top5:"$6}' >> ${log_path}/$2.log
    echo "exit_code: 0.0" >> ${log_path}/$2.log
    cat ${log_path}/$2.log
fi
}

cd $code_path

test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=16
epoch=1
lr=0.0001
num_workers=1
output_dir=$PWD/output_models
model=mobilenet_v1

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
