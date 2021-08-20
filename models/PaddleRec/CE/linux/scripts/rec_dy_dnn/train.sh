#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name train"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/rank/dnn/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改

#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}


cd $code_path
sed -i "s/  epochs: 4/  epochs: 1/g" config_bigdata.yaml
sed -i "s/  infer_end_epoch: 4/  infer_end_epoch: 1/g" config_bigdata.yaml

rm -rf output

if [ "$1" = "single" ];then #单卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    python -u ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "multi" ];then #多卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/trainer.py -m config_bigdata.yaml ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/log $log_path/$2_dist_log

elif [ "$1" = "cpu" ];then
    # CPU
    python -u ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
else
    echo "$model_name train.sh  parameter error "
fi
