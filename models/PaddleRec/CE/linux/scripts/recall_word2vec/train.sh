#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=$2
temp_path=$(echo $2|awk -F '_' '{print $2}')
echo "$2 infer"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/recall/${temp_path}/
log_path=$root_path/log/recall_word2vec/
mkdir -p $log_path
#临时环境更改

#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

cd $code_path
echo -e "\033[32m `pwd` train \033[0m";
# word2vec模型功能运行
sed -i "s/  epochs: 5/  epochs: 1/g" config_bigdata.yaml
sed -i "s/  infer_end_epoch: 5/  infer_end_epoch: 1/g" config_bigdata.yaml
rm -rf output
# linux train
if [ "$1" = "linux_dy_gpu1" ];then #单卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    python -u ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_dy_gpu2" ];then #多卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_dy_cpu" ];then
    python -u ../../../tools/trainer.py -m config.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_gpu1" ];then #单卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    python -u ../../../tools/static_trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_gpu2" ];then #多卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/static_trainer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_cpu" ];then
    python -u ../../../tools/static_trainer.py -m config.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

# mac small_data train
elif [ "$1" = "mac_dy_cpu" ];then
    python -u ../../../tools/trainer.py -m config.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "mac_st_cpu" ];then
    python -u ../../../tools/static_trainer.py -m config.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2

else
    echo "$model_name train.sh  parameters error "
fi
