#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型train阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/quant/quant_embedding
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

if [ -d "data" ];then
    rm -rf data
fi
wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
tar xf word_2evc_demo_data.tar.gz
mv word_2evc_demo_data data
if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi

echo -e "\033[32m `pwd` train \033[0m";

if [ "$1" = "linux_st_gpu1" ];then #单卡
    OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py \
    --train_data_dir data/convert_text8 \
    --dict_path data/test_build_dict \
    --num_passes 1 --batch_size 100 \
    --model_output_dir v1_cpu5_b100_lr1dir \
    --base_lr 1.0 --print_batch 1000 \
    --with_speed --is_sparse > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
