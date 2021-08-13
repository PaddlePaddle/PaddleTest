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
code_path=$cur_path/../../PaddleSlim/demo/quant/pact_quant_aware
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

echo -e "\033[32m `pwd` train \033[0m";

if [ "$1" = "no_pact" ];then #单卡
    # 普通量化
    python train.py --model MobileNetV3_large_x1_0 \
    --pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
    --num_epochs 1 --lr 0.0001 --use_pact False --batch_size 64 >${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "use_pact" ];then
    python train.py --model MobileNetV3_large_x1_0 \
    --pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
    --num_epochs 1 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
    --step_epochs 1 --l2_decay 1e-5 >${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "load" ];then  # load
    python train.py --model MobileNetV3_large_x1_0 \
    --pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
    --num_epochs 1 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
    --step_epochs 20 --l2_decay 1e-5 \
    --checkpoint_dir ./output/MobileNetV3_large_x1_0/0 \
    --checkpoint_epoch 0 >${log_path}/$2.log 2>&1
    print_info $? $2
fi


