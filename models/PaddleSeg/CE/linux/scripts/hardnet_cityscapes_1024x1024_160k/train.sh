#!/bin/bash
# 外部传入参数
# $1 训练使用单卡or多卡

#获取当前路径
cur_path=`pwd`
model=${PWD##*/}
echo "${model} 模型数train阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
#创建日志路径
if [[ ! -d "$cur_path/../../log/${model}" ]];then
mkdir -p "$cur_path/../../log/${model}"
fi

print_info(){
if [ $1 -ne 0 ];then
    echo -e "${model},train_model_$2,FAIL"
    echo "exit_code: 1.0" >> ../log/${model}/${model}_train_$2.log
else
    echo -e "${model},train_model_$2,SUCCESS"
    echo "exit_code: 0.0" >> ../log/${model}/${model}_train_$2.log
fi
}
#train
train_model_multi(){
    python -m paddle.distributed.launch train.py \
       --config configs/hardnet/hardnet_cityscapes_1024x1024_160k.yml \
       --save_interval 100 \
       --iters 100 \
       --save_dir output/hardnet_cityscapes_1024x1024_160k \
       --learning_rate 0.005 \
       --seed 123 \
       --num_workers 0 \
       --batch_size=2 >../log/${model}/${model}_train_multi.log 2>&1
    print_info $? multi
}
train_model_single(){
    python train.py \
       --config configs/hardnet/hardnet_cityscapes_1024x1024_160k.yml \
       --save_interval 100 \
       --iters 100 \
       --save_dir output/hardnet_cityscapes_1024x1024_160k \
       --learning_rate 0.0025 \
       --seed 123 \
       --num_workers 0 \
       --batch_size=2 >../log/${model}/${model}_train_single.log 2>&1
    print_info $? single
}

cd $cur_path/../../PaddleSeg
if [ "$1" == 'single' ];then
train_model_single
else
train_model_multi
fi
