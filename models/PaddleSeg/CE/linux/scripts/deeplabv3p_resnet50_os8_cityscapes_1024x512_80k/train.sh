#!/bin/bash
# 外部传入参数
# $1 训练使用单卡or多卡

# export PYTHONPATH=`pwd`:$PYTHONPATH

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
    cat ../log/${model}/${model}_train_$2.log
else
    echo -e "${model},train_model_$2,SUCCESS"
    echo "exit_code: 0.0" >> ../log/${model}/${model}_train_$2.log
fi
}

#train
train_model_multi(){
    python -m paddle.distributed.launch train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --save_interval 1000 \
       --iters 1000 \
       --learning_rate 0.005 \
       --seed 123 \
       --num_workers 0 \
       --batch_size=2 \
       --save_dir output/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k >../log/${model}/${model}_train_multi.log 2>&1
    print_info $? multi
}
train_model_single(){
    python train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --save_interval 1000 \
       --iters 1000 \
       --learning_rate 0.0025 \
       --seed 123 \
       --num_workers 0 \
       --batch_size=2 \
       --save_dir output/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k >../log/${model}/${model}_train_single.log 2>&1
    print_info $? single
}

cd $cur_path/../../PaddleSeg
if [ "$1" == 'single' ];then
train_model_single
kill -9 `ps afx | grep train | awk '{print $1}'`
else
train_model_multi
kill -9 `ps afx | grep train | awk '{print $1}'`
fi
