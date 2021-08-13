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
code_path=$cur_path/../../PaddleSlim/demo/dygraph/pruning/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/PaddleSlim
#pip uninstall paddleslim
python -m pip install -r requirements.txt
python setup.py install

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


if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python -m paddle.distributed.launch  \
    --log_dir="prune_r34_f42_linux_dy_gpu2_dist_log" train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/prune_r34_f42_linux_dy_gpu2_dist_log $log_path/$2_dist_log

elif [ "$1" = "linux_dy_con_gpu2" ];then # 多卡 收敛性
    python -m paddle.distributed.launch  \
    --log_dir="prune_r34_f42_linux_dy_gpu2_dist_log" train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/prune_r34_f42_linux_dy_gpu2_dist_log $log_path/$2_dist_log

elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python train.py \
    --use_gpu=False \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
