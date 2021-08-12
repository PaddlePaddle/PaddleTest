#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_reading_comprehension/$model_name/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

python -m paddle.distributed.launch --gpus "$3" run_du.py \
    --model_type bert \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --max_steps 100 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./dureader-yesno/ \
    --device $1 \
    --max_step 10 > $log_path/finetune_$2_$1.log 2>&1
#cat $model_name-base_finetune.log
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
