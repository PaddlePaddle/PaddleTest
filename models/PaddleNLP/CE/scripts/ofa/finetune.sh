#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型finetune阶段"

#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/ofa/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

DEVICE=$1
if [[ ${DEVICE} == "gpu" ]]; then
N_GPU=1
else
N_GPU=0
fi
MULTI=$2
if [[ ${MULTI} == "multi" ]]; then
N_GPU=2
fi

cd ../../benchmark/glue/

python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 10   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --max_steps 10 \
    --save_steps 10 \
    --output_dir ./tmp/$3/$2/ \
    --device ${DEVICE}  > $log_path/finetune_$3_$2_$1.log 2>&1

#cat $model_name-base_finetune.log
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
