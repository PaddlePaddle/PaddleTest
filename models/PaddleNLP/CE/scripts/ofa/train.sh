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
code_path=$cur_path/../../models_repo/examples/model_compression/ofa/
log_path=$root_path/log/$model_name/
data_path=$cur_path/../../models_repo/examples/benchmark/glue/tmp/$3/$2/
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

NAME=$(echo $3 | tr 'A-Z' 'a-z')

if [[ ${MULTI} == "multi" ]]; then
  python -m paddle.distributed.launch --gpus $4 run_glue_ofa.py \
    --model_type bert \
    --model_name_or_path $data_path/${NAME}_ft_model_10.pdparams \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 5 \
    --device $1 \
    --output_dir ./tmp/$3/$2/ \
    --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5  > $log_path/train_$3_$2_$1.log 2>&1
else
  python -u ./run_glue_ofa.py --model_type bert \
    --model_name_or_path $data_path/${NAME}_ft_model_10.pdparams \
    --task_name $3 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 5 \
    --device $1 \
    --output_dir ./tmp/$3/$2/ \
    --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5  > $log_path/train_$3_$2_$1.log 2>&1
fi
#cat $model_name-base_finetune.log
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
