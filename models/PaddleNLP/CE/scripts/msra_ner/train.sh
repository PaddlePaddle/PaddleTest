#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/information_extraction/msra_ner/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1
MULTI=$2

if [[ ${MULTI} == "single" ]]; then
    python -u ./train.py \
        --model_name_or_path bert-base-multilingual-uncased \
        --max_seq_length 128 \
        --batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_steps 500 \
        --max_steps 1000 \
        --output_dir ./tmp/msra_ner/ \
        --device ${DEVICE} >$log_path/train_${MULTI}_${DEVICE}.log 2>&1
else
    python -m paddle.distributed.launch --gpus "$3" ./train.py \
        --model_name_or_path bert-base-multilingual-uncased \
        --max_seq_length 128 \
        --batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_steps 500 \
        --max_steps 1000 \
        --output_dir ./tmp/msra_ner/ \
        --device ${DEVICE} >$log_path/train_${MULTI}_${DEVICE}.log 2>&1
fi
#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
