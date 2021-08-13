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
code_path=$cur_path/../../models_repo/examples/dialogue/lic2021_baseline/
log_path=$root_path/log/$model_name/
mkdir -p $log_path

#访问RD程序
cd $code_path

if [[ $1 == "gpu" ]]; then
    python -m paddle.distributed.launch --gpus $3 --log_dir ./log finetune.py \
    --model_name_or_path=unified_transformer-12L-cn \
    --train_data_path=./datasets/train.txt \
    --valid_data_path=./datasets/valid.txt \
    --save_dir=./checkpoints \
    --logging_steps=500 \
    --save_steps=8000 \
    --seed=2021 \
    --epochs=1 \
    --batch_size=8192 \
    --lr=1e-5 \
    --weight_decay=0.01 \
    --warmup_steps=4000 \
    --max_grad_norm=0.1 \
    --sort_pool_size=65536 \
    --device=$1 > $log_path/finetune_$2_$1.log 2>&1
fi
