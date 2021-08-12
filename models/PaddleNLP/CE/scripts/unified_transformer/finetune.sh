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
code_path=$cur_path/../../models_repo/examples/dialogue/unified_transformer/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1

MULTI=$2
if [[ ${DEVICE} == "gpu" ]]; then
    # from paddlenlp.datasets import load_dataset
    # train_ds, dev_ds, test1_ds, test2_ds = load_dataset('duconv', splits=('train', 'dev', 'test_1', 'test_2'))
    python -m paddle.distributed.launch --gpus $3 --log_dir ./log finetune.py \
        --model_name_or_path=unified_transformer-12L-cn-luge \
        --save_dir=./checkpoints \
        --logging_steps=100 \
        --save_steps=1000 \
        --seed=2021 \
        --epochs=1 \
        --batch_size=16 \
        --lr=5e-5 \
        --weight_decay=0.01 \
        --warmup_steps=2500 \
        --max_grad_norm=0.1 \
        --max_seq_len=512 \
        --max_response_len=128 \
        --max_knowledge_len=256 \
        --device=${DEVICE} >$log_path/finetune_${MULTI}_${DEVICE}.log 2>&1
fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
